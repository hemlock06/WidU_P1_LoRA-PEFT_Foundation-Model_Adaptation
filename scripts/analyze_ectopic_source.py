"""
이소성 클래스 출처별 분해 분석 — 5e 데이터 수정이 왜 이소성을 개선 못 했나
=============================================================================
질문: 164884008(심실이소성) 603개 복구가 이소성 AUROC를 오히려 낮춘 원인은?
가설: 복구한 164884008 레코드가 (a) 본질적으로 더 어렵거나 (b) 동반진단이 많아
      이소성 클래스 경계를 흐렸다.

검증:
  이소성 양성 test 레코드를 코드 출처로 분해 —
    · PAC/PVC 전통 이소성 (284470004/63593006/427172004/17338001)
    · 164884008 심실이소성 (5e에서 신규 복구분)
  그리고 5e의 이소성 확률로 출처별 AUROC를 따로 계산.
  PAC/PVC AUROC ≫ 164884008 AUROC 이면 → 복구분이 어려운 하위분포라는 증거.
  추가로 동반진단(multi-label row sum>1) 비율도 출처별 비교.

  ※ 5e는 full test 미학습 → 유효. 5b/5d는 누수(참고만).

사용법:
  python scripts/analyze_ectopic_source.py
"""

import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from eval_mc_fair_compare import (
    set_deterministic, inject_lora, load_ecgfm, BinaryHead, MulticlassHead,
    MCDataset, infer, MODELS, CKPT_FM, DATA_DIR, N_CLASSES, CLASS_NAMES,
)

RAW_DIR = "data/raw/cpsc2018"
ECTO_IDX = 4

SNOMED_PACPVC = {284470004, 63593006, 427172004, 17338001}
SNOMED_VE     = {164884008}


def parse_dx_from_hea(hea_path):
    codes = set()
    try:
        with open(hea_path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") and "Dx" in line and ":" in line:
                    _, _, rest = line.partition(":")
                    for tok in rest.split(","):
                        tok = tok.strip()
                        if tok.isdigit():
                            codes.add(int(tok))
    except Exception:
        pass
    return codes


def scan_hea(raw_dir):
    m = {}
    for root, _, files in os.walk(raw_dir):
        for f in files:
            if f.endswith(".hea"):
                m[f[:-4]] = os.path.join(root, f)
    return m


def group_auroc(probs, pos_mask, neg_mask):
    """pos_mask 양성 vs neg_mask 음성만으로 AUROC."""
    sel = pos_mask | neg_mask
    y = pos_mask[sel].astype(int)
    p = probs[sel]
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan"), int(pos_mask.sum())
    return roc_auc_score(y, p), int(pos_mask.sum())


def main():
    set_deterministic(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_ds = MCDataset(os.path.join(DATA_DIR, "test"))
    rids = list(test_ds.record_ids)
    ym   = test_ds.labels_mc
    ecto = ym[:, ECTO_IDX].astype(bool)

    print("=" * 74)
    print("이소성 출처별 분해 분석 (cpsc2018_mc_ml test)")
    print("=" * 74)

    hea = scan_hea(RAW_DIR)
    has_ve = np.zeros(len(rids), dtype=bool)
    has_pp = np.zeros(len(rids), dtype=bool)
    missing = 0
    for i, rid in enumerate(rids):
        if rid not in hea:
            missing += 1
            continue
        codes = parse_dx_from_hea(hea[rid])
        has_ve[i] = bool(codes & SNOMED_VE)
        has_pp[i] = bool(codes & SNOMED_PACPVC)
    if missing:
        print(f"[참고] .hea 미발견 {missing}건")

    n_ecto = int(ecto.sum())
    ve_pos = ecto & has_ve
    pp_pos = ecto & has_pp
    pp_pure = ecto & has_pp & ~has_ve      # 순수 PAC/PVC
    ve_any  = ecto & has_ve                # 164884008 포함
    comorbid = (ym.sum(axis=1) > 1)

    print(f"\n이소성 양성 총: {n_ecto}")
    print(f"  · PAC/PVC 보유:        {int(pp_pos.sum())}")
    print(f"  · 164884008 보유:      {int(ve_any.sum())}")
    print(f"  · PAC/PVC 순수(VE 없음): {int(pp_pure.sum())}")
    print(f"  · 둘 다 보유:           {int((ecto & has_pp & has_ve).sum())}")
    print(f"\n동반진단(≥2 클래스) 비율:")
    for name, mask in [("PAC/PVC 순수", pp_pure), ("164884008 보유", ve_any)]:
        m = mask.sum()
        c = (mask & comorbid).sum()
        print(f"  {name:16s}: {int(c)}/{int(m)} = {100*c/max(m,1):.1f}% 동반")

    # 모델별 이소성 확률 → 출처별 AUROC
    print("\n[ECG-FM + LoRA — 모델별 이소성 확률 추출]")
    backbone = load_ecgfm(CKPT_FM, device)
    for p in backbone.parameters(): p.requires_grad_(False)
    inject_lora(backbone, rank=8, alpha=16.0, dropout=0.0)
    loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

    ecto_neg = ~ecto
    print()
    print(f"{'모델':<20}{'PAC/PVC순수 AUROC':>20}{'164884008 AUROC':>20}{'(전체이소성)':>14}")
    print("-" * 74)
    for spec in MODELS:
        if not os.path.exists(spec["ckpt"]):
            continue
        ckpt = torch.load(spec["ckpt"], map_location=device)
        backbone.load_state_dict(ckpt["backbone_lora"], strict=False)
        head_mc = MulticlassHead().to(device)
        head_mc.load_state_dict(ckpt[spec["mc_key"]])
        head_bin = None
        if spec["bin_key"] and spec["bin_key"] in ckpt:
            head_bin = BinaryHead().to(device)
            head_bin.load_state_dict(ckpt[spec["bin_key"]])
        _, _, mc_probs, _ = infer(backbone, head_bin, head_mc, loader, device, spec["mc_act"])
        ep = mc_probs[:, ECTO_IDX]
        a_pp, n_pp = group_auroc(ep, pp_pure, ecto_neg)
        a_ve, n_ve = group_auroc(ep, ve_any, ecto_neg)
        a_all = roc_auc_score(ecto.astype(int), ep)
        leak = "" if spec["name"].startswith("5e") else "  (누수)"
        print(f"{spec['name']:<20}{a_pp:>14.4f}({n_pp:3d}){a_ve:>13.4f}({n_ve:3d}){a_all:>14.4f}{leak}")

    print("-" * 74)
    print("해석 가이드: PAC/PVC AUROC ≫ 164884008 AUROC 이면,")
    print("  복구한 164884008이 어려운 하위분포 → 이소성 클래스 확장이 경계를 흐림.")
    print("=" * 74)


if __name__ == "__main__":
    main()
