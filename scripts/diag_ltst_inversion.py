"""
LTST 역전 원인 진단 (diag_ltst_inversion.py)
==============================================
목적:
  LTST AUROC < 0.5 (역전) 패턴의 근본 원인을 3가지 테스트로 진단.

  [Test A] CPSC 2-lead 제한 테스트
    - ③ 모델을 CPSC mc test에 적용, slot 1·7 만 남기고 나머지 0-fill
    - AF subgroup vs 허혈 subgroup AUROC 비교
    - 가설: 허혈 AUROC가 2-lead 제한 시 크게 떨어지면 "lead 국소성" 가설 확인

  [Test B] LTST 점수 분포 분석
    - ③ 모델 LTST 전체 추론 → 클래스별 score 분포 출력
    - 가설: ischemic 창에 낮은 score, normal 창에 높은 score → 폴라리티 역전

  [Test C] 신호 통계 비교 (LTST vs CPSC)
    - LTST / CPSC 신호의 활성 lead (slot 1, 7) mV 스케일·분포 비교
    - 가설: mV 스케일 차이가 크면 전처리/스케일 불일치 원인

사용법:
  python scripts/diag_ltst_inversion.py
  python scripts/diag_ltst_inversion.py --tests A B   (특정 테스트만)
"""

import argparse
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    from sklearn.metrics import roc_auc_score, roc_curve
except ImportError:
    sys.exit("[오류] scikit-learn 미설치")

# ── 경로 ────────────────────────────────────────────────────────────────
CKPT_FM        = "D:/WidU_ecg-fm_emergency-detection/checkpoints/ecg-fm/mimic_iv_ecg_physionet_pretrained.pt"
OUTPUTS        = "D:/WidU_ecg-fm_emergency-detection/outputs"
PROCESSED      = "D:/WidU_ecg-fm_emergency-detection/data/processed"
CKPT_MULTISNR  = f"{OUTPUTS}/lora_multisnr/lora_multisnr_best.pt"   # ③ 모델

CPSC_MC_TEST   = f"{PROCESSED}/cpsc2018_mc/test"
LTST_DIR       = f"{PROCESSED}/ltst"

EMBED_DIM = 768
BATCH     = 64

# LTST에서 사용한 lead slot (0-indexed): ML2→index 1 (Lead II), MV2→index 7 (Lead V2)
LTST_ACTIVE_SLOTS = [1, 7]


# ── LoRA 모듈 ────────────────────────────────────────────────────────────
class LoRALinear(nn.Module):
    def __init__(self, linear, rank, alpha, dropout=0.0):
        super().__init__()
        self.original = linear
        self.original.weight.requires_grad_(False)
        if self.original.bias is not None:
            self.original.bias.requires_grad_(False)
        in_dim, out_dim = linear.in_features, linear.out_features
        self.lora_A  = nn.Linear(in_dim, rank, bias=False)
        self.lora_B  = nn.Linear(rank, out_dim, bias=False)
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    @property
    def bias(self):   return self.original.bias
    @property
    def weight(self): return self.original.weight

    def forward(self, x):
        return self.original(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling


def inject_lora(model, rank, alpha, dropout=0.0,
                target_suffixes=("self_attn.q_proj", "self_attn.v_proj")):
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear): continue
        if not any(name.endswith(s) for s in target_suffixes): continue
        parts = name.split(".")
        parent = model
        for p in parts[:-1]: parent = getattr(parent, p)
        setattr(parent, parts[-1], LoRALinear(module, rank, alpha, dropout))


# ── 모델 로드 ────────────────────────────────────────────────────────────
def load_backbone(device):
    from fairseq_signals.utils.checkpoint_utils import load_model_and_task
    result = load_model_and_task(CKPT_FM)
    if isinstance(result, tuple):
        for r in result:
            if hasattr(r, "parameters"): return r.to(device)
            if isinstance(r, list) and r and hasattr(r[0], "parameters"): return r[0].to(device)
    return result.to(device)


def load_lora_multisnr(backbone, device):
    ckpt = torch.load(CKPT_MULTISNR, map_location=device)
    rank  = ckpt.get("lora_rank", 8)
    alpha = ckpt.get("lora_alpha", 16.0)
    inject_lora(backbone, rank=rank, alpha=alpha, dropout=0.0)
    backbone.load_state_dict(ckpt["backbone_lora"], strict=False)
    backbone.eval()
    head = nn.Linear(EMBED_DIM, 1).to(device)
    head_state = ckpt["head_state"]
    # head_state 키가 'fc.weight'/'fc.bias' 형태일 수 있음
    mapped = {}
    for k, v in head_state.items():
        mapped[k.replace("fc.", "")] = v
    head.load_state_dict(mapped)
    head.eval()
    return backbone, head


@torch.no_grad()
def get_scores(backbone, head, signals_np, device, lead_mask=None):
    """signals_np: (N,12,5000). lead_mask: 살릴 slot 인덱스 리스트 (None=전체)"""
    all_scores = []
    N = len(signals_np)
    for start in range(0, N, BATCH):
        x = torch.tensor(signals_np[start:start+BATCH], dtype=torch.float32).to(device)
        if lead_mask is not None:
            # 지정 slot 외 모두 0-fill
            mask = torch.zeros(12, dtype=torch.bool)
            mask[lead_mask] = True
            x[:, ~mask, :] = 0.0
        emb = backbone(source=x, padding_mask=None, features_only=True)["x"].mean(dim=1)
        logits = head(emb).squeeze(-1)
        probs  = torch.sigmoid(logits)
        all_scores.append(probs.cpu().numpy())
    return np.concatenate(all_scores)


def auroc_safe(labels, scores):
    if len(np.unique(labels)) < 2:
        return float("nan")
    return roc_auc_score(labels, scores)


# ── 공통 출력 ─────────────────────────────────────────────────────────────
def hline(): print("-" * 60)


# ═══════════════════════════════════════════════════════════════
# Test A: CPSC 2-lead 제한 — AF vs 허혈 subgroup
# ═══════════════════════════════════════════════════════════════
def test_a(backbone, head, device):
    print("\n" + "=" * 60)
    print("[Test A] CPSC mc test — 2-lead 제한 시 AF vs 허혈 AUROC")
    print("=" * 60)

    sigs   = np.load(os.path.join(CPSC_MC_TEST, "signals.npy"))   # (N,12,5000)
    labels = np.load(os.path.join(CPSC_MC_TEST, "labels.npy"))    # 0~4
    # 5-class: 0=NSR, 1=AF, 2=허혈STD/STE, 3=전도, 4=이소성
    # binary: NSR(0) vs AF+허혈(1,2) 만 사용 (전도·이소성 제외)
    keep  = (labels == 0) | (labels == 1) | (labels == 2)
    sigs_k   = sigs[keep]
    labels_k = labels[keep]
    binary_k = (labels_k > 0).astype(int)   # AF+허혈=1, NSR=0

    n_nsr = (labels_k == 0).sum()
    n_af  = (labels_k == 1).sum()
    n_isch= (labels_k == 2).sum()
    print(f"  사용 샘플: NSR={n_nsr}, AF={n_af}, 허혈={n_isch}")

    configs = {
        "12-lead (전체)":       None,
        "2-lead (slot 1,7)":   LTST_ACTIVE_SLOTS,
    }

    for cfg_name, mask in configs.items():
        scores = get_scores(backbone, head, sigs_k, device, lead_mask=mask)
        # 전체 이진
        auroc_all   = auroc_safe(binary_k, scores)
        # AF vs NSR
        sel_af  = (labels_k == 0) | (labels_k == 1)
        auroc_af= auroc_safe((labels_k[sel_af] == 1).astype(int), scores[sel_af])
        # 허혈 vs NSR
        sel_isch= (labels_k == 0) | (labels_k == 2)
        auroc_is= auroc_safe((labels_k[sel_isch] == 2).astype(int), scores[sel_isch])

        print(f"\n  [{cfg_name}]")
        print(f"    전체 이진 AUROC       : {auroc_all:.4f}")
        print(f"    AF vs NSR AUROC       : {auroc_af:.4f}")
        print(f"    허혈 vs NSR AUROC     : {auroc_is:.4f}")
        if mask is not None:
            print(f"  → lead 제한 시 허혈 AUROC 하락이 AF보다 크면 lead 국소성 확인")


# ═══════════════════════════════════════════════════════════════
# Test B: LTST 점수 분포
# ═══════════════════════════════════════════════════════════════
def test_b(backbone, head, device):
    print("\n" + "=" * 60)
    print("[Test B] LTST 점수 분포 — 클래스별 score 통계")
    print("=" * 60)

    sigs   = np.load(os.path.join(LTST_DIR, "signals.npy"))
    labels = np.load(os.path.join(LTST_DIR, "labels.npy")).astype(int)
    n_isch = (labels == 1).sum(); n_norm = (labels == 0).sum()
    print(f"  총 {len(labels)}개  허혈에피소드(1)={n_isch}  정상(0)={n_norm}")

    scores = get_scores(backbone, head, sigs, device)
    auroc  = auroc_safe(labels, scores)
    print(f"\n  AUROC           : {auroc:.4f}  (역전이면 <0.5)")
    print(f"  1-AUROC (flip)  : {1-auroc:.4f}  (score 반전 시 예상 AUROC)")

    for cls, name in [(1, "허혈(1)"), (0, "정상(0)")]:
        s = scores[labels == cls]
        print(f"\n  [{name}]  n={len(s)}")
        print(f"    mean  : {s.mean():.4f}")
        print(f"    std   : {s.std():.4f}")
        print(f"    min   : {s.min():.4f}")
        print(f"    p25   : {np.percentile(s,25):.4f}")
        print(f"    p50   : {np.percentile(s,50):.4f}")
        print(f"    p75   : {np.percentile(s,75):.4f}")
        print(f"    max   : {s.max():.4f}")

    isch_mean = scores[labels == 1].mean()
    norm_mean = scores[labels == 0].mean()
    print(f"\n  허혈 평균 score  : {isch_mean:.4f}")
    print(f"  정상 평균 score  : {norm_mean:.4f}")
    if isch_mean < norm_mean:
        print("  ★ 허혈 < 정상 → 명확한 폴라리티 역전 확인")
    else:
        print("  ※ 허혈 >= 정상 → 폴라리티 역전 아님 (다른 원인 탐색 필요)")

    # 점수 분포 히스토그램 (텍스트)
    print("\n  점수 히스토그램 (bin=0.1):")
    print(f"  {'구간':12s}  {'허혈(1)':>10}  {'정상(0)':>10}")
    for b in range(10):
        lo, hi = b*0.1, (b+1)*0.1
        n1 = ((scores[labels==1] >= lo) & (scores[labels==1] < hi)).sum()
        n0 = ((scores[labels==0] >= lo) & (scores[labels==0] < hi)).sum()
        bar1 = "█" * (n1 * 30 // max(n_isch, 1))
        bar0 = "█" * (n0 * 30 // max(n_norm, 1))
        print(f"  [{lo:.1f}-{hi:.1f})  {n1:>5} {bar1:<20}  {n0:>5} {bar0:<20}")


# ═══════════════════════════════════════════════════════════════
# Test C: 신호 통계 비교 (LTST vs CPSC)
# ═══════════════════════════════════════════════════════════════
def test_c():
    print("\n" + "=" * 60)
    print("[Test C] 신호 통계 비교 — LTST vs CPSC (스케일·폴라리티)")
    print("=" * 60)

    ltst_sigs  = np.load(os.path.join(LTST_DIR, "signals.npy"))
    cpsc_sigs  = np.load(os.path.join(PROCESSED, "cpsc2018/test/signals.npy"))
    ltst_labels= np.load(os.path.join(LTST_DIR, "labels.npy")).astype(int)

    print(f"\n  LTST  shape: {ltst_sigs.shape}  dtype: {ltst_sigs.dtype}")
    print(f"  CPSC  shape: {cpsc_sigs.shape}  dtype: {cpsc_sigs.dtype}")

    for db_name, sigs, lbl in [
        ("LTST  ischemic", ltst_sigs[ltst_labels==1], None),
        ("LTST  normal  ", ltst_sigs[ltst_labels==0], None),
        ("CPSC  emerg   ", cpsc_sigs[np.load(os.path.join(PROCESSED,"cpsc2018/test/labels.npy"))==1], None),
        ("CPSC  normal  ", cpsc_sigs[np.load(os.path.join(PROCESSED,"cpsc2018/test/labels.npy"))==0], None),
    ]:
        # 활성 lead만 (0-fill 아닌 lead)
        active = sigs[:, LTST_ACTIVE_SLOTS, :]   # (N,2,5000)
        abs_vals = np.abs(active[active != 0])
        if len(abs_vals) == 0:
            print(f"\n  [{db_name}] 활성 신호 없음")
            continue
        print(f"\n  [{db_name}]  n={len(sigs)}")
        print(f"    |signal| mean : {abs_vals.mean():.6f}")
        print(f"    |signal| std  : {abs_vals.std():.6f}")
        print(f"    |signal| p50  : {np.percentile(abs_vals,50):.6f}")
        print(f"    |signal| p95  : {np.percentile(abs_vals,95):.6f}")
        print(f"    |signal| max  : {abs_vals.max():.6f}")
        # 부호
        raw = active.flatten()
        raw_nz = raw[raw != 0]
        print(f"    부호 분포: 양={( raw_nz>0).mean()*100:.1f}%  음={( raw_nz<0).mean()*100:.1f}%")

    # CPSC에서 slot 1,7 이외 lead의 비율 확인 (0-fill 없음 — 항상 12-lead)
    print(f"\n  ※ CPSC는 12-lead 모두 유효. LTST는 slot 1,7만 유효, 나머지 0.")
    zero_ratio = (ltst_sigs[:, [i for i in range(12) if i not in LTST_ACTIVE_SLOTS], :] == 0).mean()
    print(f"  LTST 비활성 lead 0 비율: {zero_ratio*100:.1f}%")


# ── main ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tests", nargs="+", default=["A", "B", "C"],
                        choices=["A", "B", "C"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("LTST 역전 원인 진단")
    print("=" * 60)
    print(f"디바이스: {device}")
    print(f"실행 테스트: {args.tests}")

    # Test C는 모델 불필요
    if "C" in args.tests:
        test_c()

    if "A" in args.tests or "B" in args.tests:
        print("\n[모델 로드] ECG-FM + LoRA+multi-SNR (③) ...")
        backbone = load_backbone(device)
        backbone, head = load_lora_multisnr(backbone, device)
        print("  완료")

        if "A" in args.tests:
            test_a(backbone, head, device)
        if "B" in args.tests:
            test_b(backbone, head, device)

    print("\n" + "=" * 60)
    print("진단 완료")
    print("=" * 60)


if __name__ == "__main__":
    main()
