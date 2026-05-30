"""
단계 5e 결정성 고정 재평가 + 5b/5d/5e 공정 비교
==================================================
목적:
  (1) 5e multi-label 모델을 결정성 고정(set_deterministic)으로 재평가 →
      실행 간 재현 가능한 수치 확정 (포트폴리오/기록 인용용).
  (2) 5b(lora_mc) · 5d(lora_multitask_snr) · 5e(lora_multitask_ml) 세 모델을
      ★동일한 새 테스트셋(cpsc2018_mc_ml)★ 에서 평가해 공정 비교.

공정 비교 원칙:
  - 모든 모델을 같은 multi-hot 테스트셋(labels_mc N×5, labels_bin)에서 평가.
  - 핵심 지표 = per-class AUROC (threshold-free, softmax/sigmoid 무관하게 rank 비교).
  - per-class 확률은 각 모델의 native 출력 사용:
      5b/5d = softmax(5-class),  5e = sigmoid(독립 multi-label).
  - 응급 AUROC = mean(AUROC[AF], AUROC[허혈])  (5e 정의와 동일).
  - 이진: 전용 BinaryHead AUROC(5d/5e) + MC헤드 파생 응급점수(pAF+p허혈) AUROC(전 모델).
  - per-class F1 = val에서 클래스별 F1 최대화 임계값 튜닝 → test 적용(동일 절차).
    ※ 5b/5d는 단일라벨 학습이라 multi-label F1은 참고용(절차만 동일하게 맞춤).

사용법:
  python scripts/eval_mc_fair_compare.py
"""

import argparse
import math
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

try:
    from sklearn.metrics import roc_auc_score, f1_score, roc_curve
except ImportError:
    sys.exit("[오류] scikit-learn 미설치")

EMBED_DIM = 768
N_CLASSES = 5
CLASS_NAMES = ["정상(NSR)", "AF", "급성허혈", "전도장애", "이소성"]
EMERGENCY_CLASSES = (1, 2)  # AF + 급성허혈

CKPT_FM  = "D:/WidU_ecg-fm_emergency-detection/checkpoints/ecg-fm/mimic_iv_ecg_physionet_pretrained.pt"
DATA_DIR = "D:/WidU_ecg-fm_emergency-detection/data/processed/cpsc2018_mc_ml"

MODELS = [
    {"name": "5b lora_mc",
     "ckpt": "D:/WidU_ecg-fm_emergency-detection/outputs/lora_mc/lora_mc_best.pt",
     "mc_key": "head_state", "bin_key": None, "mc_act": "softmax"},
    {"name": "5d multitask_snr",
     "ckpt": "D:/WidU_ecg-fm_emergency-detection/outputs/lora_multitask_snr/lora_multitask_snr_best.pt",
     "mc_key": "head_mc_state", "bin_key": "head_bin_state", "mc_act": "softmax"},
    {"name": "5e multitask_ml",
     "ckpt": "D:/WidU_ecg-fm_emergency-detection/outputs/lora_multitask_ml/lora_multitask_ml_best.pt",
     "mc_key": "head_mc_state", "bin_key": "head_bin_state", "mc_act": "sigmoid"},
]


def set_deterministic(seed=42):
    """추론 재현성 고정 — cuDNN 비결정 알고리즘 끄고 시드 고정."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── LoRA ─────────────────────────────────────────────────────────────────
class LoRALinear(nn.Module):
    def __init__(self, linear, rank, alpha, dropout):
        super().__init__()
        self.original = linear
        self.original.weight.requires_grad_(False)
        if self.original.bias is not None:
            self.original.bias.requires_grad_(False)
        in_dim, out_dim = linear.in_features, linear.out_features
        self.lora_A = nn.Linear(in_dim, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_dim, bias=False)
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


def inject_lora(model, rank, alpha, dropout,
                target_suffixes=("self_attn.q_proj", "self_attn.v_proj")):
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):                   continue
        if not any(name.endswith(s) for s in target_suffixes):  continue
        parts = name.split(".")
        parent = model
        for p in parts[:-1]: parent = getattr(parent, p)
        setattr(parent, parts[-1], LoRALinear(module, rank, alpha, dropout))


def load_ecgfm(ckpt_path, device):
    from fairseq_signals.utils.checkpoint_utils import load_model_and_task
    result = load_model_and_task(ckpt_path)
    if isinstance(result, tuple):
        for r in result:
            if hasattr(r, "parameters"): return r.to(device)
            if isinstance(r, list) and r and hasattr(r[0], "parameters"): return r[0].to(device)
    return result.to(device)


def extract_embedding(backbone, x):
    out = backbone(source=x, padding_mask=None, features_only=True)
    return out["x"].mean(dim=1)


# ── 헤드 ─────────────────────────────────────────────────────────────────
class BinaryHead(nn.Module):
    def __init__(self, in_dim=EMBED_DIM):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)
    def forward(self, x): return self.fc(x).squeeze(-1)


class MulticlassHead(nn.Module):
    def __init__(self, in_dim=EMBED_DIM, n_classes=N_CLASSES):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)
    def forward(self, x): return self.fc(x)


# ── 데이터 (cpsc2018_mc_ml: multi-hot) ────────────────────────────────────
class MCDataset(Dataset):
    def __init__(self, split_dir):
        self.signals    = np.load(os.path.join(split_dir, "signals.npy"))
        self.labels_mc  = np.load(os.path.join(split_dir, "labels_mc.npy"))   # (N,5)
        self.labels_bin = np.load(os.path.join(split_dir, "labels_bin.npy"))  # (N,)
        self.record_ids = np.load(os.path.join(split_dir, "record_ids.npy"),
                                  allow_pickle=True)

    def __len__(self): return len(self.labels_mc)

    def __getitem__(self, idx):
        x  = torch.tensor(self.signals[idx],           dtype=torch.float32)
        yb = torch.tensor(float(self.labels_bin[idx]), dtype=torch.float32)
        ym = torch.tensor(self.labels_mc[idx],          dtype=torch.float32)
        return x, yb, ym


@torch.no_grad()
def infer(backbone, head_bin, head_mc, loader, device, mc_act):
    backbone.eval(); head_mc.eval()
    if head_bin is not None: head_bin.eval()
    bin_l, mc_l, all_yb, all_ym = [], [], [], []
    for x, yb, ym in loader:
        x = x.to(device)
        emb = extract_embedding(backbone, x)
        mc_l.append(head_mc(emb).cpu())
        if head_bin is not None:
            bin_l.append(head_bin(emb).cpu())
        all_yb.append(yb); all_ym.append(ym)
    mc_logits = torch.cat(mc_l).numpy()
    yb = torch.cat(all_yb).numpy().astype(int)
    ym = torch.cat(all_ym).numpy().astype(int)
    if mc_act == "softmax":
        e = np.exp(mc_logits - mc_logits.max(axis=1, keepdims=True))
        mc_probs = e / e.sum(axis=1, keepdims=True)
    else:  # sigmoid
        mc_probs = 1 / (1 + np.exp(-mc_logits))
    bin_probs = None
    if head_bin is not None:
        bin_logits = torch.cat(bin_l).numpy()
        bin_probs = 1 / (1 + np.exp(-bin_logits))
    return bin_probs, yb, mc_probs, ym


def tune_thresholds(mc_probs, mc_labels, grid=None):
    if grid is None:
        grid = np.linspace(0.05, 0.95, 19)
    ths = np.full(N_CLASSES, 0.5)
    for c in range(N_CLASSES):
        yc = mc_labels[:, c]
        if yc.sum() == 0 or yc.sum() == len(yc):
            continue
        best_f1, best_t = -1.0, 0.5
        for t in grid:
            f1 = f1_score(yc, (mc_probs[:, c] >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        ths[c] = best_t
    return ths


def sens_at_spec(y, p, spec=0.95):
    if len(np.unique(y)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y, p)
    s = 1 - fpr
    idx = np.searchsorted(s[::-1], spec)
    return float(tpr[::-1][idx]) if idx < len(tpr) else float("nan")


def evaluate(bin_probs, yb, mc_probs, ym, mc_th, mask=None):
    """mask: bool array (N,) — True인 레코드만 평가에 사용 (누수 제거 subset)."""
    if mask is not None:
        sel = mask
        bin_probs = bin_probs[sel] if bin_probs is not None else None
        yb = yb[sel]; mc_probs = mc_probs[sel]; ym = ym[sel]
    per_auroc, per_f1 = [], []
    for c in range(N_CLASSES):
        yc = ym[:, c]
        per_auroc.append(roc_auc_score(yc, mc_probs[:, c]) if 0 < yc.sum() < len(yc) else float("nan"))
        per_f1.append(f1_score(yc, (mc_probs[:, c] >= mc_th[c]).astype(int), zero_division=0))
    per_auroc = np.array(per_auroc); per_f1 = np.array(per_f1)
    emerg_score = mc_probs[:, 1] + mc_probs[:, 2]   # 파생 응급점수 (rank용)
    res = {
        "per_auroc":   per_auroc,
        "per_f1":      per_f1,
        "macro_auroc": float(np.nanmean(per_auroc)),
        "macro_f1":    float(np.mean(per_f1)),
        "emerg_auroc": float(np.nanmean(per_auroc[list(EMERGENCY_CLASSES)])),
        "emerg_f1":    float(np.mean(per_f1[list(EMERGENCY_CLASSES)])),
        "deriv_bin_auroc": roc_auc_score(yb, emerg_score) if len(np.unique(yb)) >= 2 else float("nan"),
        "bin_auroc":   float("nan"),
        "bin_sens":    float("nan"),
    }
    if bin_probs is not None and len(np.unique(yb)) >= 2:
        res["bin_auroc"] = roc_auc_score(yb, bin_probs)
        res["bin_sens"]  = sens_at_spec(yb, bin_probs, 0.95)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default=DATA_DIR)
    ap.add_argument("--old_data_dir",
        default="D:/WidU_ecg-fm_emergency-detection/data/processed/cpsc2018_mc",
        help="5b/5d 학습에 쓰인 구 데이터 — 누수 제거용 (train/val 레코드 제외)")
    ap.add_argument("--ckpt_path", default=CKPT_FM)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_deterministic(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 78)
    print("5e 결정성 고정 재평가 + 5b/5d/5e 공정 비교 (동일 테스트셋: cpsc2018_mc_ml)")
    print("=" * 78)
    print(f"device={device}, seed={args.seed} (cudnn.deterministic=True)")

    val_ds  = MCDataset(os.path.join(args.data_dir, "val"))
    test_ds = MCDataset(os.path.join(args.data_dir, "test"))
    val_loader  = DataLoader(val_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"val={len(val_ds)}, test={len(test_ds)}")
    # 테스트셋 클래스 분포
    tm = test_ds.labels_mc
    print("test multi-hot 양성 수: " +
          ", ".join(f"{CLASS_NAMES[c]}={int(tm[:, c].sum())}" for c in range(N_CLASSES)))
    print(f"test 이진 응급 양성: {int(test_ds.labels_bin.sum())}/{len(test_ds)}")

    # 누수 제거 mask: 새 test 중 구 train/val(=5b/5d 학습분)에 없는 레코드만 True
    clean_mask = np.ones(len(test_ds), dtype=bool)
    if os.path.isdir(args.old_data_dir):
        excl = set()
        for sp in ("train", "val"):
            rp = os.path.join(args.old_data_dir, sp, "record_ids.npy")
            if os.path.exists(rp):
                excl |= set(np.load(rp, allow_pickle=True).tolist())
        clean_mask = np.array([rid not in excl for rid in test_ds.record_ids], dtype=bool)
    n_clean = int(clean_mask.sum())
    tmc = test_ds.labels_mc[clean_mask]
    print(f"누수제거 clean subset: {n_clean}/{len(test_ds)} "
          f"(구 train/val 제외 — 세 모델 모두 미학습)")
    print("  clean 양성 수: " +
          ", ".join(f"{CLASS_NAMES[c]}={int(tmc[:, c].sum())}" for c in range(N_CLASSES)))
    print()

    print("[ECG-FM 백본 로드 + LoRA 주입 (rank=8)]")
    backbone = load_ecgfm(args.ckpt_path, device)
    for p in backbone.parameters(): p.requires_grad_(False)
    inject_lora(backbone, rank=8, alpha=16.0, dropout=0.0)
    print()

    results = {}
    for spec in MODELS:
        if not os.path.exists(spec["ckpt"]):
            print(f"[경고] 체크포인트 없음, 건너뜀: {spec['ckpt']}")
            continue
        ckpt = torch.load(spec["ckpt"], map_location=device)
        backbone.load_state_dict(ckpt["backbone_lora"], strict=False)

        head_mc = MulticlassHead().to(device)
        head_mc.load_state_dict(ckpt[spec["mc_key"]])
        head_bin = None
        if spec["bin_key"] and spec["bin_key"] in ckpt:
            head_bin = BinaryHead().to(device)
            head_bin.load_state_dict(ckpt[spec["bin_key"]])

        # val로 임계값 튜닝 → test 적용
        _, ybv, mpv, ymv = infer(backbone, head_bin, head_mc, val_loader, device, spec["mc_act"])
        mc_th = tune_thresholds(mpv, ymv)
        bpt, ybt, mpt, ymt = infer(backbone, head_bin, head_mc, test_loader, device, spec["mc_act"])
        res_full  = evaluate(bpt, ybt, mpt, ymt, mc_th, mask=None)
        res_clean = evaluate(bpt, ybt, mpt, ymt, mc_th, mask=clean_mask)
        results[spec["name"]] = {"full": res_full, "clean": res_clean,
                                 "mc_th": mc_th, "leak": spec["name"].startswith(("5b", "5d"))}
        print(f"  [{spec['name']:18s}] 평가 완료 (mc_act={spec['mc_act']})")

    names = list(results.keys())

    def print_block(scope, title, note):
        print()
        print("=" * 78)
        print(title)
        if note: print(note)
        print("=" * 78)
        print(f"{'클래스':<16}" + "".join(f"{n:>20}" for n in names))
        for c in range(N_CLASSES):
            row = f"{CLASS_NAMES[c]:<16}"
            for n in names:
                row += f"{results[n][scope]['per_auroc'][c]:>20.4f}"
            print(row)
        print("-" * 78)
        for key, label in [("macro_auroc", "macro-AUROC"),
                           ("emerg_auroc", "응급AUROC(AF·허혈)"),
                           ("bin_auroc",   "이진헤드AUROC"),
                           ("deriv_bin_auroc", "파생응급AUROC"),
                           ("bin_sens",    "이진Sens@95Sp")]:
            row = f"{label:<16}"
            for n in names:
                v = results[n][scope][key]
                row += f"{v:>20.4f}" if v == v else f"{'—':>20}"
            print(row)

    print_block("clean", "★ 공정 비교: per-class AUROC — clean subset (누수 제거, 세 모델 모두 미학습)",
                "  → 5b/5d/5e 직접 비교는 이 표만 유효")
    print_block("full", "[참고] full test per-class AUROC",
                "  ⚠️ 5b/5d는 이 test의 ~75%를 학습함(누수) → 5b/5d 열은 부풀려진 값. 비교 무효.\n"
                "  ✅ 5e 열만 유효한 공식 수치 (5e는 이 test 미학습).")

    print()
    print("=" * 78)
    print("[결정성 고정 5e 공식 수치 — full test 1024, 기록용]")
    if "5e multitask_ml" in results:
        r = results["5e multitask_ml"]["full"]; th = results["5e multitask_ml"]["mc_th"]
        print(f"  이진헤드 AUROC = {r['bin_auroc']:.4f}, Sens@95Sp = {r['bin_sens']:.4f}")
        print(f"  응급 AUROC(AF·허혈 평균) = {r['emerg_auroc']:.4f}")
        print(f"  macro AUROC = {r['macro_auroc']:.4f}, macro F1 = {r['macro_f1']:.4f}")
        print(f"  per-class AUROC: " +
              ", ".join(f"{CLASS_NAMES[c]}={r['per_auroc'][c]:.4f}" for c in range(N_CLASSES)))
        print(f"  per-class F1   : " +
              ", ".join(f"{CLASS_NAMES[c]}={r['per_f1'][c]:.4f}" for c in range(N_CLASSES)))
        print(f"  임계값: " + ", ".join(f"{t:.2f}" for t in th))
    print("=" * 78)


if __name__ == "__main__":
    main()
