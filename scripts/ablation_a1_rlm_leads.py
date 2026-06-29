"""
Ablation A1: RLM × Lead 수 교차 평가
======================================
목적:
  RLM(p=0.5)으로 학습한 LoRA 모델 vs RLM 없이(p=0) 학습한 LoRA 모델을
  12/4/2/1-lead 테스트 구성별로 평가해 RLM의 실제 가치를 입증.

핵심 가설:
  RLM 모델  → lead 수 감소 시 완만히 하락 (강건)
  비-RLM 모델 → lead 수 감소 시 급락 (취약)
  → 이것이 '웨어러블 소수 lead 응급감지' 논지의 핵심 증거

실행 순서:
  # 1단계: RLM 없는 모델 학습 (~30분)
  python scripts/train_lora.py --rlm_p 0 \\
      --out_dir outputs/lora_no_rlm

  # 2단계: 교차 평가
  python scripts/ablation_a1_rlm_leads.py \\
      --rlm_ckpt   outputs/lora/lora_best.pt \\
      --no_rlm_ckpt outputs/lora_no_rlm/lora_best.pt

선택 인수:
  --baseline_ckpt : 베이스라인(linear probe) baseline_best.pt 경로 (옵션)
  --bootstrap_n   : bootstrap 반복 수 (기본 1000, 0=생략)
"""

import argparse
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:
    from sklearn.metrics import f1_score, roc_auc_score, roc_curve
except ImportError:
    sys.exit("[오류] scikit-learn 미설치 — pip install scikit-learn")

EMBED_DIM = 768
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


class LinearHead(nn.Module):
    """train_baseline.py / train_lora.py와 동일 구조 (fc.weight/fc.bias 키)."""

    def __init__(self, in_dim: int = EMBED_DIM):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, x):
        return self.fc(x).squeeze(-1)


LEAD_CONFIGS = [
    ("12-lead (전체)", list(range(12))),
    ("RA/LA/LL+V2 (7유도파생)", [0, 1, 2, 3, 4, 5, 7]),  # 전극3개→사지6유도 파생 + V2
    ("3-lead (I,II,V2 독립)", [0, 1, 7]),  # 위 구성의 독립 정보만
    ("4-lead (I,II,V2,V5)", [0, 1, 7, 10]),
    ("2-lead (I,II)", [0, 1]),
    ("1-lead (II)", [1]),
]


# ── LoRA (train_lora.py와 동일) ───────────────────────────────────────


class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, rank: int, alpha: float, dropout: float):
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
    def bias(self):
        return self.original.bias

    @property
    def weight(self):
        return self.original.weight

    def forward(self, x):
        return (
            self.original(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        )


def inject_lora(
    model,
    rank,
    alpha,
    dropout,
    target_suffixes=("self_attn.q_proj", "self_attn.v_proj"),
):
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(name.endswith(s) for s in target_suffixes):
            continue
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], LoRALinear(module, rank, alpha, dropout))


# ── 모델 로드 ─────────────────────────────────────────────────────────


def load_ecgfm(ckpt_path, device):
    from fairseq_signals.utils.checkpoint_utils import load_model_and_task

    result = load_model_and_task(ckpt_path)
    if isinstance(result, tuple):
        for r in result:
            if hasattr(r, "parameters"):
                return r.to(device)
            if isinstance(r, list) and r and hasattr(r[0], "parameters"):
                return r[0].to(device)
    return result.to(device)


def load_lora_model(backbone_ckpt, lora_ckpt, device):
    """ECG-FM 로드 → LoRA 주입 → lora_best.pt 가중치 복원."""
    backbone = load_ecgfm(backbone_ckpt, device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    ckpt = torch.load(lora_ckpt, map_location=device)
    rank = ckpt.get("lora_rank", 8)
    alpha = ckpt.get("lora_alpha", 16.0)

    inject_lora(backbone, rank=rank, alpha=alpha, dropout=0.0)
    backbone.load_state_dict(ckpt["backbone_lora"], strict=False)

    head = LinearHead().to(device)
    head.load_state_dict(ckpt["head_state"])
    head.eval()
    return backbone, head


def load_baseline_model(baseline_ckpt, device):
    """베이스라인 linear head만 복원 (backbone은 별도 로드 필요)."""
    ckpt = torch.load(baseline_ckpt, map_location=device)
    head = LinearHead().to(device)
    head.load_state_dict(ckpt["head_state"])
    head.eval()
    return head


# ── 데이터 ───────────────────────────────────────────────────────────


class CPSCTestDataset(Dataset):
    def __init__(self, test_dir):
        self.signals = np.load(os.path.join(test_dir, "signals.npy"))
        self.labels = np.load(os.path.join(test_dir, "labels.npy"))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.signals[idx], dtype=torch.float32),
            torch.tensor(float(self.labels[idx]), dtype=torch.float32),
        )


def apply_lead_mask(x: torch.Tensor, available: list) -> torch.Tensor:
    """x: (B,12,T). available에 없는 lead를 0-fill."""
    masked = x.clone()
    for i in range(12):
        if i not in available:
            masked[:, i, :] = 0.0
    return masked


# ── 평가 ─────────────────────────────────────────────────────────────


@torch.no_grad()
def compute_metrics(backbone, head, dataset, available_leads, device, batch_size=32):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_logits, all_labels = [], []

    for x, y in loader:
        x = apply_lead_mask(x.to(device), available_leads)
        out = backbone(source=x, padding_mask=None, features_only=True)
        emb = out["x"].mean(dim=1)  # (B,768)
        all_logits.append(head(emb).squeeze(-1).cpu())
        all_labels.append(y)

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy().astype(int)
    probs = 1 / (1 + np.exp(-logits))

    auroc = roc_auc_score(labels, probs)
    f1 = f1_score(labels, (probs >= 0.5).astype(int), zero_division=0)
    fpr, tpr, _ = roc_curve(labels, probs)
    spec = 1 - fpr
    idx = np.searchsorted(spec[::-1], 0.95)
    sens = float(tpr[::-1][idx]) if idx < len(tpr) else float("nan")

    return auroc, f1, sens, probs, labels


def bootstrap_ci(probs, labels, n=1000, seed=0):
    """AUROC와 Sens@95Sp의 95% CI를 bootstrap으로 계산."""
    rng = np.random.RandomState(seed)
    aurocs, senss = [], []
    n_samples = len(labels)
    for _ in range(n):
        idx = rng.randint(0, n_samples, n_samples)
        p_, l_ = probs[idx], labels[idx]
        if len(np.unique(l_)) < 2:
            continue
        a = roc_auc_score(l_, p_)
        fpr, tpr, _ = roc_curve(l_, p_)
        spec = 1 - fpr
        si = np.searchsorted(spec[::-1], 0.95)
        s = float(tpr[::-1][si]) if si < len(tpr) else np.nan
        aurocs.append(a)
        senss.append(s)
    auroc_ci = (np.percentile(aurocs, 2.5), np.percentile(aurocs, 97.5))
    sens_ci = (np.nanpercentile(senss, 2.5), np.nanpercentile(senss, 97.5))
    return auroc_ci, sens_ci


# ── 메인 ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default="data/processed/cpsc2018")
    parser.add_argument(
        "--ckpt_path", default="checkpoints/ecg-fm/mimic_iv_ecg_physionet_pretrained.pt"
    )
    parser.add_argument("--rlm_ckpt", default="outputs/lora/lora_best.pt")
    parser.add_argument(
        "--no_rlm_ckpt",
        default=None,
        help="RLM 없이 학습한 LoRA 체크포인트 (없으면 RLM 모델만 평가)",
    )
    parser.add_argument(
        "--baseline_ckpt",
        default=None,
        help="베이스라인 linear probe 체크포인트 (옵션)",
    )
    parser.add_argument(
        "--bootstrap_n", type=int, default=1000, help="bootstrap 반복 수 (0=생략)"
    )
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_ds = CPSCTestDataset(os.path.join(args.data_dir, "test"))

    print("=" * 72)
    print("Ablation A1: RLM × Lead 수 교차 평가")
    print("=" * 72)
    print(f"테스트 세트: {len(test_ds)} records (device={device})")
    print()

    # ── 모델 로드 ─────────────────────────────────────────────────────
    models = {}

    print("[모델 로드]")
    print(f"  RLM (p=0.5): {args.rlm_ckpt}")
    rlm_bb, rlm_head = load_lora_model(args.ckpt_path, args.rlm_ckpt, device)
    models["LoRA+RLM"] = (rlm_bb, rlm_head)

    if args.no_rlm_ckpt and os.path.exists(args.no_rlm_ckpt):
        print(f"  No-RLM (p=0): {args.no_rlm_ckpt}")
        no_bb, no_head = load_lora_model(args.ckpt_path, args.no_rlm_ckpt, device)
        models["LoRA (no RLM)"] = (no_bb, no_head)
    else:
        print("  No-RLM 체크포인트 없음 — RLM 모델만 평가")
        print(
            "  학습 방법: python scripts/train_lora.py --rlm_p 0 "
            "--out_dir .../outputs/lora_no_rlm"
        )

    if args.baseline_ckpt and os.path.exists(args.baseline_ckpt):
        print(f"  베이스라인: {args.baseline_ckpt}")
        base_head = load_baseline_model(args.baseline_ckpt, device)
        # 베이스라인도 동일 backbone 사용 (frozen, no LoRA)
        base_bb = load_ecgfm(args.ckpt_path, device)
        base_bb.eval()
        for p in base_bb.parameters():
            p.requires_grad_(False)
        models["Baseline (linear)"] = (base_bb, base_head)
    print()

    # ── 평가 루프 ─────────────────────────────────────────────────────
    results = {}  # {model_name: {config_name: (auroc, f1, sens)}}

    for model_name, (bb, head) in models.items():
        results[model_name] = {}
        print(f"[{model_name}] 평가 중...")
        for cfg_name, available in LEAD_CONFIGS:
            auroc, f1, sens, probs, labels = compute_metrics(
                bb, head, test_ds, available, device, args.batch_size
            )
            results[model_name][cfg_name] = (auroc, f1, sens, probs, labels)
            print(
                f"  {cfg_name:25s} AUROC={auroc:.4f}  F1={f1:.4f}  Sens@95Sp={sens:.4f}"
            )
        print()

    # ── 결과 테이블 ───────────────────────────────────────────────────
    col_w = 20
    model_names = list(models.keys())

    print("=" * 72)
    print("결과 비교표 (AUROC / Sens@95Sp)")
    print("=" * 72)

    header = f"{'Lead 구성':<28}"
    for mn in model_names:
        header += f"  {mn[:col_w]:<{col_w}}"
    print(header)
    print("-" * 72)

    for cfg_name, _ in LEAD_CONFIGS:
        row = f"{cfg_name:<28}"
        for mn in model_names:
            auroc, _, sens, _, _ = results[mn][cfg_name]
            row += f"  {auroc:.4f}/{sens:.4f}       "
        print(row)

    # 12-lead 대비 하락폭 (핵심 지표)
    print()
    print("12-lead 대비 AUROC 하락폭 (↓클수록 취약)")
    print("-" * 72)
    for cfg_name, _ in LEAD_CONFIGS:
        if cfg_name == "12-lead (전체)":
            continue
        row = f"{cfg_name:<28}"
        for mn in model_names:
            base_auroc = results[mn]["12-lead (전체)"][0]
            curr_auroc = results[mn][cfg_name][0]
            delta = curr_auroc - base_auroc
            row += f"  {delta:+.4f}               "
        print(row)

    # ── Bootstrap CI (12-lead, 선택) ──────────────────────────────────
    if args.bootstrap_n > 0:
        print()
        print(f"Bootstrap 95% CI (12-lead, n={args.bootstrap_n})")
        print("-" * 72)
        for mn in model_names:
            _, _, _, probs, labels = results[mn]["12-lead (전체)"]
            auroc_ci, sens_ci = bootstrap_ci(probs, labels, n=args.bootstrap_n)
            print(f"  {mn}")
            print(f"    AUROC:    {auroc_ci[0]:.4f} ~ {auroc_ci[1]:.4f}")
            print(f"    Sens@95Sp: {sens_ci[0]:.4f} ~ {sens_ci[1]:.4f}")

    print()
    print("=" * 72)
    print("다음 단계:")
    print("  → 결과를 decisions.md에 기록")
    print("  → Ablation A2: LoRA rank sweep {4, 8, 16}")
    print("  → 단계 7: 신호 품질 게이트")
    print("=" * 72)


if __name__ == "__main__":
    main()
