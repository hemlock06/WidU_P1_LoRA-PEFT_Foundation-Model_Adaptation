"""
단계 5: ECG-FM 베이스라인 선형 프로빙
======================================
목적:
  ECG-FM 백본을 완전 동결(freeze)하고 선형 분류 헤드만 학습.
  LoRA fine-tuning(단계 6) 전 Plan B 안전망 및 성능 기준선 확보.

구조:
  ECG-FM (frozen) → mean pooling → Linear(768→1) → BCEWithLogitsLoss

클래스 불균형:
  응급 70.8% vs 정상 29.2% → pos_weight = n_normal/n_emergency (train set 기준 자동 계산)

평가 지표:
  AUROC, F1@0.5, Sensitivity@95%Specificity

저장:
  --out_dir/baseline_best.pt  — val AUROC 최고 모델 가중치

사용법:
  python scripts/train_baseline.py \\
      --data_dir  D:/WidU_ecg-fm_emergency-detection/data/processed/cpsc2018 \\
      --ckpt_path D:/WidU_ecg-fm_emergency-detection/checkpoints/ecg-fm/mimic_iv_ecg_physionet_pretrained.pt \\
      --out_dir   D:/WidU_ecg-fm_emergency-detection/outputs/baseline
"""

import argparse
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:
    from sklearn.metrics import roc_auc_score, f1_score, roc_curve
except ImportError:
    sys.exit("[오류] scikit-learn 미설치 — pip install scikit-learn")

FS        = 500
N_LEADS   = 12
N_SAMPLES = 5000
EMBED_DIM = 768


# ── 데이터셋 ─────────────────────────────────────────────────────────

class CPSCDataset(Dataset):
    def __init__(self, split_dir: str):
        self.signals = np.load(os.path.join(split_dir, "signals.npy"))   # (N,12,5000)
        self.labels  = np.load(os.path.join(split_dir, "labels.npy"))    # (N,) int8

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor(self.signals[idx], dtype=torch.float32)  # (12, 5000)
        y = torch.tensor(float(self.labels[idx]), dtype=torch.float32)
        return x, y


# ── ECG-FM 로드 ───────────────────────────────────────────────────────

def load_ecgfm(ckpt_path: str, device: torch.device):
    from fairseq_signals.utils.checkpoint_utils import load_model_and_task
    result = load_model_and_task(ckpt_path)
    if isinstance(result, tuple):
        for r in result:
            if hasattr(r, "parameters"):
                return r.to(device)
            if isinstance(r, list) and r and hasattr(r[0], "parameters"):
                return r[0].to(device)
    return result.to(device)


@torch.no_grad()
def extract_embedding(backbone, x: torch.Tensor) -> torch.Tensor:
    """
    x: (B, 12, 5000)
    반환: (B, 768) — features_only=True 후 시간축 mean pooling
    """
    out  = backbone(source=x, padding_mask=None, features_only=True)
    emb  = out["x"]          # (B, T, 768)
    return emb.mean(dim=1)   # (B, 768)


# ── 분류 헤드 ─────────────────────────────────────────────────────────

class LinearHead(nn.Module):
    def __init__(self, in_dim: int = EMBED_DIM):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, x):
        return self.fc(x).squeeze(-1)  # (B,)


# ── 평가 ─────────────────────────────────────────────────────────────

def evaluate(backbone, head, loader, device):
    head.eval()
    all_logits, all_labels = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            emb    = extract_embedding(backbone, x)
            logits = head(emb).cpu()
            all_logits.append(logits)
            all_labels.append(y)

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy().astype(int)
    probs  = 1 / (1 + np.exp(-logits))   # sigmoid

    auroc = roc_auc_score(labels, probs)
    f1    = f1_score(labels, (probs >= 0.5).astype(int), zero_division=0)

    # Sensitivity @ 95% Specificity
    fpr, tpr, _ = roc_curve(labels, probs)
    spec = 1 - fpr
    idx  = np.searchsorted(spec[::-1], 0.95)          # spec 오름차순 변환
    sens_at_95spec = float(tpr[::-1][idx]) if idx < len(tpr) else float("nan")

    return auroc, f1, sens_at_95spec


# ── 학습 루프 ─────────────────────────────────────────────────────────

def train(args):
    # 재현성 보장
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print("단계 5: ECG-FM 베이스라인 선형 프로빙")
    print("=" * 65)
    print(f"디바이스: {device}")
    print(f"시드:     {args.seed}")
    print(f"데이터:   {args.data_dir}")
    print(f"체크포인트: {args.ckpt_path}")
    print(f"출력:     {args.out_dir}")
    print()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── 데이터 로드 ───────────────────────────────────────────────────
    train_ds = CPSCDataset(os.path.join(args.data_dir, "train"))
    val_ds   = CPSCDataset(os.path.join(args.data_dir, "val"))
    test_ds  = CPSCDataset(os.path.join(args.data_dir, "test"))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, num_workers=0, pin_memory=True)

    n_pos = int((train_ds.labels == 1).sum())
    n_neg = int((train_ds.labels == 0).sum())
    pos_weight = torch.tensor(n_neg / n_pos, dtype=torch.float32).to(device)
    print(f"[데이터] train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    print(f"         응급={n_pos}, 정상={n_neg}, pos_weight={pos_weight.item():.4f}")
    print()

    # ── 모델 로드 ─────────────────────────────────────────────────────
    print("[모델] ECG-FM 백본 로드 중...")
    backbone = load_ecgfm(args.ckpt_path, device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    total_params = sum(p.numel() for p in backbone.parameters())
    print(f"       {type(backbone).__name__}, {total_params/1e6:.1f}M params (전부 동결)")

    head = LinearHead().to(device)
    head_params = sum(p.numel() for p in head.parameters())
    print(f"       LinearHead(768→1): {head_params} 파라미터 (학습 대상)")
    print()

    # ── 옵티마이저 · 손실 ─────────────────────────────────────────────
    optimizer = torch.optim.Adam(head.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_auroc  = 0.0
    best_epoch  = 0
    best_path   = os.path.join(args.out_dir, "baseline_best.pt")

    print(f"{'Epoch':>5} {'TrainLoss':>10} {'ValAUROC':>9} {'ValF1':>7} {'Sens@95Sp':>10}")
    print("-" * 47)

    for epoch in range(1, args.epochs + 1):
        head.train()
        total_loss = 0.0
        t0 = time.time()

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            emb    = extract_embedding(backbone, x)
            logits = head(emb)
            loss   = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)

        avg_loss = total_loss / len(train_ds)
        val_auroc, val_f1, val_sens = evaluate(backbone, head, val_loader, device)

        marker = " ←" if val_auroc > best_auroc else ""
        print(f"{epoch:5d} {avg_loss:10.4f} {val_auroc:9.4f} {val_f1:7.4f} "
              f"{val_sens:10.4f}{marker}")

        if val_auroc > best_auroc:
            best_auroc = val_auroc
            best_epoch = epoch
            torch.save({"epoch": epoch,
                        "head_state": head.state_dict(),
                        "val_auroc": val_auroc,
                        "val_f1": val_f1}, best_path)

    # ── 테스트 평가 ───────────────────────────────────────────────────
    print()
    print(f"최고 val AUROC: {best_auroc:.4f} (epoch {best_epoch})")
    ckpt = torch.load(best_path, map_location=device)
    head.load_state_dict(ckpt["head_state"])

    test_auroc, test_f1, test_sens = evaluate(backbone, head, test_loader, device)

    print()
    print("=" * 65)
    print("단계 5 결과 (테스트 세트)")
    print("=" * 65)
    print(f"  AUROC              : {test_auroc:.4f}")
    print(f"  F1 (@threshold 0.5): {test_f1:.4f}")
    print(f"  Sensitivity@95%Sp  : {test_sens:.4f}")
    print()
    print(f"  모델 저장: {best_path}")
    print()
    print("다음 단계:")
    print("  → decisions.md에 이 수치 기록")
    print("  → 단계 6: LoRA fine-tuning (scripts/train_lora.py)")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir",
        default="D:/WidU_ecg-fm_emergency-detection/data/processed/cpsc2018")
    parser.add_argument("--ckpt_path",
        default="D:/WidU_ecg-fm_emergency-detection/checkpoints/ecg-fm/mimic_iv_ecg_physionet_pretrained.pt")
    parser.add_argument("--out_dir",
        default="D:/WidU_ecg-fm_emergency-detection/outputs/baseline")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs",     type=int, default=30)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
