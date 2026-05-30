"""
단계 7: 신호품질 게이트 (ECG-FM 동결 + 선형 프로빙)
=====================================================
목적:
  ECG-FM 백본을 완전 동결하고 선형 분류 헤드만 학습.
  불량 신호(unacceptable)를 응급 탐지 전 단계에서 거부.

라벨 규약:
  1 = unacceptable (품질 불량, positive — 게이트 탐지 대상)
  0 = acceptable   (품질 양호, negative)

구조:
  ECG-FM (frozen) → mean pooling → Linear(768→1) → BCEWithLogitsLoss

클래스 불균형:
  양호 ≈ 77% vs 불량 ≈ 23% → pos_weight = n_acceptable/n_unacceptable (자동 계산)

평가 지표:
  AUROC, F1@0.5, Sensitivity@95%Specificity
  (Sens@95%Sp = 5% 오거부율에서 불량 신호 탐지율 — 게이트 핵심 지표)

저장:
  --out_dir/gate_best.pt  — val AUROC 최고 모델 (head_state 저장)

사용법:
  python scripts/train_gate.py \\
      --data_dir  D:/WidU_ecg-fm_emergency-detection/data/processed/physionet2011 \\
      --ckpt_path D:/WidU_ecg-fm_emergency-detection/checkpoints/ecg-fm/mimic_iv_ecg_physionet_pretrained.pt \\
      --out_dir   D:/WidU_ecg-fm_emergency-detection/outputs/gate
"""

import argparse
import os
import sys
import time

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
    sys.exit("[오류] scikit-learn 미설치 — pip install scikit-learn")

FS        = 500
N_LEADS   = 12
N_SAMPLES = 5000
EMBED_DIM = 768


# ── 데이터셋 ─────────────────────────────────────────────────────────

class GateDataset(Dataset):
    def __init__(self, split_dir: str):
        self.signals = np.load(os.path.join(split_dir, "signals.npy"))   # (N,12,5000)
        self.labels  = np.load(os.path.join(split_dir, "labels.npy"))    # (N,) int8

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor(self.signals[idx], dtype=torch.float32)   # (12,5000)
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
    """x: (B,12,5000) → (B,768)"""
    out = backbone(source=x, padding_mask=None, features_only=True)
    return out["x"].mean(dim=1)


# ── 분류 헤드 ─────────────────────────────────────────────────────────

class LinearHead(nn.Module):
    def __init__(self, in_dim: int = EMBED_DIM):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, x):
        return self.fc(x).squeeze(-1)


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
    probs  = 1 / (1 + np.exp(-logits))

    auroc = roc_auc_score(labels, probs)
    f1    = f1_score(labels, (probs >= 0.5).astype(int), zero_division=0)

    # Sensitivity @ 95% Specificity
    # positive class = 1 (unacceptable/불량)
    fpr, tpr, _ = roc_curve(labels, probs)
    spec = 1 - fpr
    idx  = np.searchsorted(spec[::-1], 0.95)
    sens_at_95spec = float(tpr[::-1][idx]) if idx < len(tpr) else float("nan")

    return auroc, f1, sens_at_95spec


# ── 학습 루프 ─────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print("단계 7: 신호품질 게이트 (ECG-FM 동결 + 선형 프로빙)")
    print("=" * 65)
    print(f"디바이스: {device}")
    print(f"데이터:   {args.data_dir}")
    print(f"출력:     {args.out_dir}")
    print()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── 데이터 로드 ───────────────────────────────────────────────────
    train_ds = GateDataset(os.path.join(args.data_dir, "train"))
    val_ds   = GateDataset(os.path.join(args.data_dir, "val"))
    test_ds  = GateDataset(os.path.join(args.data_dir, "test"))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, num_workers=0, pin_memory=True)

    # pos_weight: 불량(1)이 소수 → 업가중
    n_good = int((train_ds.labels == 0).sum())
    n_bad  = int((train_ds.labels == 1).sum())
    pos_weight = torch.tensor(n_good / max(n_bad, 1), dtype=torch.float32).to(device)

    print(f"[데이터] train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    print(f"         train 불량(1)={n_bad}, 양호(0)={n_good}, pos_weight={pos_weight.item():.4f}")
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

    best_auroc = 0.0
    best_epoch = 0
    best_path  = os.path.join(args.out_dir, "gate_best.pt")

    print(f"{'Epoch':>5} {'TrainLoss':>10} {'ValAUROC':>9} {'ValF1':>7} {'Sens@95Sp':>10}")
    print("-" * 47)

    for epoch in range(1, args.epochs + 1):
        head.train()
        total_loss = 0.0

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

        marker = " <-" if val_auroc > best_auroc else ""
        print(f"{epoch:5d} {avg_loss:10.4f} {val_auroc:9.4f} {val_f1:7.4f} "
              f"{val_sens:10.4f}{marker}")
        sys.stdout.flush()

        if val_auroc > best_auroc:
            best_auroc = val_auroc
            best_epoch = epoch
            torch.save({"epoch":      epoch,
                        "head_state": head.state_dict(),
                        "val_auroc":  val_auroc,
                        "val_f1":     val_f1,
                        "val_sens":   val_sens}, best_path)

    # ── 테스트 평가 ───────────────────────────────────────────────────
    print()
    print(f"최고 val AUROC: {best_auroc:.4f} (epoch {best_epoch})")
    ckpt = torch.load(best_path, map_location=device)
    head.load_state_dict(ckpt["head_state"])

    test_auroc, test_f1, test_sens = evaluate(backbone, head, test_loader, device)

    print()
    print("=" * 65)
    print("단계 7 결과 (테스트 세트)")
    print("=" * 65)
    print(f"  AUROC              : {test_auroc:.4f}")
    print(f"  F1 (@threshold 0.5): {test_f1:.4f}")
    print(f"  Sensitivity@95%Sp  : {test_sens:.4f}")
    print(f"    (5% 오거부율에서 불량 신호 {test_sens*100:.1f}% 탐지)")
    print()
    print(f"  모델 저장: {best_path}")
    print()
    print("다음 단계:")
    print("  → records/에 결과 기록 후 단계 9: 외부검증")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir",
        default="D:/WidU_ecg-fm_emergency-detection/data/processed/physionet2011")
    parser.add_argument("--ckpt_path",
        default="D:/WidU_ecg-fm_emergency-detection/checkpoints/ecg-fm/mimic_iv_ecg_physionet_pretrained.pt")
    parser.add_argument("--out_dir",
        default="D:/WidU_ecg-fm_emergency-detection/outputs/gate")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs",     type=int, default=50)
    parser.add_argument("--lr",         type=float, default=1e-3)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
