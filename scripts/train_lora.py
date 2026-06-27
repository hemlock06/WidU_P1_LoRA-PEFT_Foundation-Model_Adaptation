"""
단계 6: ECG-FM LoRA Fine-tuning
=================================
목적:
  ECG-FM 백본에 LoRA를 수동 주입하여 응급 감지 분류기 fine-tuning.
  베이스라인(단계 5) 대비 AUROC·Sensitivity 개선 목표.

LoRA 설정 (decisions.md 확정):
  rank=8, alpha=16, dropout=0.1
  대상: encoder.layers.N.self_attn.{q,v}_proj (N=0~11, 12레이어)
  학습 파라미터: ~295,681개 (전체 90.9M의 0.33%)

증강:
  RLM (Random Lead Masking): 각 lead 독립적으로 p=0.5 확률 0-fill
  (사전학습과 동일 설정 — decisions.md Pre-flight 2 참조)

사용법:
  python scripts/train_lora.py \\
      --data_dir  data/processed/cpsc2018 \\
      --ckpt_path checkpoints/ecg-fm/mimic_iv_ecg_physionet_pretrained.pt \\
      --out_dir   outputs/lora
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
    from sklearn.metrics import roc_auc_score, f1_score, roc_curve
except ImportError:
    sys.exit("[오류] scikit-learn 미설치 — pip install scikit-learn")

EMBED_DIM = 768


# ── LoRA 모듈 ─────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """
    기존 nn.Linear를 LoRA로 감싸는 래퍼.
    forward: W·x + (B·A·x) × (alpha/rank)
    원본 가중치는 동결, A·B만 학습.
    """

    def __init__(self, linear: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        self.original = linear
        self.original.weight.requires_grad_(False)
        if self.original.bias is not None:
            self.original.bias.requires_grad_(False)

        in_dim  = linear.in_features
        out_dim = linear.out_features

        self.lora_A   = nn.Linear(in_dim, rank, bias=False)
        self.lora_B   = nn.Linear(rank, out_dim, bias=False)
        self.scaling  = alpha / rank
        self.dropout  = nn.Dropout(dropout)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    @property
    def bias(self):
        return self.original.bias

    @property
    def weight(self):
        return self.original.weight

    def forward(self, x):
        return self.original(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling


def inject_lora(model: nn.Module, rank: int, alpha: float, dropout: float,
                target_suffixes: tuple = ("self_attn.q_proj", "self_attn.v_proj")):
    """
    model 내의 target_suffixes에 해당하는 Linear 레이어를 LoRALinear로 교체.
    교체된 레이어 이름 목록 반환.
    """
    replaced = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(name.endswith(s) for s in target_suffixes):
            continue

        # 부모 모듈 탐색
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        child_name = parts[-1]

        lora_layer = LoRALinear(module, rank, alpha, dropout)
        setattr(parent, child_name, lora_layer)
        replaced.append(name)

    return replaced


def count_trainable(model: nn.Module):
    total   = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ── 데이터셋 ─────────────────────────────────────────────────────────

class CPSCDataset(Dataset):
    def __init__(self, split_dir: str):
        self.signals = np.load(os.path.join(split_dir, "signals.npy"))
        self.labels  = np.load(os.path.join(split_dir, "labels.npy"))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor(self.signals[idx], dtype=torch.float32)
        y = torch.tensor(float(self.labels[idx]), dtype=torch.float32)
        return x, y


# ── 증강 ─────────────────────────────────────────────────────────────

def random_lead_mask(x: torch.Tensor, p: float = 0.5) -> torch.Tensor:
    """
    RLM: 각 lead를 독립적으로 p 확률로 0-fill.
    x: (B, 12, T)
    사전학습 시 동일 설정 사용 (decisions.md Pre-flight 2 참조).
    """
    B, C, T = x.shape
    mask = (torch.rand(B, C, 1, device=x.device) > p).float()
    return x * mask


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


def extract_embedding(backbone: nn.Module, x: torch.Tensor) -> torch.Tensor:
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

@torch.no_grad()
def evaluate(backbone, head, loader, device):
    backbone.eval()
    head.eval()
    all_logits, all_labels = [], []

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

    fpr, tpr, _ = roc_curve(labels, probs)
    spec = 1 - fpr
    idx  = np.searchsorted(spec[::-1], 0.95)
    sens_at_95spec = float(tpr[::-1][idx]) if idx < len(tpr) else float("nan")

    return auroc, f1, sens_at_95spec


# ── 학습 루프 ─────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print("단계 6: ECG-FM LoRA Fine-tuning")
    print("=" * 65)
    print(f"디바이스:   {device}")
    print(f"데이터:     {args.data_dir}")
    print(f"체크포인트: {args.ckpt_path}")
    print(f"출력:       {args.out_dir}")
    print(f"LoRA:       rank={args.lora_rank}, alpha={args.lora_alpha}, "
          f"dropout={args.lora_dropout}")
    print(f"RLM:        p={args.rlm_p}")
    print()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── 데이터 ────────────────────────────────────────────────────────
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

    # ── ECG-FM 로드 + LoRA 주입 ───────────────────────────────────────
    print("[모델] ECG-FM 백본 로드 중...")
    backbone = load_ecgfm(args.ckpt_path, device)
    backbone.train()

    # 전체 동결 후 LoRA만 해제
    for p in backbone.parameters():
        p.requires_grad_(False)

    replaced = inject_lora(
        backbone,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )
    print(f"       LoRA 주입 완료: {len(replaced)}개 레이어")
    for r in replaced[:4]:
        print(f"         {r}")
    if len(replaced) > 4:
        print(f"         ... 외 {len(replaced)-4}개")

    total, trainable = count_trainable(backbone)
    print(f"       백본 전체={total/1e6:.1f}M, LoRA 학습={trainable:,}개 "
          f"({100*trainable/total:.2f}%)")

    head = LinearHead().to(device)
    head_params = sum(p.numel() for p in head.parameters())
    print(f"       LinearHead(768→1): {head_params}개")
    total_trainable = trainable + head_params
    print(f"       총 학습 파라미터: {total_trainable:,}개")
    print()

    # ── 옵티마이저 ────────────────────────────────────────────────────
    trainable_params = ([p for p in backbone.parameters() if p.requires_grad]
                        + list(head.parameters()))
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr,
                                  weight_decay=args.weight_decay)

    # 코사인 LR 스케줄러 (warmup 없음 — 단순화)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.1
    )

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_auroc = 0.0
    best_epoch = 0
    best_path  = os.path.join(args.out_dir, "lora_best.pt")

    print(f"{'Epoch':>5} {'TrainLoss':>10} {'ValAUROC':>9} "
          f"{'ValF1':>7} {'Sens@95Sp':>10} {'LR':>8}")
    print("-" * 55)

    for epoch in range(1, args.epochs + 1):
        backbone.train()
        head.train()
        total_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            # RLM 증강
            if args.rlm_p > 0:
                x = random_lead_mask(x, p=args.rlm_p)

            emb    = extract_embedding(backbone, x)
            logits = head(emb)
            loss   = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * len(y)

        scheduler.step()
        avg_loss = total_loss / len(train_ds)
        val_auroc, val_f1, val_sens = evaluate(backbone, head, val_loader, device)
        cur_lr = scheduler.get_last_lr()[0]

        marker = " ←" if val_auroc > best_auroc else ""
        print(f"{epoch:5d} {avg_loss:10.4f} {val_auroc:9.4f} {val_f1:7.4f} "
              f"{val_sens:10.4f} {cur_lr:8.2e}{marker}")

        if val_auroc > best_auroc:
            best_auroc = val_auroc
            best_epoch = epoch
            torch.save({
                "epoch":        epoch,
                "backbone_lora": {k: v for k, v in backbone.state_dict().items()
                                  if "lora_" in k},
                "head_state":   head.state_dict(),
                "val_auroc":    val_auroc,
                "val_f1":       val_f1,
                "lora_rank":    args.lora_rank,
                "lora_alpha":   args.lora_alpha,
            }, best_path)

    # ── 테스트 평가 ───────────────────────────────────────────────────
    print()
    print(f"최고 val AUROC: {best_auroc:.4f} (epoch {best_epoch})")
    ckpt = torch.load(best_path, map_location=device)
    # LoRA 가중치만 복원 (나머지는 이미 로드됨)
    missing, unexpected = backbone.load_state_dict(ckpt["backbone_lora"], strict=False)
    head.load_state_dict(ckpt["head_state"])

    test_auroc, test_f1, test_sens = evaluate(backbone, head, test_loader, device)

    print()
    print("=" * 65)
    print("단계 6 결과 (테스트 세트)")
    print("=" * 65)
    print(f"  AUROC              : {test_auroc:.4f}  (베이스라인 0.9435)")
    print(f"  F1 (@threshold 0.5): {test_f1:.4f}  (베이스라인 0.9177)")
    print(f"  Sensitivity@95%Sp  : {test_sens:.4f}  (베이스라인 0.7564)")
    print()
    print(f"  모델 저장: {best_path}")
    print()
    print("다음 단계:")
    print("  → decisions.md에 이 수치 기록")
    print("  → 단계 7: 신호 품질 게이트 (Pre-flight 1 완료 후)")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir",
        default="data/processed/cpsc2018")
    parser.add_argument("--ckpt_path",
        default="checkpoints/ecg-fm/mimic_iv_ecg_physionet_pretrained.pt")
    parser.add_argument("--out_dir",
        default="outputs/lora")
    parser.add_argument("--batch_size",   type=int,   default=16)
    parser.add_argument("--epochs",       type=int,   default=30)
    parser.add_argument("--lr",           type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--lora_rank",    type=int,   default=8)
    parser.add_argument("--lora_alpha",   type=float, default=16.0)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--rlm_p",        type=float, default=0.5,
                        help="RLM: 각 lead가 0-fill될 확률 (0=비활성)")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
