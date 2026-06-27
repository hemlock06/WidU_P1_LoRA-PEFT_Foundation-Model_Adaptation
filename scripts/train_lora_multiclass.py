"""
단계 5b: ECG-FM LoRA 다중분류 Fine-tuning (5-class)
=====================================================
목적:
  단계 6의 LoRA+RLM 구조를 그대로 가져와 헤드를 5-class로 확장.
  심장 다중 taxonomy(정상/AF/급성허혈/전도장애/이소성) 분류 학습.

데이터:
  data/processed/cpsc2018_mc/ (preprocess_cpsc2018_mc.py 출력)
  labels.npy: int8 0~4 (5-class), labels_bin.npy: 파생 이진 (분석용)

LoRA 설정: rank=8, alpha=16, dropout=0.1, q_proj·v_proj
RLM: p=0.5 (동일)
손실: CrossEntropyLoss + 클래스 가중치 (역빈도)

평가지표:
  - Macro-F1 (val/test best 기준)
  - Per-class AUROC (one-vs-rest)
  - Confusion matrix
  - 파생 이진 AUROC: P(class1) + P(class2)를 응급 점수로 변환

사용법:
  python scripts/train_lora_multiclass.py
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
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        f1_score,
        roc_auc_score,
    )
except ImportError:
    sys.exit("[오류] scikit-learn 미설치 — pip install scikit-learn")

EMBED_DIM = 768
N_CLASSES = 5
CLASS_NAMES = [
    "정상(NSR)",
    "AF",
    "급성허혈(STD/STE)",
    "전도장애(I-AVB/LBBB/RBBB)",
    "이소성(PAC/PVC)",
]
EMERGENCY_CLASSES = (1, 2)  # AF + 급성허혈 = 응급


# ── LoRA 모듈 (train_lora.py와 동일) ─────────────────────────────────


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
    replaced = []
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
        replaced.append(name)
    return replaced


# ── 데이터셋 ─────────────────────────────────────────────────────────


class CPSCMulticlassDataset(Dataset):
    def __init__(self, split_dir: str):
        self.signals = np.load(os.path.join(split_dir, "signals.npy"))
        self.labels = np.load(os.path.join(split_dir, "labels.npy"))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor(self.signals[idx], dtype=torch.float32)
        y = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return x, y


def random_lead_mask(x: torch.Tensor, p: float = 0.5) -> torch.Tensor:
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
    out = backbone(source=x, padding_mask=None, features_only=True)
    return out["x"].mean(dim=1)


# ── 분류 헤드 (5-class) ───────────────────────────────────────────────


class MulticlassHead(nn.Module):
    def __init__(self, in_dim: int = EMBED_DIM, n_classes: int = N_CLASSES):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        return self.fc(x)


# ── 평가 ─────────────────────────────────────────────────────────────


@torch.no_grad()
def evaluate(backbone, head, loader, device):
    backbone.eval()
    head.eval()
    all_logits, all_labels = [], []

    for x, y in loader:
        x = x.to(device)
        emb = extract_embedding(backbone, x)
        logits = head(emb).cpu()
        all_logits.append(logits)
        all_labels.append(y)

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy().astype(int)
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    preds = probs.argmax(axis=-1)

    # Macro-F1 (per-class F1 평균)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(labels, preds, average="weighted", zero_division=0)

    # Per-class AUROC (one-vs-rest)
    aurocs = []
    for c in range(N_CLASSES):
        y_c = (labels == c).astype(int)
        if y_c.sum() == 0 or y_c.sum() == len(y_c):
            aurocs.append(float("nan"))
        else:
            aurocs.append(roc_auc_score(y_c, probs[:, c]))

    # 파생 이진 응급 점수: P(class1) + P(class2)
    bin_score = probs[:, EMERGENCY_CLASSES[0]] + probs[:, EMERGENCY_CLASSES[1]]
    bin_labels = np.isin(labels, EMERGENCY_CLASSES).astype(int)
    if 0 < bin_labels.sum() < len(bin_labels):
        bin_auroc = roc_auc_score(bin_labels, bin_score)
    else:
        bin_auroc = float("nan")

    return {
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class_auroc": aurocs,
        "bin_auroc": bin_auroc,
        "preds": preds,
        "labels": labels,
        "probs": probs,
    }


# ── 학습 ─────────────────────────────────────────────────────────────


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("단계 5b: ECG-FM LoRA 다중분류 Fine-tuning (5-class)")
    print("=" * 70)
    print(f"디바이스:   {device}")
    print(f"데이터:     {args.data_dir}")
    print(f"체크포인트: {args.ckpt_path}")
    print(f"출력:       {args.out_dir}")
    print(
        f"LoRA:       rank={args.lora_rank}, alpha={args.lora_alpha}, "
        f"dropout={args.lora_dropout}"
    )
    print(f"RLM:        p={args.rlm_p}")
    print()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── 데이터 ────────────────────────────────────────────────────────
    train_ds = CPSCMulticlassDataset(os.path.join(args.data_dir, "train"))
    val_ds = CPSCMulticlassDataset(os.path.join(args.data_dir, "val"))
    test_ds = CPSCMulticlassDataset(os.path.join(args.data_dir, "test"))

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # 클래스 가중치 (역빈도)
    counts = np.array(
        [(train_ds.labels == c).sum() for c in range(N_CLASSES)], dtype=np.float64
    )
    weights = len(train_ds.labels) / (N_CLASSES * counts)
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

    print(f"[데이터] train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    print("[클래스] 가중치 (역빈도):")
    for c in range(N_CLASSES):
        print(
            f"  [{c}] {CLASS_NAMES[c]:30s}: n={int(counts[c]):4d}  w={weights[c]:.4f}"
        )
    print()

    # ── ECG-FM 로드 + LoRA 주입 ───────────────────────────────────────
    print("[모델] ECG-FM 백본 로드 중...")
    backbone = load_ecgfm(args.ckpt_path, device)
    backbone.train()
    for p in backbone.parameters():
        p.requires_grad_(False)
    replaced = inject_lora(
        backbone, rank=args.lora_rank, alpha=args.lora_alpha, dropout=args.lora_dropout
    )
    print(f"       LoRA 주입: {len(replaced)}개 레이어")
    n_lora = sum(p.numel() for p in backbone.parameters() if p.requires_grad)

    head = MulticlassHead().to(device)
    n_head = sum(p.numel() for p in head.parameters())
    print(f"       MulticlassHead(768→5): {n_head}개")
    print(f"       총 학습 파라미터: {n_lora + n_head:,}개")
    print()

    # ── 옵티마이저 ────────────────────────────────────────────────────
    params = [p for p in backbone.parameters() if p.requires_grad] + list(
        head.parameters()
    )
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.1
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_f1 = 0.0
    best_epoch = 0
    best_path = os.path.join(args.out_dir, "lora_mc_best.pt")

    print(
        f"{'Epoch':>5} {'TrainLoss':>10} {'ValF1m':>7} "
        f"{'ValF1w':>7} {'BinAUROC':>9} {'LR':>9}"
    )
    print("-" * 60)

    for epoch in range(1, args.epochs + 1):
        backbone.train()
        head.train()
        total_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            if args.rlm_p > 0:
                x = random_lead_mask(x, p=args.rlm_p)
            emb = extract_embedding(backbone, x)
            logits = head(emb)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * len(y)

        scheduler.step()
        avg_loss = total_loss / len(train_ds)
        metrics = evaluate(backbone, head, val_loader, device)
        lr_cur = scheduler.get_last_lr()[0]

        marker = " ←" if metrics["macro_f1"] > best_f1 else ""
        print(
            f"{epoch:5d} {avg_loss:10.4f} {metrics['macro_f1']:7.4f} "
            f"{metrics['weighted_f1']:7.4f} {metrics['bin_auroc']:9.4f} "
            f"{lr_cur:9.2e}{marker}"
        )

        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "backbone_lora": {
                        k: v for k, v in backbone.state_dict().items() if "lora_" in k
                    },
                    "head_state": head.state_dict(),
                    "val_macro_f1": metrics["macro_f1"],
                    "val_bin_auroc": metrics["bin_auroc"],
                    "lora_rank": args.lora_rank,
                    "lora_alpha": args.lora_alpha,
                    "n_classes": N_CLASSES,
                    "class_names": CLASS_NAMES,
                },
                best_path,
            )

    # ── 테스트 평가 ───────────────────────────────────────────────────
    print()
    print(f"최고 val macro-F1: {best_f1:.4f} (epoch {best_epoch})")
    ckpt = torch.load(best_path, map_location=device)
    backbone.load_state_dict(ckpt["backbone_lora"], strict=False)
    head.load_state_dict(ckpt["head_state"])

    res = evaluate(backbone, head, test_loader, device)

    print()
    print("=" * 70)
    print("단계 5b 결과 (테스트 세트)")
    print("=" * 70)
    print(f"  Macro-F1     : {res['macro_f1']:.4f}")
    print(f"  Weighted-F1  : {res['weighted_f1']:.4f}")
    print(f"  파생 이진 AUROC (응급=class1+2): {res['bin_auroc']:.4f}")
    print()
    print("  Per-class AUROC (one-vs-rest):")
    for c in range(N_CLASSES):
        print(f"    [{c}] {CLASS_NAMES[c]:30s}: AUROC={res['per_class_auroc'][c]:.4f}")
    print()
    print("  Confusion Matrix (rows=true, cols=pred):")
    cm = confusion_matrix(res["labels"], res["preds"], labels=list(range(N_CLASSES)))
    print(f"    {'pred→':<3s} " + " ".join(f"{c:>5d}" for c in range(N_CLASSES)))
    for r in range(N_CLASSES):
        print(f"    [{r}]   " + " ".join(f"{cm[r, c]:>5d}" for c in range(N_CLASSES)))
    print()
    print("  Classification Report:")
    print(
        classification_report(
            res["labels"],
            res["preds"],
            target_names=CLASS_NAMES,
            zero_division=0,
            digits=4,
        )
    )
    print(f"  모델 저장: {best_path}")
    print("=" * 70)

    # 평가 결과를 npz로 저장 (records 작성용)
    np.savez(
        os.path.join(args.out_dir, "test_results.npz"),
        labels=res["labels"],
        preds=res["preds"],
        probs=res["probs"],
        macro_f1=res["macro_f1"],
        weighted_f1=res["weighted_f1"],
        per_class_auroc=np.array(res["per_class_auroc"]),
        bin_auroc=res["bin_auroc"],
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data_dir", default="data/processed/cpsc2018_mc")
    parser.add_argument(
        "--ckpt_path", default="checkpoints/ecg-fm/mimic_iv_ecg_physionet_pretrained.pt"
    )
    parser.add_argument("--out_dir", default="outputs/lora_mc")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--rlm_p", type=float, default=0.5)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
