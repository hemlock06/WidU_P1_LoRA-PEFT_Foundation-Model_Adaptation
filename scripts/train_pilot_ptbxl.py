"""
PTB-XL 파일럿 학습 (train_pilot_ptbxl.py)
==========================================
목적:
  CPSC 2018 + PTB-XL 혼합 데이터로 LoRA+RLM+multi-SNR 모델 학습.
  STAFF-III 외부검증에서 AUROC 개선 여부를 파일럿으로 확인.

전략:
  - CPSC (train 2217) + PTB-XL train 혼합
  - PTB-XL 비율: 1:1 비율 (oversampling or undersampling)
  - LoRA rank=8, alpha=16, RLM p=0.5, multi-SNR {24,18,12,6,0}dB
  - 20 epoch (파일럿 — 전체 30보다 짧음)
  - 평가: CPSC test + STAFF-III

사용법:
  python scripts/train_pilot_ptbxl.py
  python scripts/train_pilot_ptbxl.py --ptbxl_ratio 0.5 --epochs 20
"""

import argparse
import math
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    from sklearn.metrics import f1_score, roc_auc_score, roc_curve
except ImportError:
    sys.exit("[오류] scikit-learn 미설치")

# ── 경로 ────────────────────────────────────────────────────────────────
CKPT_FM = "checkpoints/ecg-fm/mimic_iv_ecg_physionet_pretrained.pt"
PROCESSED = "data/processed"
OUTPUTS = "outputs"
NSTDB_DIR = "data/raw/nstdb"
OUT_DIR = f"{OUTPUTS}/pilot_ptbxl"

CPSC_TRAIN = f"{PROCESSED}/cpsc2018/train"
CPSC_VAL = f"{PROCESSED}/cpsc2018/val"
CPSC_TEST = f"{PROCESSED}/cpsc2018/test"
PTBXL_TRAIN = f"{PROCESSED}/ptbxl/train"
STAFF_DIR = f"{PROCESSED}/staffiii"
LTST_DIR = f"{PROCESSED}/ltst"

EMBED_DIM = 768
SEED = 42


# ── LoRA ─────────────────────────────────────────────────────────────────
class LoRALinear(nn.Module):
    def __init__(self, linear, rank, alpha, dropout=0.0):
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
    dropout=0.0,
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


# ── 노이즈 증강 (train_lora.py와 동일) ───────────────────────────────────
def load_noise_templates(nstdb_dir, n_samples=5000):
    templates = {"bw": [], "em": [], "ma": []}
    for ntype in list(templates.keys()):
        for ext in [".dat", ""]:
            path = os.path.join(nstdb_dir, ntype)
            try:
                import wfdb

                rec = wfdb.rdrecord(path)
                sig = rec.p_signal[:, 0].astype(np.float32)
                # 여러 구간으로 분할
                for i in range(0, len(sig) - n_samples, n_samples // 2):
                    templates[ntype].append(sig[i : i + n_samples])
                break
            except Exception:
                continue
    return templates


def add_noise(x, noise_templates, snr_db, device):
    """x: (B,12,5000) tensor. 랜덤 lead·SNR로 노이즈 추가."""
    if not noise_templates or all(len(v) == 0 for v in noise_templates.values()):
        return x
    x = x.clone()
    B = x.shape[0]
    ntypes = [k for k, v in noise_templates.items() if v]
    for b in range(B):
        ntype = random.choice(ntypes)
        tmpl = random.choice(noise_templates[ntype])
        tmpl = torch.tensor(tmpl, dtype=torch.float32, device=device)
        for lead in range(12):
            if random.random() > 0.5:
                sig = x[b, lead]
                rms_sig = sig.pow(2).mean().sqrt().clamp(min=1e-8)
                rms_noise = tmpl.pow(2).mean().sqrt().clamp(min=1e-8)
                scale = rms_sig / rms_noise / (10 ** (snr_db / 20))
                x[b, lead] = sig + tmpl * scale
    return x


# ── 데이터셋 ─────────────────────────────────────────────────────────────
class ECGDataset(Dataset):
    def __init__(self, data_dir):
        self.signals = np.load(os.path.join(data_dir, "signals.npy"))
        self.labels = np.load(os.path.join(data_dir, "labels.npy"))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor(self.signals[idx], dtype=torch.float32)
        y = torch.tensor(float(self.labels[idx]), dtype=torch.float32)
        return x, y


# ── 분류 헤드 ─────────────────────────────────────────────────────────────
class LinearHead(nn.Module):
    def __init__(self, in_dim=EMBED_DIM):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, x):
        return self.fc(x).squeeze(-1)


# ── ECG-FM 로드 ───────────────────────────────────────────────────────────
def load_ecgfm(device):
    from fairseq_signals.utils.checkpoint_utils import load_model_and_task

    result = load_model_and_task(CKPT_FM)
    if isinstance(result, tuple):
        for r in result:
            if hasattr(r, "parameters"):
                return r.to(device)
            if isinstance(r, list) and r and hasattr(r[0], "parameters"):
                return r[0].to(device)
    return result.to(device)


# ── 평가 ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(backbone, head, loader, device, label=""):
    head.eval()
    backbone.eval()
    all_logits, all_labels = [], []
    for x, y in loader:
        x = x.to(device)
        emb = backbone(source=x, padding_mask=None, features_only=True)["x"].mean(dim=1)
        logits = head(emb).cpu()
        all_logits.append(logits)
        all_labels.append(y)
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy().astype(int)
    probs = 1 / (1 + np.exp(-logits))
    if len(np.unique(labels)) < 2:
        return float("nan"), float("nan"), float("nan")
    auroc = roc_auc_score(labels, probs)
    f1 = f1_score(labels, (probs >= 0.5).astype(int), zero_division=0)
    fpr, tpr, _ = roc_curve(labels, probs)
    spec = 1 - fpr
    idx = np.searchsorted(spec[::-1], 0.95)
    sens = float(tpr[::-1][idx]) if idx < len(tpr) else float("nan")
    return auroc, f1, sens


# ── 메인 ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument(
        "--rlm_p", type=float, default=0.5, help="Random Lead Masking 확률"
    )
    parser.add_argument(
        "--p_noise", type=float, default=0.75, help="배치당 노이즈 증강 적용 확률"
    )
    parser.add_argument(
        "--snr_levels", nargs="+", type=float, default=[24.0, 18.0, 12.0, 6.0, 0.0]
    )
    parser.add_argument(
        "--ptbxl_ratio",
        type=float,
        default=1.0,
        help="PTB-XL : CPSC 비율 (1.0=동수, 0.5=CPSC 절반)",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out_dir", default=OUT_DIR)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 65)
    print("PTB-XL 파일럿: CPSC + PTB-XL 혼합 학습")
    print("=" * 65)
    print(f"디바이스: {device}  |  seed={args.seed}")
    print(f"PTB-XL 비율: {args.ptbxl_ratio}  |  epochs={args.epochs}")
    print()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── 데이터 로드 ───────────────────────────────────────────────────────
    cpsc_ds = ECGDataset(CPSC_TRAIN)
    ptbxl_ds = ECGDataset(PTBXL_TRAIN)
    val_ds = ECGDataset(CPSC_VAL)
    cpsc_test_ds = ECGDataset(CPSC_TEST)

    n_cpsc = len(cpsc_ds)
    n_ptbxl = len(ptbxl_ds)
    print(f"[데이터] CPSC train={n_cpsc}, PTB-XL train={n_ptbxl}")
    print(f"         CPSC val={len(val_ds)}, CPSC test={len(cpsc_test_ds)}")

    # 혼합: WeightedRandomSampler로 CPSC:PTB-XL = 1:ptbxl_ratio
    combined = ConcatDataset([cpsc_ds, ptbxl_ds])
    w_cpsc = 1.0
    w_ptbxl = args.ptbxl_ratio
    weights = [w_cpsc] * n_cpsc + [w_ptbxl] * n_ptbxl
    sampler = WeightedRandomSampler(
        weights, num_samples=int(n_cpsc * (1 + args.ptbxl_ratio)), replacement=True
    )
    train_loader = DataLoader(
        combined,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(cpsc_test_ds, batch_size=32, shuffle=False, num_workers=0)

    # CPSC 클래스 가중치 (합산 데이터 기준 근사)
    all_labels = np.concatenate([cpsc_ds.labels, ptbxl_ds.labels])
    n_pos = (all_labels == 1).sum()
    n_neg = (all_labels == 0).sum()
    pos_weight = torch.tensor(n_neg / n_pos, dtype=torch.float32).to(device)
    print(
        f"         혼합 응급={n_pos}, 정상={n_neg}, pos_weight={pos_weight.item():.3f}"
    )

    # 외부검증 로더
    ext_loaders = {}
    if os.path.isdir(STAFF_DIR):
        ext_loaders["STAFF-III"] = DataLoader(
            ECGDataset(STAFF_DIR), batch_size=64, shuffle=False, num_workers=0
        )
        print(f"         STAFF-III={len(ECGDataset(STAFF_DIR))}")
    if os.path.isdir(LTST_DIR):
        ext_loaders["LTST"] = DataLoader(
            ECGDataset(LTST_DIR), batch_size=64, shuffle=False, num_workers=0
        )
        print(f"         LTST={len(ECGDataset(LTST_DIR))}")
    print()

    # ── 노이즈 템플릿 ─────────────────────────────────────────────────────
    noise_templates = {}
    if os.path.isdir(NSTDB_DIR):
        try:
            noise_templates = load_noise_templates(NSTDB_DIR)
            n_tmpl = sum(len(v) for v in noise_templates.values())
            print(f"[노이즈] NSTDB 템플릿 {n_tmpl}개 로드")
        except Exception as e:
            print(f"[노이즈] 로드 실패 ({e}) — 증강 없이 진행")

    # ── 모델 ─────────────────────────────────────────────────────────────
    print("[모델] ECG-FM 로드 중...")
    backbone = load_ecgfm(device)
    inject_lora(backbone, rank=args.lora_rank, alpha=args.lora_alpha, dropout=0.1)
    backbone.train()
    trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    total = sum(p.numel() for p in backbone.parameters())
    print(f"       LoRA 주입: {trainable:,} / {total:,} 파라미터 학습")

    head = LinearHead().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        list(p for p in backbone.parameters() if p.requires_grad)
        + list(head.parameters()),
        lr=args.lr,
        weight_decay=1e-2,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=5e-5
    )

    best_auroc = 0.0
    best_epoch = 0
    best_path = os.path.join(args.out_dir, "pilot_ptbxl_best.pt")

    print()
    print(f"{'Epoch':>5} {'Loss':>8} {'ValAUROC':>9} {'ValF1':>7} {'STAFF':>7}")
    print("-" * 48)

    for epoch in range(1, args.epochs + 1):
        backbone.train()
        head.train()
        total_loss = 0.0
        n_samples = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            # RLM
            if args.rlm_p > 0:
                mask = torch.rand(x.shape[0], 12, device=device) < args.rlm_p
                x[mask.unsqueeze(-1).expand_as(x)] = 0.0

            # multi-SNR 증강
            if noise_templates and random.random() < args.p_noise:
                snr = random.choice(args.snr_levels)
                x = add_noise(x, noise_templates, snr, device)

            emb = backbone(source=x, padding_mask=None, features_only=True)["x"].mean(
                dim=1
            )
            logits = head(emb)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(p for p in backbone.parameters() if p.requires_grad)
                + list(head.parameters()),
                max_norm=1.0,
            )
            optimizer.step()
            total_loss += loss.item() * len(y)
            n_samples += len(y)

        scheduler.step()
        avg_loss = total_loss / max(n_samples, 1)
        val_auroc, val_f1, _ = evaluate(backbone, head, val_loader, device)

        # STAFF-III 매 에폭 평가 (빠른 피드백)
        staff_auroc = float("nan")
        if "STAFF-III" in ext_loaders:
            staff_auroc, _, _ = evaluate(
                backbone, head, ext_loaders["STAFF-III"], device
            )

        marker = " ←" if val_auroc > best_auroc else ""
        staff_str = f"{staff_auroc:.4f}" if not np.isnan(staff_auroc) else "  n/a "
        print(
            f"{epoch:5d} {avg_loss:8.4f} {val_auroc:9.4f} {val_f1:7.4f} {staff_str}{marker}"
        )

        if val_auroc > best_auroc:
            best_auroc = val_auroc
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "head_state": head.state_dict(),
                    "backbone_lora": backbone.state_dict(),
                    "val_auroc": val_auroc,
                    "lora_rank": args.lora_rank,
                    "lora_alpha": args.lora_alpha,
                },
                best_path,
            )

    # ── 최종 평가 ─────────────────────────────────────────────────────────
    print()
    print(f"최고 val AUROC: {best_auroc:.4f} (epoch {best_epoch})")
    ckpt = torch.load(best_path, map_location=device)
    backbone.load_state_dict(ckpt["backbone_lora"], strict=False)
    head.load_state_dict(ckpt["head_state"])

    print()
    print("=" * 65)
    print("파일럿 최종 결과 (best checkpoint)")
    print("=" * 65)

    cpsc_a, cpsc_f, cpsc_s = evaluate(backbone, head, test_loader, device)
    print(f"  CPSC test   AUROC={cpsc_a:.4f}  F1={cpsc_f:.4f}  Sens={cpsc_s:.4f}")

    for db_name, loader in ext_loaders.items():
        a, f, s = evaluate(backbone, head, loader, device)
        print(f"  {db_name:12s} AUROC={a:.4f}  F1={f:.4f}  Sens={s:.4f}")

    print()
    print(f"  체크포인트: {best_path}")
    print("=" * 65)
    print()
    if "STAFF-III" in ext_loaders:
        s_auroc, _, _ = evaluate(backbone, head, ext_loaders["STAFF-III"], device)
        if s_auroc > 0.60:
            print("STAFF-III AUROC > 0.60 — PTB-XL 전체 학습 진행 권장")
        elif s_auroc > 0.55:
            print("△  STAFF-III AUROC 0.55~0.60 — 부분 개선, 추가 검토 필요")
        else:
            print("[오류] STAFF-III AUROC ≤ 0.55 — 개선 미미, scope boundary 정리 고려")


if __name__ == "__main__":
    main()
