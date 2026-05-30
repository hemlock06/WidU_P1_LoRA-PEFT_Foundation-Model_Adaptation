"""
CPSC 2018 + PTB-XL 혼합 학습 (train_lora_mixed.py)
====================================================
목적:
  CPSC 2018 + PTB-XL 혼합 데이터로 LoRA+RLM+multi-SNR 모델 풀 학습.
  CPSC 단일 학습 모델(lora_multisnr_best.pt)과 성능 비교 후 채택 여부 결정.

전략:
  - CPSC train(2217) + PTB-XL train 혼합 (WeightedRandomSampler, 기본 1:1)
  - LoRA rank=8, alpha=16, dropout=0.1, q_proj·v_proj (12레이어)
  - RLM p=0.5, multi-SNR {24,18,12,6,0}dB, p_noise=0.75
  - 30 epoch, CosineAnnealingLR
  - val AUROC 기준 best checkpoint 저장
  - 최종 평가: CPSC test + 외부검증 4종 (CACHET / INCART / STAFF-III / LTST)

비교 대상 (CPSC 단일):
  outputs/lora_multisnr/lora_multisnr_best.pt  — AUROC=0.9463, Sens@95Sp=0.7620

출력:
  outputs/lora_mixed/lora_mixed_best.pt

사용법:
  python scripts/train_lora_mixed.py
  python scripts/train_lora_mixed.py --ptbxl_ratio 1.0 --epochs 30
"""

import argparse
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, ConcatDataset, WeightedRandomSampler

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    from sklearn.metrics import roc_auc_score, f1_score, roc_curve
except ImportError:
    sys.exit("[오류] scikit-learn 미설치")

# ── 경로 ────────────────────────────────────────────────────────────────
CKPT_FM      = "D:/WidU_ecg-fm_emergency-detection/checkpoints/ecg-fm/mimic_iv_ecg_physionet_pretrained.pt"
PROCESSED    = "D:/WidU_ecg-fm_emergency-detection/data/processed"
OUTPUTS      = "D:/WidU_ecg-fm_emergency-detection/outputs"
NSTDB_DIR    = "D:/WidU_ecg-fm_emergency-detection/data/raw/nstdb"
OUT_DIR      = f"{OUTPUTS}/lora_mixed"

CPSC_TRAIN   = f"{PROCESSED}/cpsc2018/train"
CPSC_VAL     = f"{PROCESSED}/cpsc2018/val"
CPSC_TEST    = f"{PROCESSED}/cpsc2018/test"
PTBXL_TRAIN  = f"{PROCESSED}/ptbxl/train"

# 외부검증 경로
EXT_DIRS = {
    "CACHET":   f"{PROCESSED}/cachet",
    "INCART":   f"{PROCESSED}/incart",
    "STAFF-III":f"{PROCESSED}/staffiii",
    "LTST":     f"{PROCESSED}/ltst",
}

EMBED_DIM = 768
SEED      = 42


# ── LoRA ─────────────────────────────────────────────────────────────────
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


# ── 노이즈 증강 ────────────────────────────────────────────────────────────
def load_noise_templates(nstdb_dir, n_samples=5000):
    templates = {"bw": [], "em": [], "ma": []}
    for ntype in list(templates.keys()):
        path = os.path.join(nstdb_dir, ntype)
        try:
            import wfdb
            rec = wfdb.rdrecord(path)
            sig = rec.p_signal[:, 0].astype(np.float32)
            for i in range(0, len(sig) - n_samples, n_samples // 2):
                templates[ntype].append(sig[i:i + n_samples])
        except Exception:
            continue
    return templates


def add_noise(x, noise_templates, snr_db, device):
    """x: (B,12,5000) tensor"""
    if not noise_templates or all(len(v) == 0 for v in noise_templates.values()):
        return x
    x = x.clone()
    B = x.shape[0]
    ntypes = [k for k, v in noise_templates.items() if v]
    for b in range(B):
        ntype = random.choice(ntypes)
        tmpl  = random.choice(noise_templates[ntype])
        tmpl  = torch.tensor(tmpl, dtype=torch.float32, device=device)
        for lead in range(12):
            if random.random() > 0.5:
                sig = x[b, lead]
                rms_sig   = sig.pow(2).mean().sqrt().clamp(min=1e-8)
                rms_noise = tmpl.pow(2).mean().sqrt().clamp(min=1e-8)
                scale = rms_sig / rms_noise / (10 ** (snr_db / 20))
                x[b, lead] = sig + tmpl * scale
    return x


# ── 데이터셋 ────────────────────────────────────────────────────────────────
class ECGDataset(Dataset):
    def __init__(self, data_dir):
        self.signals = np.load(os.path.join(data_dir, "signals.npy"))
        self.labels  = np.load(os.path.join(data_dir, "labels.npy"))

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor(self.signals[idx], dtype=torch.float32)
        y = torch.tensor(float(self.labels[idx]), dtype=torch.float32)
        return x, y


# ── 분류 헤드 ──────────────────────────────────────────────────────────────
class LinearHead(nn.Module):
    def __init__(self, in_dim=EMBED_DIM):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, x): return self.fc(x).squeeze(-1)


# ── ECG-FM 로드 ──────────────────────────────────────────────────────────────
def load_ecgfm(device):
    from fairseq_signals.utils.checkpoint_utils import load_model_and_task
    result = load_model_and_task(CKPT_FM)
    if isinstance(result, tuple):
        for r in result:
            if hasattr(r, "parameters"): return r.to(device)
            if isinstance(r, list) and r and hasattr(r[0], "parameters"): return r[0].to(device)
    return result.to(device)


# ── 평가 ──────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(backbone, head, loader, device):
    head.eval(); backbone.eval()
    all_logits, all_labels = [], []
    for x, y in loader:
        x = x.to(device)
        emb    = backbone(source=x, padding_mask=None, features_only=True)["x"].mean(dim=1)
        logits = head(emb).cpu()
        all_logits.append(logits); all_labels.append(y)
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy().astype(int)
    probs  = 1 / (1 + np.exp(-logits))
    if len(np.unique(labels)) < 2:
        return float("nan"), float("nan"), float("nan")
    auroc = roc_auc_score(labels, probs)
    f1    = f1_score(labels, (probs >= 0.5).astype(int), zero_division=0)
    fpr, tpr, _ = roc_curve(labels, probs)
    spec = 1 - fpr
    idx  = np.searchsorted(spec[::-1], 0.95)
    sens = float(tpr[::-1][idx]) if idx < len(tpr) else float("nan")
    return auroc, f1, sens


# ── 메인 ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs",       type=int,   default=30)
    parser.add_argument("--batch_size",   type=int,   default=16)
    parser.add_argument("--lr",           type=float, default=1e-4,
                        help="warm start이므로 기본 1e-4 (cold start 5e-4보다 낮게)")
    parser.add_argument("--lora_rank",    type=int,   default=8)
    parser.add_argument("--lora_alpha",   type=float, default=16.0)
    parser.add_argument("--rlm_p",        type=float, default=0.5)
    parser.add_argument("--p_noise",      type=float, default=0.75)
    parser.add_argument("--snr_levels",   nargs="+",  type=float,
                        default=[24.0, 18.0, 12.0, 6.0, 0.0])
    parser.add_argument("--ptbxl_ratio",  type=float, default=1.0,
                        help="PTB-XL 샘플 가중치 (1.0=CPSC와 동수, 0.5=절반)")
    parser.add_argument("--ckpt_start",   default=f"{OUTPUTS}/lora_multisnr/lora_multisnr_best.pt",
                        help="warm start 체크포인트 경로 (빈 문자열이면 cold start)")
    parser.add_argument("--seed",         type=int,   default=SEED)
    parser.add_argument("--out_dir",      default=OUT_DIR)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 65)
    print("CPSC + PTB-XL 혼합 학습 (LoRA+RLM+multi-SNR)")
    print("=" * 65)
    print(f"디바이스: {device}  |  seed={args.seed}")
    print(f"PTB-XL 비율: {args.ptbxl_ratio}  |  epochs={args.epochs}")
    print()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── 데이터 로드 ─────────────────────────────────────────────────────────
    cpsc_ds  = ECGDataset(CPSC_TRAIN)
    ptbxl_ds = ECGDataset(PTBXL_TRAIN)
    val_ds   = ECGDataset(CPSC_VAL)
    test_ds  = ECGDataset(CPSC_TEST)

    n_cpsc  = len(cpsc_ds)
    n_ptbxl = len(ptbxl_ds)
    print(f"[데이터] CPSC  train={n_cpsc}, val={len(val_ds)}, test={len(test_ds)}")
    print(f"         PTB-XL train={n_ptbxl}")

    # WeightedRandomSampler: CPSC:PTB-XL = 1:ptbxl_ratio (에폭당 샘플 수 고정)
    combined = ConcatDataset([cpsc_ds, ptbxl_ds])
    weights  = [1.0] * n_cpsc + [args.ptbxl_ratio] * n_ptbxl
    num_samples = int(n_cpsc * (1 + args.ptbxl_ratio))
    sampler  = WeightedRandomSampler(weights, num_samples=num_samples, replacement=True)
    train_loader = DataLoader(combined, batch_size=args.batch_size,
                              sampler=sampler, num_workers=0, pin_memory=True)
    val_loader  = DataLoader(val_ds,  batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

    # pos_weight (혼합 전체 기준)
    all_labels = np.concatenate([cpsc_ds.labels, ptbxl_ds.labels])
    n_pos = (all_labels == 1).sum(); n_neg = (all_labels == 0).sum()
    pos_weight = torch.tensor(n_neg / n_pos, dtype=torch.float32).to(device)
    print(f"         혼합 응급={n_pos}, 정상={n_neg}, pos_weight={pos_weight.item():.3f}")

    # 외부검증 — 학습 중 로드하지 않음, 최종 평가 시 1회 로드
    print("         외부검증(CACHET/INCART/STAFF-III/LTST): 최종 평가 시 로드")
    print()

    # ── 노이즈 템플릿 ──────────────────────────────────────────────────────
    noise_templates = {}
    if os.path.isdir(NSTDB_DIR):
        try:
            noise_templates = load_noise_templates(NSTDB_DIR)
            n_tmpl = sum(len(v) for v in noise_templates.values())
            print(f"[노이즈] NSTDB 템플릿 {n_tmpl}개 로드")
        except Exception as e:
            print(f"[노이즈] 로드 실패 ({e}) — 증강 없이 진행")
    print()

    # ── 모델 ──────────────────────────────────────────────────────────────
    print("[모델] ECG-FM 로드 중...")
    backbone = load_ecgfm(device)
    inject_lora(backbone, rank=args.lora_rank, alpha=args.lora_alpha, dropout=0.1)
    backbone.train()
    trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in backbone.parameters())
    print(f"       LoRA 주입: {trainable:,} / {total:,} 파라미터 학습")
    print()

    head      = LinearHead().to(device)

    # warm start: 기존 CPSC 학습 체크포인트에서 초기화
    if args.ckpt_start and os.path.exists(args.ckpt_start):
        ckpt_warm = torch.load(args.ckpt_start, map_location=device)
        backbone.load_state_dict(ckpt_warm["backbone_lora"], strict=False)
        head.load_state_dict(ckpt_warm["head_state"])
        print(f"[Warm start] {args.ckpt_start}")
        print(f"             val AUROC={ckpt_warm.get('val_auroc', 'n/a'):.4f} (원본 CPSC 학습)")
    else:
        print("[Cold start] 랜덤 LoRA 초기화")
    print()

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        list(p for p in backbone.parameters() if p.requires_grad) + list(head.parameters()),
        lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=5e-5)

    best_auroc = 0.0; best_epoch = 0
    best_path  = os.path.join(args.out_dir, "lora_mixed_best.pt")

    print(f"{'Ep':>3} {'Loss':>8} {'ValAUROC':>9} {'ValF1':>7}")
    print("-" * 35)

    for epoch in range(1, args.epochs + 1):
        backbone.train(); head.train()
        total_loss = 0.0; n_samples = 0
        t0 = time.time()

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            # RLM
            if args.rlm_p > 0:
                mask = torch.rand(x.shape[0], 12, device=device) < args.rlm_p
                x[mask.unsqueeze(-1).expand_as(x)] = 0.0

            # multi-SNR 증강
            if noise_templates and random.random() < args.p_noise:
                snr = random.choice(args.snr_levels)
                x   = add_noise(x, noise_templates, snr, device)

            emb    = backbone(source=x, padding_mask=None, features_only=True)["x"].mean(dim=1)
            logits = head(emb)
            loss   = criterion(logits, y)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(p for p in backbone.parameters() if p.requires_grad)
                + list(head.parameters()), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * len(y)
            n_samples  += len(y)

        scheduler.step()
        avg_loss = total_loss / max(n_samples, 1)
        val_auroc, val_f1, _ = evaluate(backbone, head, val_loader, device)

        marker = " ←" if val_auroc > best_auroc else ""
        print(f"{epoch:3d} {avg_loss:8.4f} {val_auroc:9.4f} {val_f1:7.4f}{marker}", flush=True)

        if val_auroc > best_auroc:
            best_auroc = val_auroc; best_epoch = epoch
            torch.save({
                "epoch":          epoch,
                "head_state":     head.state_dict(),
                "backbone_lora":  backbone.state_dict(),
                "val_auroc":      val_auroc,
                "lora_rank":      args.lora_rank,
                "lora_alpha":     args.lora_alpha,
                "ptbxl_ratio":    args.ptbxl_ratio,
                "train_data":     "CPSC2018+PTB-XL",
            }, best_path)

    # ── 최종 평가 ─────────────────────────────────────────────────────────
    print()
    print(f"최고 val AUROC: {best_auroc:.4f} (epoch {best_epoch})")
    ckpt = torch.load(best_path, map_location=device)
    backbone.load_state_dict(ckpt["backbone_lora"], strict=False)
    head.load_state_dict(ckpt["head_state"])

    print()
    print("=" * 65)
    print("최종 결과 — best checkpoint (val AUROC 기준)")
    print("=" * 65)
    print()
    print("  [비교 기준: CPSC 단일 ③ lora_multisnr]")
    print("    CPSC test  AUROC=0.9463  F1=0.9155  Sens@95Sp=0.7620")
    print()
    print("  [혼합 모델: CPSC+PTB-XL lora_mixed]")

    a, f, s = evaluate(backbone, head, test_loader, device)
    print(f"    CPSC test  AUROC={a:.4f}  F1={f:.4f}  Sens@95Sp={s:.4f}")
    print()

    # 외부검증 — 최종 1회, 필요한 것만 RAM에 올려 평가 후 즉시 해제
    for db_name, db_dir in EXT_DIRS.items():
        if not os.path.isdir(db_dir):
            continue
        try:
            ext_ds = ECGDataset(db_dir)
            ext_loader = DataLoader(ext_ds, batch_size=64, shuffle=False, num_workers=0)
            a, f, s = evaluate(backbone, head, ext_loader, device)
            s_str = f"{s:.4f}" if not np.isnan(s) else " n/a"
            print(f"    {db_name:12s} AUROC={a:.4f}  F1={f:.4f}  Sens@95Sp={s_str}", flush=True)
            del ext_ds, ext_loader  # 즉시 메모리 해제
        except Exception as e:
            print(f"    {db_name:12s} 로드 실패: {e}")

    print()
    print(f"  체크포인트: {best_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
