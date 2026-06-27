"""
단계 5e: 멀티헤드 단일 백본 학습 — multi-label 다중분류 (이진 + multi-label + multi-SNR)
====================================================================================
★2026-05-29 신규 (구 5d+SNR train_lora_multitask.py의 multi-label 버전):
  근본원인 분석(records/01 §다중분류 약클래스) 후속 — 데이터 결함 수정분
  (cpsc2018_mc_ml: 164884008 복구 + multi-hot 라벨) 위에서 재학습.

구조:
  ECG-FM (frozen) → LoRA(rank=8) → mean pool (768)
                                    ├── BinaryHead(768→1)     sigmoid: 이진 응급
                                    └── MulticlassHead(768→5) sigmoid: multi-label 5-class
  손실: loss = α·BCE(이진) + (1−α)·BCE_multilabel(다중, 클래스별 pos_weight)
        ※ 구 버전의 softmax CE → 독립 sigmoid BCE (동반 진단 허용)

지표 재정의(미션 정렬, records/01 §다중분류 약클래스):
  - 헤드라인: 이진 응급 AUROC, AF AUROC, 허혈 AUROC (응급 관련)
  - 보조: macro-AUROC, per-class F1(val 튜닝 임계값), 응급가중 F1(AF+허혈)
  - best 선택 composite = (bin_auroc + macro_auroc)/2  (threshold-free)

데이터:
  data/processed/cpsc2018_mc_ml/
    signals.npy, labels_mc.npy (N,5 multi-hot), labels_bin.npy (0/1: AF·허혈→1)

Warm start (기본):
  outputs/lora_multitask_snr/lora_multitask_snr_best.pt
  → backbone_lora + head_bin + head_mc 재사용 (multi-SNR 강건성 + 이진 성능 계승)

사용법:
  python scripts/train_lora_multitask_ml.py
  python scripts/train_lora_multitask_ml.py --alpha 0.5 --lr 1e-4
"""

import argparse
import math
import os
import os as _os
import sys
import sys as _sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

_sys.path.insert(0, _os.path.dirname(__file__))
from multisnr import MultiSNRNoise

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

try:
    from sklearn.metrics import f1_score, roc_auc_score, roc_curve
except ImportError:
    sys.exit("[오류] scikit-learn 미설치")

EMBED_DIM = 768
N_CLASSES = 5
CLASS_NAMES = [
    "정상(NSR)",
    "AF",
    "급성허혈(STD/STE)",
    "전도장애(I-AVB/LBBB/RBBB)",
    "이소성(PAC/PVC)",
]
EMERGENCY_CLASSES = (1, 2)  # AF + 급성허혈

CKPT_FM = "checkpoints/ecg-fm/mimic_iv_ecg_physionet_pretrained.pt"
DATA_DIR = "data/processed/cpsc2018_mc_ml"
NSTDB_DIR = "data/raw/nstdb"
OUT_DIR = "outputs/lora_multitask_ml"
WARM_CKPT = "outputs/lora_multitask_snr/lora_multitask_snr_best.pt"


# ── LoRA ─────────────────────────────────────────────────────────────────
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


# ── 데이터셋: 이진 + multi-hot 라벨 동시 반환 ─────────────────────────────
class CPSCMultiLabelDataset(Dataset):
    def __init__(self, split_dir: str):
        self.signals = np.load(os.path.join(split_dir, "signals.npy"))
        self.labels_mc = np.load(os.path.join(split_dir, "labels_mc.npy"))  # (N,5)
        self.labels_bin = np.load(os.path.join(split_dir, "labels_bin.npy"))  # (N,)

    def __len__(self):
        return len(self.labels_mc)

    def __getitem__(self, idx):
        x = torch.tensor(self.signals[idx], dtype=torch.float32)
        yb = torch.tensor(float(self.labels_bin[idx]), dtype=torch.float32)
        ym = torch.tensor(self.labels_mc[idx], dtype=torch.float32)  # (5,) 멀티핫
        return x, yb, ym


def random_lead_mask(x, p=0.5):
    B, C, T = x.shape
    mask = (torch.rand(B, C, 1, device=x.device) > p).float()
    return x * mask


# ── ECG-FM 로드 ──────────────────────────────────────────────────────────
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


def extract_embedding(backbone, x):
    out = backbone(source=x, padding_mask=None, features_only=True)
    return out["x"].mean(dim=1)


# ── 헤드 ─────────────────────────────────────────────────────────────────
class BinaryHead(nn.Module):
    def __init__(self, in_dim=EMBED_DIM):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, x):
        return self.fc(x).squeeze(-1)


class MulticlassHead(nn.Module):
    def __init__(self, in_dim=EMBED_DIM, n_classes=N_CLASSES):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        return self.fc(x)  # logits — sigmoid는 손실/평가에서


# ── 평가 ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def infer(backbone, head_bin, head_mc, loader, device):
    backbone.eval()
    head_bin.eval()
    head_mc.eval()
    all_bin, all_mc, all_yb, all_ym = [], [], [], []
    for x, yb, ym in loader:
        x = x.to(device)
        emb = extract_embedding(backbone, x)
        all_bin.append(head_bin(emb).cpu())
        all_mc.append(head_mc(emb).cpu())
        all_yb.append(yb)
        all_ym.append(ym)
    bin_logits = torch.cat(all_bin).numpy()
    mc_logits = torch.cat(all_mc).numpy()
    yb = torch.cat(all_yb).numpy().astype(int)
    ym = torch.cat(all_ym).numpy().astype(int)  # (N,5)
    bin_probs = 1 / (1 + np.exp(-bin_logits))
    mc_probs = 1 / (1 + np.exp(-mc_logits))  # 독립 sigmoid (multi-label)
    return bin_probs, yb, mc_probs, ym


def tune_thresholds(mc_probs, mc_labels, grid=None):
    """클래스별 F1 최대화 임계값 (val에서 탐색)."""
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


def compute_metrics(bin_probs, yb, mc_probs, ym, mc_thresholds=None):
    # 이진
    if len(np.unique(yb)) >= 2:
        bin_auroc = roc_auc_score(yb, bin_probs)
        bin_f1 = f1_score(yb, (bin_probs >= 0.5).astype(int), zero_division=0)
        fpr, tpr, _ = roc_curve(yb, bin_probs)
        spec = 1 - fpr
        idx = np.searchsorted(spec[::-1], 0.95)
        bin_sens = float(tpr[::-1][idx]) if idx < len(tpr) else float("nan")
    else:
        bin_auroc = bin_f1 = bin_sens = float("nan")

    # multi-label per-class
    if mc_thresholds is None:
        mc_thresholds = np.full(N_CLASSES, 0.5)
    per_auroc, per_f1 = [], []
    for c in range(N_CLASSES):
        yc = ym[:, c]
        if 0 < yc.sum() < len(yc):
            per_auroc.append(roc_auc_score(yc, mc_probs[:, c]))
        else:
            per_auroc.append(float("nan"))
        per_f1.append(
            f1_score(
                yc, (mc_probs[:, c] >= mc_thresholds[c]).astype(int), zero_division=0
            )
        )
    per_auroc = np.array(per_auroc)
    per_f1 = np.array(per_f1)
    macro_auroc = np.nanmean(per_auroc)
    macro_f1 = float(np.mean(per_f1))
    # 응급 관련 (AF=1, 허혈=2)
    emerg_auroc = float(np.nanmean(per_auroc[list(EMERGENCY_CLASSES)]))
    emerg_f1 = float(np.mean(per_f1[list(EMERGENCY_CLASSES)]))
    return {
        "bin_auroc": bin_auroc,
        "bin_f1": bin_f1,
        "bin_sens": bin_sens,
        "per_auroc": per_auroc,
        "per_f1": per_f1,
        "macro_auroc": macro_auroc,
        "macro_f1": macro_f1,
        "emerg_auroc": emerg_auroc,
        "emerg_f1": emerg_f1,
        "mc_thresholds": mc_thresholds,
    }


# ── 학습 ─────────────────────────────────────────────────────────────────
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("단계 5e: 멀티헤드 단일 백본 — multi-label (이진 + 5-class + multi-SNR)")
    print("=" * 70)
    print(f"디바이스:   {device}")
    print(f"데이터:     {args.data_dir}")
    print(f"warm start: {args.warm_ckpt if args.warm_ckpt else '없음 (cold start)'}")
    print(
        f"alpha (BCE 비중): {args.alpha}  →  loss = {args.alpha}·BCE_bin + {1 - args.alpha:.2f}·BCE_mc"
    )
    print(f"LR={args.lr}, epochs={args.epochs}, batch={args.batch_size}")
    snr_set = tuple(int(s) for s in args.snr_set.split(","))
    print(f"multi-SNR:  {snr_set}dB, p_noise={args.p_noise}")
    print()

    os.makedirs(args.out_dir, exist_ok=True)

    train_ds = CPSCMultiLabelDataset(os.path.join(args.data_dir, "train"))
    val_ds = CPSCMultiLabelDataset(os.path.join(args.data_dir, "val"))
    test_ds = CPSCMultiLabelDataset(os.path.join(args.data_dir, "test"))

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

    # multi-label 클래스별 pos_weight (역빈도)
    mc = train_ds.labels_mc.astype(np.float64)  # (N,5)
    n_pos_c = mc.sum(axis=0)
    n_neg_c = len(mc) - n_pos_c
    mc_pos_weight = torch.tensor(
        n_neg_c / np.maximum(n_pos_c, 1), dtype=torch.float32
    ).to(device)

    # 이진 pos_weight
    n_pos = (train_ds.labels_bin == 1).sum()
    n_neg = (train_ds.labels_bin == 0).sum()
    pos_weight = torch.tensor(n_neg / max(n_pos, 1), dtype=torch.float32).to(device)

    print(f"[데이터] train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    print(
        f"         이진 응급={int(n_pos)}, 정상={int(n_neg)}, pos_weight={pos_weight.item():.3f}"
    )
    print("[multi-label 클래스별 pos_weight]")
    for c in range(N_CLASSES):
        print(
            f"  [{c}] {CLASS_NAMES[c]:30s} n_pos={int(n_pos_c[c]):4d}  pos_w={mc_pos_weight[c].item():.3f}"
        )
    print()

    print("[모델] ECG-FM 백본 로드 중...")
    backbone = load_ecgfm(args.ckpt_path, device)
    backbone.train()
    for p in backbone.parameters():
        p.requires_grad_(False)
    replaced = inject_lora(
        backbone, rank=args.lora_rank, alpha=args.lora_alpha, dropout=args.lora_dropout
    )
    print(
        f"       LoRA 주입: {len(replaced)} 레이어, "
        f"학습 파라미터 {sum(p.numel() for p in backbone.parameters() if p.requires_grad):,}"
    )

    head_bin = BinaryHead().to(device)
    head_mc = MulticlassHead().to(device)
    print("       BinaryHead(768→1) + MulticlassHead(768→5, sigmoid) 추가")

    if args.warm_ckpt and os.path.exists(args.warm_ckpt):
        warm = torch.load(args.warm_ckpt, map_location=device)
        backbone.load_state_dict(warm["backbone_lora"], strict=False)
        if "head_bin_state" in warm:
            head_bin.load_state_dict(warm["head_bin_state"])
        elif "head_state" in warm:
            head_bin.load_state_dict(warm["head_state"])
        if "head_mc_state" in warm and not args.fresh_mc_head:
            head_mc.load_state_dict(warm["head_mc_state"])
            print(f"[Warm start] {args.warm_ckpt}")
            print("             백본 LoRA + head_bin + head_mc 재사용")
        else:
            print(f"[Warm start] {args.warm_ckpt}")
            print("             백본 LoRA + head_bin 재사용, head_mc 신규 초기화")
    else:
        print("[Cold start] 모든 LoRA + head 랜덤 초기화")
    print()

    params = (
        [p for p in backbone.parameters() if p.requires_grad]
        + list(head_bin.parameters())
        + list(head_mc.parameters())
    )
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.1
    )
    bce_bin = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    bce_mc = nn.BCEWithLogitsLoss(pos_weight=mc_pos_weight)  # multi-label

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print("[증강] NSTDB 노이즈 로드 + 500Hz 리샘플 중...")
    multisnr = MultiSNRNoise(
        nstdb_dir=args.nstdb_dir, snr_set=snr_set, device=device, seed=args.seed
    )
    print(
        "       pool 길이: "
        + ", ".join(
            f"{t}={multisnr.noise_pool[t].shape[0]:,}" for t in ("bw", "em", "ma")
        )
    )
    print()

    best_composite = 0.0
    best_epoch = 0
    best_path = os.path.join(args.out_dir, "lora_multitask_ml_best.pt")

    print(
        f"{'Ep':>3} {'Loss':>8} {'BCEb':>6} {'BCEm':>6}  "
        f"{'BinAUROC':>9} {'MacAUROC':>9} {'Compose':>8}"
    )
    print("-" * 62)

    for epoch in range(1, args.epochs + 1):
        backbone.train()
        head_bin.train()
        head_mc.train()
        tot, tot_b, tot_c, n = 0.0, 0.0, 0.0, 0

        for x, yb, ym in train_loader:
            x = x.to(device)
            yb = yb.to(device)
            ym = ym.to(device)
            x = multisnr.inject(x, p_noise=args.p_noise)
            if args.rlm_p > 0:
                x = random_lead_mask(x, p=args.rlm_p)

            emb = extract_embedding(backbone, x)
            bin_logits = head_bin(emb)
            mc_logits = head_mc(emb)
            lb = bce_bin(bin_logits, yb)
            lc = bce_mc(mc_logits, ym)
            loss = args.alpha * lb + (1 - args.alpha) * lc

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

            bs = x.size(0)
            tot += loss.item() * bs
            tot_b += lb.item() * bs
            tot_c += lc.item() * bs
            n += bs

        scheduler.step()
        bp, yb_v, mp, ym_v = infer(backbone, head_bin, head_mc, val_loader, device)
        m = compute_metrics(bp, yb_v, mp, ym_v)
        composite = (m["bin_auroc"] + m["macro_auroc"]) / 2

        marker = " ←" if composite > best_composite else ""
        print(
            f"{epoch:3d} {tot / n:8.4f} {tot_b / n:6.4f} {tot_c / n:6.4f}  "
            f"{m['bin_auroc']:9.4f} {m['macro_auroc']:9.4f} {composite:8.4f}{marker}",
            flush=True,
        )

        if composite > best_composite:
            best_composite = composite
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "backbone_lora": backbone.state_dict(),
                    "head_bin_state": head_bin.state_dict(),
                    "head_mc_state": head_mc.state_dict(),
                    "val_bin_auroc": m["bin_auroc"],
                    "val_macro_auroc": m["macro_auroc"],
                    "val_composite": composite,
                    "alpha": args.alpha,
                    "lora_rank": args.lora_rank,
                    "lora_alpha": args.lora_alpha,
                    "n_classes": N_CLASSES,
                    "class_names": CLASS_NAMES,
                    "emergency_classes": EMERGENCY_CLASSES,
                    "multilabel": True,
                },
                best_path,
            )

    # ── 테스트 ────────────────────────────────────────────────────────────
    print()
    print(f"최고 val composite: {best_composite:.4f} (epoch {best_epoch})")
    ckpt = torch.load(best_path, map_location=device)
    backbone.load_state_dict(ckpt["backbone_lora"], strict=False)
    head_bin.load_state_dict(ckpt["head_bin_state"])
    head_mc.load_state_dict(ckpt["head_mc_state"])

    # 임계값은 val에서 튜닝 → test 적용 (누수 방지)
    bp_v, ybv, mp_v, ymv = infer(backbone, head_bin, head_mc, val_loader, device)
    mc_th = tune_thresholds(mp_v, ymv)
    bp_t, ybt, mp_t, ymt = infer(backbone, head_bin, head_mc, test_loader, device)
    res = compute_metrics(bp_t, ybt, mp_t, ymt, mc_thresholds=mc_th)

    print()
    print("=" * 70)
    print("단계 5e 결과 (CPSC mc_ml test, best ckpt) — multi-label")
    print("=" * 70)
    print()
    print("  [이진 응급] ★ 헤드라인")
    print(f"    AUROC    = {res['bin_auroc']:.4f}")
    print(f"    F1@0.5   = {res['bin_f1']:.4f}")
    print(f"    Sens@95Sp= {res['bin_sens']:.4f}")
    print()
    print("  [응급 관련 클래스 (AF + 허혈)]")
    print(f"    응급 macro-AUROC = {res['emerg_auroc']:.4f}")
    print(f"    응급 macro-F1    = {res['emerg_f1']:.4f}")
    print()
    print("  [multi-label 5-class — per class]")
    print(f"    {'class':<28}{'AUROC':>8}{'F1':>8}{'thr':>7}")
    for c in range(N_CLASSES):
        print(
            f"    {CLASS_NAMES[c]:<28}{res['per_auroc'][c]:>8.4f}"
            f"{res['per_f1'][c]:>8.4f}{mc_th[c]:>7.2f}"
        )
    print(f"    {'─ macro':<28}{res['macro_auroc']:>8.4f}{res['macro_f1']:>8.4f}")
    print()
    print("  [비교 기준 (구 단일라벨 5b/5d, 테스트셋 상이 — 근사)]")
    print("    5b 단일:  Macro-F1=0.6762, 이소성 F1=0.3478, 이소성 AUROC=0.8620")
    print("    5d+SNR:   이진 AUROC=0.9134, Macro-F1=0.6834, 이소성 AUROC=0.8662")
    print()
    print(f"  체크포인트: {best_path}")
    print("=" * 70)

    np.savez(
        os.path.join(args.out_dir, "test_results.npz"),
        bin_labels=ybt,
        bin_probs=bp_t,
        mc_labels=ymt,
        mc_probs=mp_t,
        mc_thresholds=mc_th,
        bin_auroc=res["bin_auroc"],
        bin_f1=res["bin_f1"],
        bin_sens=res["bin_sens"],
        per_auroc=res["per_auroc"],
        per_f1=res["per_f1"],
        macro_auroc=res["macro_auroc"],
        macro_f1=res["macro_f1"],
        emerg_auroc=res["emerg_auroc"],
        emerg_f1=res["emerg_f1"],
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data_dir", default=DATA_DIR)
    parser.add_argument("--ckpt_path", default=CKPT_FM)
    parser.add_argument("--out_dir", default=OUT_DIR)
    parser.add_argument(
        "--warm_ckpt",
        default=WARM_CKPT,
        help="warm start 체크포인트 (빈 문자열이면 cold start)",
    )
    parser.add_argument(
        "--fresh_mc_head",
        action="store_true",
        help="head_mc를 warm에서 로드하지 않고 신규 초기화",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="BCE_bin 비중 (0=다중만, 1=이진만, 0.5=균등)",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--rlm_p", type=float, default=0.5)
    parser.add_argument("--nstdb_dir", default=NSTDB_DIR)
    parser.add_argument("--p_noise", type=float, default=0.75)
    parser.add_argument("--snr_set", type=str, default="24,18,12,6,0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
