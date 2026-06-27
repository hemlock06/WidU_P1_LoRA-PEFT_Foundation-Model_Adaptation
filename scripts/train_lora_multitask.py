"""
단계 5d+SNR: 멀티헤드 단일 백본 학습 (이진 + 다중분류 + multi-SNR)
=================================================================
목적:
  연구계획서(records/00_research_plan.md §1)의 핵심 설계 — "동일 백본 위에
  이진 응급 헤드 + 다중분류 5-class 헤드 병행" — 에 multi-SNR 모션 증강을 추가.
  단계 5d(증강 없음)에서 학습된 멀티헤드 체크포인트를 warm start로 활용하여
  노이즈 강건성을 추가 이식한다.

구조:
  ECG-FM (frozen) → LoRA(rank=8) → mean pool (768)
                                    ├── BinaryHead(768→1)     sigmoid: 이진 응급
                                    └── MulticlassHead(768→5) softmax: 5-class

  증강 순서: multi-SNR 노이즈 주입 → RLM 마스킹 → ECG-FM+LoRA+헤드
  손실: loss = α · BCE(이진) + (1−α) · CE_w(다중)
        α 기본 0.5, multi CE는 클래스 가중(역빈도)

데이터:
  data/processed/cpsc2018_mc/
    signals.npy, labels.npy (0~4), labels_bin.npy (0/1: AF·허혈→1)

Warm start (기본):
  outputs/lora_multitask/lora_multitask_best.pt
  → backbone_lora + 이진/다중 head 모두 재사용 (multi-SNR 추가 계속학습)

평가:
  - 이진: AUROC, F1@0.5, Sens@95Sp
  - 다중: Macro-F1, Weighted-F1, Per-class AUROC, Confusion Matrix
  - best 선택: composite = (bin_AUROC + multi_macro_F1) / 2

사용법:
  python scripts/train_lora_multitask.py
  python scripts/train_lora_multitask.py --alpha 0.3 --lr 1e-4 --p_noise 0.75
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
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        f1_score,
        roc_auc_score,
        roc_curve,
    )
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
DATA_DIR = "data/processed/cpsc2018_mc"
NSTDB_DIR = "data/raw/nstdb"
OUT_DIR = "outputs/lora_multitask_snr"
WARM_CKPT = "outputs/lora_multitask/lora_multitask_best.pt"


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


# ── 데이터셋: 이진 + 다중 라벨 동시 반환 ──────────────────────────────────
class CPSCMultiTaskDataset(Dataset):
    def __init__(self, split_dir: str, mmap: bool = True):
        # mmap=True: signals를 RAM에 적재하지 않고 디스크에서 샘플 단위 읽기 (OS page-cache로
        # 속도 유지, 메모리 압박 시 회수 → 16GB RAM OOM 방지). 학습 수식 불변.
        self.signals = np.load(
            os.path.join(split_dir, "signals.npy"), mmap_mode="r" if mmap else None
        )
        self.labels_mc = np.load(os.path.join(split_dir, "labels.npy"))
        self.labels_bin = np.load(os.path.join(split_dir, "labels_bin.npy"))

    def __len__(self):
        return len(self.labels_mc)

    def __getitem__(self, idx):
        x = torch.tensor(np.ascontiguousarray(self.signals[idx]), dtype=torch.float32)
        yb = torch.tensor(float(self.labels_bin[idx]), dtype=torch.float32)
        ym = torch.tensor(int(self.labels_mc[idx]), dtype=torch.long)
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
        return self.fc(x)


# ── 평가 ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(backbone, head_bin, head_mc, loader, device):
    backbone.eval()
    head_bin.eval()
    head_mc.eval()
    all_bin_logits, all_mc_logits = [], []
    all_yb, all_ym = [], []

    for x, yb, ym in loader:
        x = x.to(device)
        emb = extract_embedding(backbone, x)
        bin_logits = head_bin(emb).cpu()
        mc_logits = head_mc(emb).cpu()
        all_bin_logits.append(bin_logits)
        all_mc_logits.append(mc_logits)
        all_yb.append(yb)
        all_ym.append(ym)

    if device.type == "cuda":
        torch.cuda.empty_cache()  # 장기 실행 단편화 누적 방지
    bin_logits = torch.cat(all_bin_logits).numpy()
    mc_logits = torch.cat(all_mc_logits).numpy()
    yb = torch.cat(all_yb).numpy().astype(int)
    ym = torch.cat(all_ym).numpy().astype(int)

    # 이진 평가
    bin_probs = 1 / (1 + np.exp(-bin_logits))
    if len(np.unique(yb)) >= 2:
        bin_auroc = roc_auc_score(yb, bin_probs)
        bin_f1 = f1_score(yb, (bin_probs >= 0.5).astype(int), zero_division=0)
        fpr, tpr, _ = roc_curve(yb, bin_probs)
        spec = 1 - fpr
        idx = np.searchsorted(spec[::-1], 0.95)
        bin_sens = float(tpr[::-1][idx]) if idx < len(tpr) else float("nan")
    else:
        bin_auroc = bin_f1 = bin_sens = float("nan")

    # 다중 평가
    mc_probs = torch.softmax(torch.tensor(mc_logits), dim=-1).numpy()
    mc_preds = mc_probs.argmax(axis=-1)
    macro_f1 = f1_score(ym, mc_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(ym, mc_preds, average="weighted", zero_division=0)
    per_auroc = []
    for c in range(N_CLASSES):
        yc = (ym == c).astype(int)
        if 0 < yc.sum() < len(yc):
            per_auroc.append(roc_auc_score(yc, mc_probs[:, c]))
        else:
            per_auroc.append(float("nan"))

    return {
        "bin_auroc": bin_auroc,
        "bin_f1": bin_f1,
        "bin_sens": bin_sens,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_auroc": per_auroc,
        "mc_preds": mc_preds,
        "mc_labels": ym,
        "mc_probs": mc_probs,
        "bin_probs": bin_probs,
        "bin_labels": yb,
    }


# ── 학습 ─────────────────────────────────────────────────────────────────
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("단계 5d+SNR: 멀티헤드 단일 백본 학습 (이진 + 다중분류 + multi-SNR)")
    print("=" * 70)
    print(f"디바이스:   {device}")
    print(f"데이터:     {args.data_dir}")
    print(f"NSTDB:      {args.nstdb_dir}")
    print(f"warm start: {args.warm_ckpt if args.warm_ckpt else '없음 (cold start)'}")
    print(
        f"alpha (BCE 비중): {args.alpha}  →  loss = {args.alpha}·BCE + {1 - args.alpha:.2f}·CE"
    )
    print(f"LR={args.lr}, epochs={args.epochs}, batch={args.batch_size}")
    print(f"LoRA rank={args.lora_rank}, alpha={args.lora_alpha}, RLM p={args.rlm_p}")
    snr_set = tuple(int(s) for s in args.snr_set.split(","))
    print(
        f"multi-SNR:  {snr_set}dB, p_noise={args.p_noise} ({int((1 - args.p_noise) * 100)}% clean 유지)"
    )
    print()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── 데이터 ────────────────────────────────────────────────────────────
    train_ds = CPSCMultiTaskDataset(os.path.join(args.data_dir, "train"))
    val_ds = CPSCMultiTaskDataset(os.path.join(args.data_dir, "val"))
    test_ds = CPSCMultiTaskDataset(os.path.join(args.data_dir, "test"))

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    # 클래스 가중치 (multi, 역빈도)
    counts = np.array(
        [(train_ds.labels_mc == c).sum() for c in range(N_CLASSES)], dtype=np.float64
    )
    mc_w = len(train_ds.labels_mc) / (N_CLASSES * counts)
    class_weights = torch.tensor(mc_w, dtype=torch.float32).to(device)

    # 이진 pos_weight
    n_pos = (train_ds.labels_bin == 1).sum()
    n_neg = (train_ds.labels_bin == 0).sum()
    pos_weight = torch.tensor(n_neg / max(n_pos, 1), dtype=torch.float32).to(device)

    print(f"[데이터] train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    print(
        f"         이진 응급={int(n_pos)}, 정상={int(n_neg)}, pos_weight={pos_weight.item():.3f}"
    )
    print("[다중분류 가중치]")
    for c in range(N_CLASSES):
        print(f"  [{c}] {CLASS_NAMES[c]:30s} n={int(counts[c]):4d}  w={mc_w[c]:.4f}")
    print()

    # ── 모델 ──────────────────────────────────────────────────────────────
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
    print("       BinaryHead(768→1) + MulticlassHead(768→5) 추가")

    # warm start
    if args.warm_ckpt and os.path.exists(args.warm_ckpt):
        warm = torch.load(args.warm_ckpt, map_location=device)
        backbone.load_state_dict(warm["backbone_lora"], strict=False)
        # lora_multitask 형식: head_bin_state / lora_multisnr 형식: head_state
        if "head_bin_state" in warm:
            head_bin.load_state_dict(warm["head_bin_state"])
        elif "head_state" in warm:
            head_bin.load_state_dict(warm["head_state"])
        if "head_mc_state" in warm:
            head_mc.load_state_dict(warm["head_mc_state"])
        val_ref = warm.get("val_composite", warm.get("val_auroc", "n/a"))
        print(f"[Warm start] {args.warm_ckpt}")
        if isinstance(val_ref, float):
            print(f"             원본 val composite/AUROC={val_ref:.4f}")
        print(
            "             → 백본 LoRA + 이진/다중 head 재사용 (multi-SNR 추가 계속학습)"
        )
    else:
        print("[Cold start] 모든 LoRA + head 랜덤 초기화")
    print()

    # ── 옵티마이저 ────────────────────────────────────────────────────────
    params = (
        [p for p in backbone.parameters() if p.requires_grad]
        + list(head_bin.parameters())
        + list(head_mc.parameters())
    )
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.1
    )
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    ce_loss = nn.CrossEntropyLoss(weight=class_weights)

    # ── multi-SNR 증강기 ─────────────────────────────────────────────────
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print("[증강] NSTDB 노이즈 로드 + 500Hz 리샘플 중...")
    multisnr = MultiSNRNoise(
        nstdb_dir=args.nstdb_dir,
        snr_set=snr_set,
        device=device,
        seed=args.seed,
        noise_mode=args.noise_mode,
    )
    print(
        "       pool 길이: "
        + ", ".join(
            f"{t}={multisnr.noise_pool[t].shape[0]:,}" for t in ("bw", "em", "ma")
        )
    )
    print(
        f"       노이즈 모드: {args.noise_mode} "
        f"({'리드당 1종' if args.noise_mode == 'single' else 'bw·em·ma 가중합성(동시 중첩)'})"
    )
    print()

    best_composite = 0.0
    best_epoch = 0
    best_path = os.path.join(args.out_dir, "lora_multitask_snr_best.pt")
    resume_path = os.path.join(args.out_dir, "resume.pt")
    done_path = os.path.join(args.out_dir, "DONE.txt")
    start_epoch = 1

    # ── 자동 재개(self-healing): OOM 등으로 중단돼도 마지막 저장 epoch부터 이어감 ──
    if args.resume and os.path.exists(resume_path):
        rs = torch.load(resume_path, map_location=device)
        backbone.load_state_dict(rs["lora_state"], strict=False)
        head_bin.load_state_dict(rs["head_bin_state"])
        head_mc.load_state_dict(rs["head_mc_state"])
        optimizer.load_state_dict(rs["opt_state"])
        scheduler.load_state_dict(rs["sched_state"])
        start_epoch = rs["epoch"] + 1
        best_composite = rs["best_composite"]
        best_epoch = rs["best_epoch"]
        # rng 상태 복원 → 무중단과 동일한 노이즈/셔플 realization (결정적 resume)
        if "np_state" in rs:
            np.random.set_state(rs["np_state"])
            torch.set_rng_state(rs["torch_state"])
            if rs.get("cuda_state") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rs["cuda_state"])
            multisnr.rng.bit_generator.state = rs["multisnr_state"]
        print(
            f"[RESUME] epoch {start_epoch}부터 재개 (best={best_composite:.4f}@ep{best_epoch}, rng복원)"
        )

    print(
        f"{'Ep':>3} {'Loss':>8} {'BCE':>6} {'CE':>6}  "
        f"{'BinAUROC':>9} {'MacroF1':>8} {'Compose':>8}"
    )
    print("-" * 60)

    for epoch in range(start_epoch, args.epochs + 1):
        backbone.train()
        head_bin.train()
        head_mc.train()
        tot, tot_b, tot_c, n = 0.0, 0.0, 0.0, 0

        for bi, (x, yb, ym) in enumerate(train_loader):
            x = x.to(device)
            yb = yb.to(device)
            ym = ym.to(device)

            # ① multi-SNR 노이즈 주입 (per-sample 게이트 + per-lead 독립 SNR)
            x = multisnr.inject(x, p_noise=args.p_noise)
            # ② RLM 마스킹 (노이즈 다음 — 스펙 8-5 순서)
            if args.rlm_p > 0:
                x = random_lead_mask(x, p=args.rlm_p)

            emb = extract_embedding(backbone, x)
            bin_logits = head_bin(emb)
            mc_logits = head_mc(emb)
            lb = bce_loss(bin_logits, yb)
            lc = ce_loss(mc_logits, ym)
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
            if device.type == "cuda" and bi % 100 == 99:
                torch.cuda.empty_cache()  # 단편화 누적 방지 (epoch 내 주기적)

        scheduler.step()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        val = evaluate(backbone, head_bin, head_mc, val_loader, device)
        composite = (val["bin_auroc"] + val["macro_f1"]) / 2

        marker = " ←" if composite > best_composite else ""
        print(
            f"{epoch:3d} {tot / n:8.4f} {tot_b / n:6.4f} {tot_c / n:6.4f}  "
            f"{val['bin_auroc']:9.4f} {val['macro_f1']:8.4f} {composite:8.4f}{marker}",
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
                    "val_bin_auroc": val["bin_auroc"],
                    "val_macro_f1": val["macro_f1"],
                    "val_composite": composite,
                    "alpha": args.alpha,
                    "noise_mode": args.noise_mode,
                    "lora_rank": args.lora_rank,
                    "lora_alpha": args.lora_alpha,
                    "n_classes": N_CLASSES,
                    "class_names": CLASS_NAMES,
                    "emergency_classes": EMERGENCY_CLASSES,
                },
                best_path,
            )

        # 매 epoch resume 상태 저장 (LoRA delta+heads+opt+sched만 — 수 MB, 자동 재개용)
        torch.save(
            {
                "epoch": epoch,
                "lora_state": {
                    k: v for k, v in backbone.state_dict().items() if "lora_" in k
                },
                "head_bin_state": head_bin.state_dict(),
                "head_mc_state": head_mc.state_dict(),
                "opt_state": optimizer.state_dict(),
                "sched_state": scheduler.state_dict(),
                "best_composite": best_composite,
                "best_epoch": best_epoch,
                # rng 상태까지 저장 → resume이 나도 무중단과 동일(결정적). 섭동 방지.
                "np_state": np.random.get_state(),
                "torch_state": torch.get_rng_state(),
                "cuda_state": (
                    torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else None
                ),
                "multisnr_state": multisnr.rng.bit_generator.state,
            },
            resume_path,
        )

    # ── 학습 완료 마커 (test eval보다 먼저 — eval이 OOM나도 완료는 기록) ──
    with open(done_path, "w", encoding="utf-8") as _f:
        _f.write(
            f"trained {args.epochs} epochs, best composite={best_composite:.4f} @ep{best_epoch}\n"
        )
    print(f"[DONE] {done_path} — 학습 완료(전 epoch)")

    # ── 테스트 ────────────────────────────────────────────────────────────
    print()
    print(f"최고 val composite: {best_composite:.4f} (epoch {best_epoch})")
    ckpt = torch.load(best_path, map_location=device)
    backbone.load_state_dict(ckpt["backbone_lora"], strict=False)
    head_bin.load_state_dict(ckpt["head_bin_state"])
    head_mc.load_state_dict(ckpt["head_mc_state"])

    res = evaluate(backbone, head_bin, head_mc, test_loader, device)

    print()
    print("=" * 70)
    print("단계 5d 결과 (CPSC mc test, best ckpt)")
    print("=" * 70)
    print()
    print("  [이진 응급]")
    print(f"    AUROC    = {res['bin_auroc']:.4f}")
    print(f"    F1@0.5   = {res['bin_f1']:.4f}")
    print(f"    Sens@95Sp= {res['bin_sens']:.4f}")
    print()
    print("  [다중분류 5-class]")
    print(f"    Macro-F1     = {res['macro_f1']:.4f}")
    print(f"    Weighted-F1  = {res['weighted_f1']:.4f}")
    print()
    print("  Per-class AUROC:")
    for c in range(N_CLASSES):
        print(f"    [{c}] {CLASS_NAMES[c]:30s}: {res['per_auroc'][c]:.4f}")
    print()
    print("  Confusion Matrix (rows=true, cols=pred):")
    cm = confusion_matrix(
        res["mc_labels"], res["mc_preds"], labels=list(range(N_CLASSES))
    )
    print(f"    {'pred→':<3s} " + " ".join(f"{c:>5d}" for c in range(N_CLASSES)))
    for r in range(N_CLASSES):
        print(f"    [{r}]   " + " ".join(f"{cm[r, c]:>5d}" for c in range(N_CLASSES)))
    print()
    print("  Classification Report:")
    print(
        classification_report(
            res["mc_labels"],
            res["mc_preds"],
            target_names=CLASS_NAMES,
            zero_division=0,
            digits=4,
        )
    )
    print()
    print("  [비교 기준]")
    print(
        "    5d (멀티헤드, no-SNR):         AUROC=0.9140, Macro-F1=0.6840 (cpsc_mc task)"
    )
    print("    5b 다중 단일 (lora_mc):        Macro-F1=0.6762, 이진파생=0.9263")
    print(
        f"  [5d+SNR 멀티헤드]            AUROC={res['bin_auroc']:.4f}, "
        f"Macro-F1={res['macro_f1']:.4f}"
    )
    print()
    print(f"  체크포인트: {best_path}")
    print("=" * 70)

    # npz 저장
    np.savez(
        os.path.join(args.out_dir, "test_results.npz"),
        bin_labels=res["bin_labels"],
        bin_probs=res["bin_probs"],
        mc_labels=res["mc_labels"],
        mc_preds=res["mc_preds"],
        mc_probs=res["mc_probs"],
        bin_auroc=res["bin_auroc"],
        bin_f1=res["bin_f1"],
        bin_sens=res["bin_sens"],
        macro_f1=res["macro_f1"],
        weighted_f1=res["weighted_f1"],
        per_auroc=np.array(res["per_auroc"]),
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
        "--alpha",
        type=float,
        default=0.5,
        help="BCE 비중 (0=다중분류만, 1=이진만, 0.5=균등)",
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
    parser.add_argument(
        "--p_noise",
        type=float,
        default=0.75,
        help="샘플당 노이즈 주입 확률 (1-p_noise는 clean 유지)",
    )
    parser.add_argument(
        "--snr_set", type=str, default="24,18,12,6,0", help="쉼표 구분 SNR 집합 (dB)"
    )
    parser.add_argument(
        "--noise_mode",
        type=str,
        default="single",
        choices=["single", "mixed", "mixed_temporal"],
        help="single=리드당 1종 / mixed=3종 가중합성 / mixed_temporal=+시간 엔벨로프",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="out_dir/resume.pt 있으면 마지막 epoch부터 자동 재개 (self-healing)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
