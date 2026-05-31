"""
단계 6-③: ECG-FM LoRA + RLM + multi-SNR Fine-tuning
=====================================================
목적:
  대조군 ②(train_lora.py, clean)에 multi-SNR 모션 증강을 추가한 실험군 ③.
  ECG-FM이 보지 못한 모션 강건성을 학습 단계에 이식 (thesis 핵심 contribution).

ablation 대조 구조 (사용자 확정):
  ① 베이스라인 (선형 프로빙)        : train_baseline.py
  ② LoRA + RLM (clean)              : train_lora.py
  ③ LoRA + RLM + multi-SNR (본 파일): train_lora_multisnr.py

증강 순서 (스펙 8-5):
  clean → multi-SNR 노이즈 주입 → RLM 마스킹 → ECG-FM+LoRA+헤드

설정 (decisions.md 2026-05-25):
  SNR 이산 집합 {24,18,12,6,0}dB / p_noise=0.75(25% clean) / lead별 독립 SNR
  LoRA rank=8 alpha=16 dropout=0.1, q_proj·v_proj (②와 동일)

사용법:
  python scripts/train_lora_multisnr.py
  python scripts/train_lora_multisnr.py --p_noise 0.7 --snr_set 24,18,12,6,0
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Windows 콘솔(cp949) 인코딩 에러 방지
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# 공용 빌딩블록은 ②(train_lora.py)에서 그대로 재사용 — 중복·드리프트 방지
from train_lora import (
    CPSCDataset,
    LinearHead,
    count_trainable,
    evaluate,
    extract_embedding,
    inject_lora,
    load_ecgfm,
    random_lead_mask,
)
from multisnr import MultiSNRNoise


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print("단계 6-③: ECG-FM LoRA + RLM + multi-SNR Fine-tuning")
    print("=" * 65)
    print(f"디바이스:   {device}")
    print(f"데이터:     {args.data_dir}")
    print(f"체크포인트: {args.ckpt_path}")
    print(f"NSTDB:      {args.nstdb_dir}")
    print(f"출력:       {args.out_dir}")
    print(f"LoRA:       rank={args.lora_rank}, alpha={args.lora_alpha}, "
          f"dropout={args.lora_dropout}")
    print(f"RLM:        p={args.rlm_p}")
    snr_set = tuple(int(s) for s in args.snr_set.split(","))
    print(f"multi-SNR:  set={snr_set}dB, p_noise={args.p_noise} "
          f"({int((1-args.p_noise)*100)}% clean 유지)")
    print()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── 재현성: 시드 고정 ─────────────────────────────────────────────
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

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

    # ── multi-SNR 증강기 (노이즈 pool을 500Hz로 적재) ──────────────────
    print("[증강] NSTDB 노이즈 로드 + 500Hz 리샘플 중...")
    multisnr = MultiSNRNoise(nstdb_dir=args.nstdb_dir, snr_set=snr_set,
                             device=device, seed=args.seed)
    print(f"       pool 길이: "
          + ", ".join(f"{t}={multisnr.noise_pool[t].shape[0]:,}"
                      for t in ("bw", "em", "ma")))
    print()

    # ── ECG-FM 로드 + LoRA 주입 ───────────────────────────────────────
    print("[모델] ECG-FM 백본 로드 중...")
    backbone = load_ecgfm(args.ckpt_path, device)
    for p in backbone.parameters():
        p.requires_grad_(False)

    replaced = inject_lora(backbone, rank=args.lora_rank,
                           alpha=args.lora_alpha, dropout=args.lora_dropout)
    total, trainable = count_trainable(backbone)
    print(f"       LoRA 주입: {len(replaced)}개 레이어, "
          f"학습 {trainable:,}개 ({100*trainable/total:.2f}%)")

    head = LinearHead().to(device)
    head_params = sum(p.numel() for p in head.parameters())
    print(f"       총 학습 파라미터: {trainable + head_params:,}개")
    print()

    # ── 옵티마이저 ────────────────────────────────────────────────────
    trainable_params = ([p for p in backbone.parameters() if p.requires_grad]
                        + list(head.parameters()))
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.1)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_auroc, best_epoch = 0.0, 0
    best_path = os.path.join(args.out_dir, "lora_multisnr_best.pt")

    print(f"{'Epoch':>5} {'TrainLoss':>10} {'ValAUROC':>9} "
          f"{'ValF1':>7} {'Sens@95Sp':>10} {'LR':>8}")
    print("-" * 55)

    for epoch in range(1, args.epochs + 1):
        backbone.train()
        head.train()
        total_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            # ① multi-SNR 노이즈 주입 (per-sample 게이트 + per-lead 독립 SNR)
            x = multisnr.inject(x, p_noise=args.p_noise)
            # ② RLM 마스킹 (노이즈 다음 — 스펙 8-5 순서)
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
            best_auroc, best_epoch = val_auroc, epoch
            torch.save({
                "epoch": epoch,
                "backbone_lora": {k: v for k, v in backbone.state_dict().items()
                                  if "lora_" in k},
                "head_state": head.state_dict(),
                "val_auroc": val_auroc, "val_f1": val_f1,
                "lora_rank": args.lora_rank, "lora_alpha": args.lora_alpha,
                "snr_set": snr_set, "p_noise": args.p_noise,
            }, best_path)

    # ── 테스트 평가 (clean test set — ②와 동일 조건 비교) ──────────────
    print()
    print(f"최고 val AUROC: {best_auroc:.4f} (epoch {best_epoch})")
    ckpt = torch.load(best_path, map_location=device)
    backbone.load_state_dict(ckpt["backbone_lora"], strict=False)
    head.load_state_dict(ckpt["head_state"])
    test_auroc, test_f1, test_sens = evaluate(backbone, head, test_loader, device)

    print()
    print("=" * 65)
    print("단계 6-③ 결과 (clean 테스트 세트)")
    print("=" * 65)
    print(f"  AUROC              : {test_auroc:.4f}  "
          f"(① 0.9435 / ② 0.9473)")
    print(f"  F1 (@threshold 0.5): {test_f1:.4f}  "
          f"(① 0.9177 / ② 0.9284)")
    print(f"  Sensitivity@95%Sp  : {test_sens:.4f}  (① 0.7564)")
    print()
    print(f"  모델 저장: {best_path}")
    print()
    print("주의: clean test에서 ②와 비슷하거나 약간 낮을 수 있음 (정상).")
    print("      multi-SNR의 진짜 가치는 단계 8 'SNR 저하 곡선'에서 드러남")
    print("      → 노이즈 환경에서 ②보다 강건해야 contribution 성립.")
    print("=" * 65)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_dir",
        default="data/processed/cpsc2018")
    p.add_argument("--ckpt_path",
        default="checkpoints/ecg-fm/mimic_iv_ecg_physionet_pretrained.pt")
    p.add_argument("--nstdb_dir",
        default="data/raw/nstdb")
    p.add_argument("--out_dir",
        default="outputs/lora_multisnr")
    p.add_argument("--batch_size",   type=int,   default=16)
    p.add_argument("--epochs",       type=int,   default=30)
    p.add_argument("--lr",           type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--lora_rank",    type=int,   default=8)
    p.add_argument("--lora_alpha",   type=float, default=16.0)
    p.add_argument("--lora_dropout", type=float, default=0.1)
    p.add_argument("--rlm_p",        type=float, default=0.5)
    p.add_argument("--snr_set",      type=str,   default="24,18,12,6,0",
                   help="쉼표 구분 SNR 집합 (dB)")
    p.add_argument("--p_noise",      type=float, default=0.75,
                   help="샘플당 노이즈 주입 확률 (1-p_noise는 clean 유지)")
    p.add_argument("--seed",         type=int,   default=42)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
