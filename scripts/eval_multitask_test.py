"""
멀티태스크 체크포인트 → CPSC mc test 5-class per-class AUROC + Macro-F1 평가 (추론 전용).
=====================================================================================
용도: 학습이 막판에 중단돼도 best 체크포인트만으로 test 지표를 산출(재학습 불필요).
      train_lora_multitask.py의 evaluate()를 그대로 재사용 → 학습 시 평가와 동일 코드.
사용:
  python scripts/eval_multitask_test.py --ckpt outputs/lora_multitask_mixed/lora_multitask_snr_best.pt \
         --out_csv results/multitask_mixed_metrics.csv --tag mixed
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from train_lora_multitask import (
    CLASS_NAMES,
    N_CLASSES,
    BinaryHead,
    CPSCMultiTaskDataset,
    MulticlassHead,
    evaluate,
    inject_lora,
    load_ecgfm,
)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_dir", default="data/processed/cpsc2018_mc")
    ap.add_argument(
        "--ckpt_fm", default="checkpoints/ecg-fm/mimic_iv_ecg_physionet_pretrained.pt"
    )
    ap.add_argument("--out_csv", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bb = load_ecgfm(args.ckpt_fm, dev)
    for p in bb.parameters():
        p.requires_grad_(False)
    ck = torch.load(args.ckpt, map_location=dev)
    inject_lora(
        bb, rank=ck.get("lora_rank", 8), alpha=ck.get("lora_alpha", 16.0), dropout=0.0
    )
    bb.load_state_dict(ck["backbone_lora"], strict=False)
    hb = BinaryHead().to(dev)
    hb.load_state_dict(ck["head_bin_state"])
    hm = MulticlassHead().to(dev)
    hm.load_state_dict(ck["head_mc_state"])

    test = CPSCMultiTaskDataset(os.path.join(args.data_dir, "test"))
    loader = DataLoader(test, batch_size=args.batch, shuffle=False, num_workers=0)
    res = evaluate(bb, hb, hm, loader, dev)

    print("=" * 60)
    print(f"멀티태스크 test 평가  tag={args.tag or '-'}  ckpt={args.ckpt}")
    print(
        f"  noise_mode={ck.get('noise_mode')} alpha={ck.get('alpha')} "
        f"epoch={ck.get('epoch')} val_composite={ck.get('val_composite')}"
    )
    print("=" * 60)
    print(
        f"  [이진 응급]  AUROC={res['bin_auroc']:.4f}  F1@0.5={res['bin_f1']:.4f}  "
        f"Sens@95Sp={res['bin_sens']:.4f}"
    )
    print(
        f"  [다중 5-class]  Macro-F1={res['macro_f1']:.4f}  Weighted-F1={res['weighted_f1']:.4f}"
    )
    print("  Per-class AUROC:")
    for c in range(N_CLASSES):
        print(f"    [{c}] {CLASS_NAMES[c]:28s} {res['per_auroc'][c]:.4f}")

    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow(["metric", "value"])
            wr.writerow(["tag", args.tag])
            wr.writerow(["noise_mode", ck.get("noise_mode")])
            wr.writerow(["epoch", ck.get("epoch")])
            wr.writerow(["bin_auroc", round(res["bin_auroc"], 4)])
            wr.writerow(["bin_f1", round(res["bin_f1"], 4)])
            wr.writerow(["bin_sens95", round(res["bin_sens"], 4)])
            wr.writerow(["macro_f1", round(res["macro_f1"], 4)])
            wr.writerow(["weighted_f1", round(res["weighted_f1"], 4)])
            for c in range(N_CLASSES):
                wr.writerow([f"auroc_{CLASS_NAMES[c]}", round(res["per_auroc"][c], 4)])
        print(f"\n[저장] {args.out_csv}")


if __name__ == "__main__":
    main()
