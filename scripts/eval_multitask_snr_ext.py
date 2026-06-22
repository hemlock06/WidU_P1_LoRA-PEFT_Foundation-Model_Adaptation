"""
멀티태스크 체크포인트 2차 평가: SNR 저하 곡선(이진 응급) + 외부검증(CACHET/INCART).
A vs B(mixed vs mixed_temporal) 공정 비교 — 평가 노이즈는 '균일(mixed)'로 통일하고
A·B에 동일 시드(1000+i)로 같은 노이즈 realization 적용. head_bin(이진 응급)만 사용.
"""
from __future__ import annotations
import argparse, os, sys, csv
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from train_lora_multitask import load_ecgfm, inject_lora, BinaryHead
from multisnr import MultiSNRNoise

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
from sklearn.metrics import roc_auc_score, roc_curve

SNR_LEVELS = [None, 24, 18, 12, 6, 0, -6]
CKPT_FM = "checkpoints/ecg-fm/mimic_iv_ecg_physionet_pretrained.pt"


def load_mt(ckpt_path, dev):
    bb = load_ecgfm(CKPT_FM, dev)
    for p in bb.parameters():
        p.requires_grad_(False)
    ck = torch.load(ckpt_path, map_location=dev)
    inject_lora(bb, rank=ck.get("lora_rank", 8), alpha=ck.get("lora_alpha", 16.0), dropout=0.0)
    bb.load_state_dict(ck["backbone_lora"], strict=False)
    hb = BinaryHead().to(dev); hb.load_state_dict(ck["head_bin_state"])
    bb.eval(); hb.eval()
    return bb, hb


@torch.no_grad()
def bin_metrics(bb, hb, sigs, labs, dev, aug=None, snr=None, bs=64):
    probs = []
    for i in range(0, len(sigs), bs):
        x = torch.tensor(sigs[i:i + bs], dtype=torch.float32, device=dev)
        if aug is not None and snr is not None:
            x = aug.inject_fixed(x, snr)
        emb = bb(source=x, padding_mask=None, features_only=True)["x"].mean(1)
        probs.append(torch.sigmoid(hb(emb)).cpu().numpy())
    p = np.concatenate(probs); y = labs.astype(int)
    au = roc_auc_score(y, p)
    fpr, tpr, _ = roc_curve(y, p); spec = 1 - fpr
    idx = np.searchsorted(spec[::-1], 0.95)
    sens = float(tpr[::-1][idx]) if idx < len(tpr) else float("nan")
    return au, sens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--mc_dir", default="data/processed/cpsc2018_mc/test")
    ap.add_argument("--nstdb", default="data/raw/nstdb")
    ap.add_argument("--noise_mode", default="mixed", help="평가 노이즈(균일). A·B 통일")
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bb, hb = load_mt(args.ckpt, dev)
    rows = [("metric", "value"), ("tag", args.tag), ("eval_noise", args.noise_mode)]

    sigs = np.load(os.path.join(args.mc_dir, "signals.npy"))
    labs = np.load(os.path.join(args.mc_dir, "labels_bin.npy"))
    aug = MultiSNRNoise(nstdb_dir=args.nstdb, device=dev, seed=42, noise_mode=args.noise_mode)
    print(f"[{args.tag}] SNR 곡선 (이진 응급 AUROC, 평가노이즈={args.noise_mode})")
    for i, snr in enumerate(SNR_LEVELS):
        aug.rng = np.random.default_rng(1000 + i)   # A·B 동일 노이즈 realization
        au, sens = bin_metrics(bb, hb, sigs, labs, dev,
                               aug=(None if snr is None else aug), snr=snr)
        tag = "clean" if snr is None else f"{snr}dB"
        rows += [(f"snr_{tag}_auroc", round(au, 4)), (f"snr_{tag}_sens95", round(sens, 4))]
        print(f"   {tag:>6}: AUROC={au:.4f}  Sens95={sens:.4f}")

    for db, path in [("CACHET", "data/processed/cachet"), ("INCART", "data/processed/incart"),
                     ("STAFF", "data/processed/staffiii"), ("LTST", "data/processed/ltst")]:
        if not os.path.isdir(path):
            continue
        s = np.load(os.path.join(path, "signals.npy")); l = np.load(os.path.join(path, "labels.npy"))
        au, sens = bin_metrics(bb, hb, s, l, dev)
        rows += [(f"ext_{db}_auroc", round(au, 4)), (f"ext_{db}_sens95", round(sens, 4))]
        print(f"   {db:>6}: AUROC={au:.4f}  Sens95={sens:.4f}")

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    print(f"[저장] {args.out_csv}")


if __name__ == "__main__":
    main()
