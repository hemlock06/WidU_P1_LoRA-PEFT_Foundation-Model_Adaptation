"""
단계 8-(2): SNR 저하 곡선 — 모션 강건성 측정
=============================================
목적:
  test set에 고정 SNR 노이즈를 주입하며 정확도가 얼마나 버티는지 측정.
  ② LoRA+RLM(clean) vs ③ LoRA+RLM+multi-SNR 을 동일 노이즈 조건에서 비교.
  → "multi-SNR이 모션 환경에서 더 강건"이 thesis 핵심 contribution의 정량 증거.

핵심 논리 (강건성 검증):
  - 두 모델 모두 같은 noisy test 입력으로 평가 (동일 시드 → 동일 노이즈 realization).
  - SNR이 낮아질수록(노이즈↑) 두 곡선이 벌어져야 함:
    ②는 clean만 학습 → 급락, ③는 multi-SNR 학습 → 완만.
  - -6dB는 학습 분포(최저 0dB) 밖 → 외삽 강건성 추가 확인.

평가 SNR: [clean, 24, 18, 12, 6, 0, -6] dB
  주의: 노이즈는 전 lead에 주입(inject_fixed), RLM 마스킹은 적용 안 함
        (순수 모션 강건성만 분리 측정).

사용법:
  python scripts/eval_snr_curve.py
"""

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from train_lora import (
    CPSCDataset,
    LinearHead,
    inject_lora,
    load_ecgfm,
)
from multisnr import MultiSNRNoise

try:
    from sklearn.metrics import roc_auc_score, f1_score, roc_curve
except ImportError:
    sys.exit("[오류] scikit-learn 미설치")

SNR_LEVELS = [None, 24, 18, 12, 6, 0, -6]   # None = clean


def load_lora_model(ckpt_path, ecgfm_ckpt, device):
    """저장된 LoRA 체크포인트(②/③)를 ECG-FM에 복원."""
    ckpt = torch.load(ckpt_path, map_location=device)
    backbone = load_ecgfm(ecgfm_ckpt, device)
    for p in backbone.parameters():
        p.requires_grad_(False)
    inject_lora(backbone, rank=ckpt["lora_rank"], alpha=ckpt["lora_alpha"], dropout=0.0)
    backbone.load_state_dict(ckpt["backbone_lora"], strict=False)
    head = LinearHead().to(device)
    head.load_state_dict(ckpt["head_state"])
    backbone.eval()
    head.eval()
    return backbone, head


@torch.no_grad()
def eval_at_snr(backbone, head, loader, device, multisnr, snr_db):
    """snr_db 노이즈 주입 후 test 평가. snr_db=None이면 clean."""
    all_logits, all_labels = [], []
    for x, y in loader:
        x = x.to(device)
        if snr_db is not None:
            x = multisnr.inject_fixed(x, snr_db)
        emb = backbone(source=x, padding_mask=None, features_only=True)["x"].mean(dim=1)
        all_logits.append(head(emb).cpu())
        all_labels.append(y)
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy().astype(int)
    probs = 1 / (1 + np.exp(-logits))
    auroc = roc_auc_score(labels, probs)
    f1 = f1_score(labels, (probs >= 0.5).astype(int), zero_division=0)
    fpr, tpr, _ = roc_curve(labels, probs)
    spec = 1 - fpr
    idx = np.searchsorted(spec[::-1], 0.95)
    sens = float(tpr[::-1][idx]) if idx < len(tpr) else float("nan")
    return auroc, f1, sens


def run_model(name, ckpt_path, args, device, test_loader, multisnr):
    print(f"\n[{name}] {ckpt_path}")
    if not os.path.exists(ckpt_path):
        print(f"  [주의] 체크포인트 없음 — 스킵")
        return None
    backbone, head = load_lora_model(ckpt_path, args.ckpt_path, device)
    aurocs = []
    for i, snr in enumerate(SNR_LEVELS):
        # 동일 시드 → ②와 ③가 같은 노이즈 realization을 봄 (공정 비교)
        multisnr.rng = np.random.default_rng(1000 + i)
        auroc, f1, sens = eval_at_snr(backbone, head, test_loader, device, multisnr, snr)
        aurocs.append(auroc)
        tag = "clean" if snr is None else f"{snr:>3}dB"
        print(f"    SNR {tag}: AUROC={auroc:.4f}  F1={f1:.4f}  Sens@95Sp={sens:.4f}")
    del backbone, head
    torch.cuda.empty_cache()
    return aurocs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_dir",
        default="data/processed/cpsc2018")
    p.add_argument("--ckpt_path",
        default="checkpoints/ecg-fm/mimic_iv_ecg_physionet_pretrained.pt")
    p.add_argument("--nstdb_dir",
        default="data/raw/nstdb")
    p.add_argument("--lora2_ckpt",
        default="outputs/lora/lora_best.pt")
    p.add_argument("--lora3_ckpt",
        default="outputs/lora_multisnr/lora_multisnr_best.pt")
    p.add_argument("--out_dir",
        default="outputs/snr_curve")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--noise_mode", type=str, default="single",
        choices=["single", "mixed"],
        help="평가 노이즈 합성 모드 (single=리드당 1종 / mixed=3종 가중합성)")
    p.add_argument("--label2", type=str, default="lora2_clean",
        help="첫째 모델(--lora2_ckpt) CSV 행 라벨")
    p.add_argument("--label3", type=str, default="lora3_multisnr",
        help="둘째 모델(--lora3_ckpt) CSV 행 라벨")
    p.add_argument("--name2", type=str, default="② LoRA+RLM (clean)")
    p.add_argument("--name3", type=str, default="③ LoRA+RLM+multi-SNR")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    print("=" * 65)
    print("단계 8-(2): SNR 저하 곡선 — ② clean vs ③ multi-SNR")
    print("=" * 65)
    print(f"디바이스: {device}")
    print(f"평가 SNR: {['clean' if s is None else f'{s}dB' for s in SNR_LEVELS]}")

    test_ds = CPSCDataset(os.path.join(args.data_dir, "test"))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    print(f"test set: {len(test_ds)}개")

    multisnr = MultiSNRNoise(nstdb_dir=args.nstdb_dir, device=device, seed=42,
                             noise_mode=args.noise_mode)
    print(f"평가 노이즈 모드: {args.noise_mode}")

    res2 = run_model(args.name2, args.lora2_ckpt, args, device, test_loader, multisnr)
    res3 = run_model(args.name3, args.lora3_ckpt, args, device, test_loader, multisnr)

    # ── CSV 저장 ──────────────────────────────────────────────────────
    csv_path = os.path.join(args.out_dir, "snr_curve.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("SNR_dB," + ",".join(["clean" if s is None else str(s) for s in SNR_LEVELS]) + "\n")
        if res2: f.write(f"{args.label2}," + ",".join(f"{a:.4f}" for a in res2) + "\n")
        if res3: f.write(f"{args.label3}," + ",".join(f"{a:.4f}" for a in res3) + "\n")
    print(f"\n[저장] CSV: {csv_path}")

    # ── PNG 그래프 ────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xticks = ["clean" if s is None else f"{s}" for s in SNR_LEVELS]
        xpos = list(range(len(SNR_LEVELS)))
        plt.figure(figsize=(8, 5))
        if res2: plt.plot(xpos, res2, "o-", label="② LoRA+RLM (clean)")
        if res3: plt.plot(xpos, res3, "s-", label="③ LoRA+RLM+multi-SNR")
        plt.xticks(xpos, xticks)
        plt.xlabel("SNR (dB)  ←  noise increases")
        plt.ylabel("AUROC")
        plt.title("SNR degradation curve: motion robustness")
        plt.grid(True, alpha=0.3)
        plt.legend()
        png_path = os.path.join(args.out_dir, "snr_curve.png")
        plt.savefig(png_path, dpi=150, bbox_inches="tight")
        print(f"[저장] PNG: {png_path}")
    except Exception as e:
        print(f"[경고] PNG 생성 실패: {e}")

    # ── 해석 ──────────────────────────────────────────────────────────
    if res2 and res3:
        print("\n[해석] 강건성 핵심 포인트")
        for i, s in enumerate(SNR_LEVELS):
            tag = "clean" if s is None else f"{s}dB"
            gap = res3[i] - res2[i]
            print(f"  {tag:>6}: ③-② = {gap:+.4f}")
        print("  → 노이즈가 강할수록(SNR↓) ③의 우위(+)가 커지면 multi-SNR 효과 입증")
        print("  → clean에서 ③≈② 또는 약간 낮음은 정상 (강건성-clean 트레이드오프)")


if __name__ == "__main__":
    main()
