"""
N-lead 강건성 전구간 평가 (ablation_nlead_curve.py)
====================================================
목적:
  1~12-lead 전 구간에서 AUROC를 측정하여 "lead 수 vs 성능" 곡선 생성.
  기존 ablation_a1_rlm_leads.py는 4개 고정 포인트(12/4/2/1)만 평가 →
  전구간 곡선으로 강건성 경향을 정량화한다.

방법:
  각 N (1~12)에 대해 12개 lead 중 N개를 무작위 선택(seed 고정, M회 반복)하고
  CPSC 2018 test에서 AUROC를 측정. mean ± std로 보고.

평가 모델:
  ③  LoRA+RLM+multi-SNR (단일 이진, 강건성 서사 기준)
  P1  단일 백본 멀티태스크 α=0.7 (BinaryHead 사용)

출력:
  results/nlead_curve.csv    — N별 mean/std/min/max AUROC
  results/nlead_curve.png    — 곡선 그래프 (강건성 곡선)

사용법:
  python scripts/ablation_nlead_curve.py
  python scripts/ablation_nlead_curve.py --n_trials 30 --models ③
"""

import argparse
import itertools
import math
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    sys.exit("[오류] scikit-learn 미설치")

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── 경로 ─────────────────────────────────────────────────────────────────
CKPT_FM     = "checkpoints/ecg-fm/mimic_iv_ecg_physionet_pretrained.pt"
CPSC_TEST   = "data/processed/cpsc2018/test"
CPSC_MC_TEST= "data/processed/cpsc2018_mc/test"

CKPT_III    = "outputs/lora_multisnr/lora_multisnr_best.pt"
CKPT_P1     = "outputs/lora_multitask_snr_a07/lora_multitask_snr_best.pt"

OUT_DIR     = "results"

LEAD_NAMES  = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]


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
                targets=("self_attn.q_proj", "self_attn.v_proj")):
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear): continue
        if not any(name.endswith(t) for t in targets): continue
        parts = name.split(".")
        parent = model
        for p in parts[:-1]: parent = getattr(parent, p)
        setattr(parent, parts[-1], LoRALinear(module, rank, alpha, dropout))


# ── 헤드 ─────────────────────────────────────────────────────────────────
class BinaryHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(768, 1)
    def forward(self, x): return self.fc(x).squeeze(-1)


class MCHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(768, 5)
    def forward(self, x): return self.fc(x)


# ── 데이터셋 ─────────────────────────────────────────────────────────────
class ECGDataset(Dataset):
    def __init__(self, data_dir, binary_label_file="labels.npy"):
        self.signals = np.load(os.path.join(data_dir, "signals.npy"))
        self.labels  = np.load(os.path.join(data_dir, binary_label_file))
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return (torch.tensor(self.signals[idx], dtype=torch.float32),
                torch.tensor(float(self.labels[idx]), dtype=torch.float32))


# ── ECG-FM 로드 ───────────────────────────────────────────────────────────
def load_ecgfm(device):
    from fairseq_signals.utils.checkpoint_utils import load_model_and_task
    result = load_model_and_task(CKPT_FM)
    if isinstance(result, tuple):
        for r in result:
            if hasattr(r, "parameters"): return r.to(device)
            if isinstance(r, list) and r and hasattr(r[0], "parameters"): return r[0].to(device)
    return result.to(device)


# ── lead 마스킹 ───────────────────────────────────────────────────────────
def apply_lead_mask(x: torch.Tensor, available: list) -> torch.Tensor:
    """x: (B,12,T). available에 없는 lead를 0-fill."""
    masked = x.clone()
    for i in range(12):
        if i not in available:
            masked[:, i, :] = 0.0
    return masked


# ── 단일 lead 구성 평가 ───────────────────────────────────────────────────
@torch.no_grad()
def eval_leads(backbone, head, dataset, available_leads, device, batch_size=32,
               is_multitask=False):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_probs, all_labels = [], []
    backbone.eval()
    head.eval()
    for x, y in loader:
        x = apply_lead_mask(x.to(device), available_leads)
        emb = backbone(source=x, padding_mask=None, features_only=True)["x"].mean(dim=1)
        logit = head(emb)
        if is_multitask:
            logit = logit.squeeze(-1)  # BinaryHead already squeezes
        prob = torch.sigmoid(logit).cpu().numpy()
        all_probs.extend(prob.tolist())
        all_labels.extend(y.numpy().tolist())
        del x, emb, logit
    probs  = np.array(all_probs)
    labels = np.array(all_labels, dtype=int)
    if device.type == "cuda":
        torch.cuda.empty_cache()   # 200+회 반복 시 단편화 누적 방지 (장기 실행 안정화)
    return roc_auc_score(labels, probs)


# ── N-lead 전구간 평가 ────────────────────────────────────────────────────
def evaluate_nlead_curve(backbone, head, dataset, n_trials, seed, device,
                         is_multitask=False, n_min=1, n_max=12):
    """
    N=n_min~n_max 각각에 대해 n_trials회 무작위 lead 조합 평가.
    반환: {N: {"mean":, "std":, "min":, "max":, "all":[]}}
    """
    rng = np.random.default_rng(seed)
    results = {}

    for n in range(n_min, n_max + 1):
        aurocs = []
        if n == 12:
            # 12-lead는 모든 lead 사용 — 1회만
            aurocs.append(eval_leads(backbone, head, dataset,
                                     list(range(12)), device, is_multitask=is_multitask))
        else:
            # n_trials회 무작위 조합
            tried = set()
            attempts = 0
            while len(aurocs) < n_trials and attempts < n_trials * 10:
                combo = tuple(sorted(rng.choice(12, n, replace=False).tolist()))
                if combo in tried:
                    attempts += 1
                    continue
                tried.add(combo)
                a = eval_leads(backbone, head, dataset, list(combo), device,
                               is_multitask=is_multitask)
                aurocs.append(a)
                attempts += 1

        results[n] = {
            "mean": float(np.mean(aurocs)),
            "std":  float(np.std(aurocs)),
            "min":  float(np.min(aurocs)),
            "max":  float(np.max(aurocs)),
            "all":  aurocs,
        }
        status = f"  N={n:2d}: mean={results[n]['mean']:.4f} ± {results[n]['std']:.4f}"
        status += f"  (min={results[n]['min']:.4f}, max={results[n]['max']:.4f})"
        status += f"  [{len(aurocs)}회]"
        print(status, flush=True)

    return results


# ── 차트 생성 ────────────────────────────────────────────────────────────
def make_chart(results_dict, out_path):
    """results_dict: {"모델명": {N: {...}}}"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [경고] matplotlib 없음 — 차트 생략")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {"③ LoRA+RLM+multi-SNR": "#E07B39", "P1 (단일백본 α=0.7)": "#2E6DB4"}

    for model_name, res in results_dict.items():
        ns    = sorted(res.keys())
        means = [res[n]["mean"] for n in ns]
        stds  = [res[n]["std"]  for n in ns]
        color = colors.get(model_name, "green")
        ax.plot(ns, means, "o-", label=model_name, color=color, linewidth=2, markersize=5)
        ax.fill_between(ns,
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        alpha=0.15, color=color)

    ax.set_xlabel("사용 lead 수 (N)", fontsize=12)
    ax.set_ylabel("AUROC", fontsize=12)
    ax.set_title("N-lead 강건성: lead 수별 AUROC (CPSC 2018 test)", fontsize=13)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels([str(i) for i in range(1, 13)])
    ax.set_ylim(0.85, 0.98)
    ax.axhline(0.90, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"  차트 저장: {out_path}")
    plt.close()


# ── CSV 저장 ─────────────────────────────────────────────────────────────
def save_csv(results_dict, out_path):
    rows = ["model,n_leads,mean_auroc,std_auroc,min_auroc,max_auroc"]
    for model_name, res in results_dict.items():
        for n in sorted(res.keys()):
            r = res[n]
            rows.append(f"{model_name},{n},{r['mean']:.4f},{r['std']:.4f},"
                        f"{r['min']:.4f},{r['max']:.4f}")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(rows))
    print(f"  CSV 저장: {out_path}")


# ── 메인 ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n_trials", type=int, default=20,
                        help="N별 무작위 lead 조합 반복 횟수 (기본 20)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models", default="both",
                        choices=["③", "P1", "both"],
                        help="평가할 모델 (기본 both)")
    parser.add_argument("--ckpt3", default=CKPT_III,
                        help="③ 모델 체크포인트 (mixed 비교 시 override)")
    parser.add_argument("--name3", default="③ LoRA+RLM+multi-SNR",
                        help="③ 모델 라벨 (CSV/차트 키)")
    parser.add_argument("--out_csv", default=os.path.join(OUT_DIR, "nlead_curve.csv"))
    parser.add_argument("--out_png", default=os.path.join(OUT_DIR, "nlead_curve.png"))
    parser.add_argument("--n_min", type=int, default=1, help="평가 시작 N (청크 실행용)")
    parser.add_argument("--n_max", type=int, default=12, help="평가 끝 N (청크 실행용)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 65)
    print("N-lead 강건성 전구간 평가 (1~12-lead)")
    print("=" * 65)
    print(f"디바이스: {device}  |  n_trials={args.n_trials}  |  seed={args.seed}")
    print(f"평가 모델: {args.models}")
    print()

    # ── ECG-FM 로드 (공유) ────────────────────────────────────────────────
    print("[백본 로드] ECG-FM ...")
    backbone = load_ecgfm(device)
    inject_lora(backbone, rank=8, alpha=16.0, dropout=0.1)
    backbone.eval()

    all_results = {}

    # ── ③ 모델 ────────────────────────────────────────────────────────────
    if args.models in ("③", "both"):
        print(f"\n[① {args.name3} 평가]  ckpt={args.ckpt3}")
        ckpt = torch.load(args.ckpt3, map_location=device)
        backbone.load_state_dict(ckpt["backbone_lora"], strict=False)
        head = BinaryHead().to(device)
        head.load_state_dict(ckpt["head_state"])
        dataset = ECGDataset(CPSC_TEST)
        print(f"  데이터: CPSC test {len(dataset)}개")
        res = evaluate_nlead_curve(backbone, head, dataset,
                                   args.n_trials, args.seed, device, is_multitask=False,
                                   n_min=args.n_min, n_max=args.n_max)
        all_results[args.name3] = res

    # ── P1 모델 ──────────────────────────────────────────────────────────
    if args.models in ("P1", "both"):
        print("\n[② P1 단일백본 α=0.7 평가]")
        ckpt = torch.load(CKPT_P1, map_location=device)
        backbone.load_state_dict(ckpt["backbone_lora"], strict=False)
        # BinaryHead만 사용 (head_bin_state 키)
        head = BinaryHead().to(device)
        head.load_state_dict(ckpt["head_bin_state"])
        # mc 데이터셋은 labels_bin.npy(이진) 사용
        dataset_mc = ECGDataset(CPSC_MC_TEST, binary_label_file="labels_bin.npy")
        print(f"  데이터: CPSC mc test {len(dataset_mc)}개 (labels_bin)")
        res = evaluate_nlead_curve(backbone, head, dataset_mc,
                                   args.n_trials, args.seed, device, is_multitask=False)
        all_results["P1 (단일백본 α=0.7)"] = res

    # ── 결과 저장 ─────────────────────────────────────────────────────────
    print()
    csv_path = args.out_csv
    png_path = args.out_png
    save_csv(all_results, csv_path)
    make_chart(all_results, png_path)

    # ── 요약 출력 ─────────────────────────────────────────────────────────
    print()
    print("=" * 65)
    print("요약")
    print("=" * 65)
    for model_name, res in all_results.items():
        print(f"\n{model_name}")
        print(f"  {'N':>4}  {'mean':>7}  {'±std':>7}  {'min':>7}  {'max':>7}")
        print(f"  {'-'*42}")
        for n in sorted(res.keys()):
            r = res[n]
            print(f"  {n:4d}  {r['mean']:7.4f}  {r['std']:7.4f}  "
                  f"{r['min']:7.4f}  {r['max']:7.4f}")
    print()
    print(f"결과 파일: {csv_path}")
    print(f"차트:      {png_path}")


if __name__ == "__main__":
    main()
