"""
단계 9: 외부 데이터셋 추론·평가 (eval_external.py)
=====================================================
목적:
  CPSC 2018으로 학습된 모델(①②③)을 외부 DB에 적용해
  일반화 성능(도메인 외 강건성)을 측정한다.

대상 모델:
  ① 베이스라인  (ECG-FM frozen + Linear)
  ② LoRA + RLM  (LoRA fine-tuned)
  ③ LoRA + RLM + multi-SNR  (노이즈 강건)

대상 DB:
  - CACHET-CADB  : 1,362개  (AF=747, NSR=615)
  - INCART        : 7,811개  (응급=540, 정상=7271)
  - STAFF-III     : 6,650개  (응급=3550, 정상=3100)
  (LTST: 다운로드 완료 후 별도 실행)

평가 지표:
  AUROC, F1@0.5, Sensitivity@95%Specificity

사용법:
  python scripts/eval_external.py
  python scripts/eval_external.py --dbs cachet incart
  python scripts/eval_external.py --models baseline lora
"""

import argparse
import csv
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:
    from sklearn.metrics import roc_auc_score, f1_score, roc_curve
except ImportError:
    sys.exit("[오류] scikit-learn 미설치 — pip install scikit-learn")

# ── 경로 기본값 ────────────────────────────────────────────────────────
CKPT_ECG_FM  = "checkpoints/ecg-fm/mimic_iv_ecg_physionet_pretrained.pt"
PROCESSED    = "data/processed"
OUTPUTS      = "outputs"

MODEL_PATHS  = {
    "baseline"    : f"{OUTPUTS}/baseline/baseline_best.pt",
    "lora"        : f"{OUTPUTS}/lora/lora_best.pt",
    "lora_multisnr": f"{OUTPUTS}/lora_multisnr/lora_multisnr_best.pt",
}

DB_DIRS = {
    "cachet" : f"{PROCESSED}/cachet",
    "incart"  : f"{PROCESSED}/incart",
    "staffiii": f"{PROCESSED}/staffiii",
    "ltst"    : f"{PROCESSED}/ltst",   # 다운로드 완료 후 추가
}

DB_LABELS = {
    "cachet" : "CACHET-CADB",
    "incart"  : "INCART",
    "staffiii": "STAFF-III",
    "ltst"    : "LTST",
}

EMBED_DIM = 768
BATCH_SIZE = 64


# ── LoRA 모듈 (train_lora.py와 동일) ──────────────────────────────────

class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.original = linear
        self.original.weight.requires_grad_(False)
        if self.original.bias is not None:
            self.original.bias.requires_grad_(False)
        in_dim  = linear.in_features
        out_dim = linear.out_features
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
        if not isinstance(module, nn.Linear):
            continue
        if not any(name.endswith(s) for s in target_suffixes):
            continue
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], LoRALinear(module, rank, alpha, dropout))


# ── 데이터셋 ──────────────────────────────────────────────────────────

class ExternalDataset(Dataset):
    def __init__(self, data_dir: str):
        self.signals = np.load(os.path.join(data_dir, "signals.npy"))  # (N,12,5000)
        self.labels  = np.load(os.path.join(data_dir, "labels.npy"))   # (N,) int

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor(self.signals[idx], dtype=torch.float32)
        y = torch.tensor(float(self.labels[idx]),  dtype=torch.float32)
        return x, y


# ── ECG-FM 로드 ────────────────────────────────────────────────────────

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


# ── 임베딩 추출 ────────────────────────────────────────────────────────

@torch.no_grad()
def extract_embedding(backbone, x: torch.Tensor) -> torch.Tensor:
    out = backbone(source=x, padding_mask=None, features_only=True)
    return out["x"].mean(dim=1)  # (B, 768)


# ── 분류 헤드 ─────────────────────────────────────────────────────────

class LinearHead(nn.Module):
    def __init__(self, in_dim: int = EMBED_DIM):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, x):
        return self.fc(x).squeeze(-1)


# ── 평가 ──────────────────────────────────────────────────────────────

def evaluate(backbone, head, loader, device):
    head.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x   = x.to(device)
            emb = extract_embedding(backbone, x)
            logits = head(emb).cpu()
            all_logits.append(logits)
            all_labels.append(y)

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy().astype(int)
    probs  = 1 / (1 + np.exp(-logits))

    auroc = roc_auc_score(labels, probs)
    f1    = f1_score(labels, (probs >= 0.5).astype(int), zero_division=0)

    fpr, tpr, _ = roc_curve(labels, probs)
    spec = 1 - fpr
    idx  = np.searchsorted(spec[::-1], 0.95)
    sens_at_95sp = float(tpr[::-1][idx]) if idx < len(tpr) else float("nan")

    return auroc, f1, sens_at_95sp


# ── 모델 로드 헬퍼 ────────────────────────────────────────────────────

def load_model(model_name: str, backbone, device: torch.device):
    """
    backbone을 in-place 수정해 LoRA를 주입하고,
    저장된 가중치를 로드해 (backbone, head) 반환.
    backbone은 매 모델마다 새로 로드해야 함.
    """
    ckpt_path = MODEL_PATHS[model_name]
    ckpt = torch.load(ckpt_path, map_location=device)

    head = LinearHead().to(device)
    head.load_state_dict(ckpt["head_state"])
    head.eval()

    if model_name == "baseline":
        # 백본 완전 동결
        for p in backbone.parameters():
            p.requires_grad_(False)
        backbone.eval()
    else:
        # LoRA 주입 후 가중치 복원
        rank  = ckpt.get("lora_rank", 8)
        alpha = ckpt.get("lora_alpha", 16.0)
        inject_lora(backbone, rank=rank, alpha=alpha, dropout=0.0)
        backbone.load_state_dict(ckpt["backbone_lora"], strict=False)
        backbone.eval()

    return backbone, head


# ── 메인 ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ecgfm_ckpt", default=CKPT_ECG_FM)
    parser.add_argument("--dbs",    nargs="+",
                        default=["cachet", "incart", "staffiii"],
                        choices=list(DB_DIRS.keys()))
    parser.add_argument("--models", nargs="+",
                        default=["baseline", "lora", "lora_multisnr"],
                        choices=list(MODEL_PATHS.keys()))
    parser.add_argument("--batch",  type=int, default=BATCH_SIZE)
    parser.add_argument("--out_csv", default=f"{OUTPUTS}/external_eval_results.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("단계 9: 외부 데이터셋 평가")
    print("=" * 70)
    print(f"디바이스  : {device}")
    print(f"평가 모델 : {args.models}")
    print(f"평가 DB   : {args.dbs}")
    print()

    # DB별 DataLoader 준비
    loaders = {}
    for db in args.dbs:
        db_dir = DB_DIRS[db]
        if not os.path.isdir(db_dir):
            print(f"[건너뜀] {db}: 폴더 없음 ({db_dir})")
            continue
        ds = ExternalDataset(db_dir)
        n_pos = int((ds.labels == 1).sum())
        n_neg = int((ds.labels == 0).sum())
        print(f"  {DB_LABELS[db]:15s}: {len(ds):5d}개  응급={n_pos}  정상={n_neg}")
        loaders[db] = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=0)
    print()

    MODEL_DISPLAY = {
        "baseline"     : "① 베이스라인",
        "lora"         : "② LoRA+RLM",
        "lora_multisnr": "③ LoRA+RLM+multi-SNR",
    }

    results = []   # (model, db, auroc, f1, sens)

    for model_name in args.models:
        if not os.path.isfile(MODEL_PATHS[model_name]):
            print(f"[건너뜀] {model_name}: 체크포인트 없음")
            continue

        print(f"\n{'─'*70}")
        print(f"  모델: {MODEL_DISPLAY[model_name]}")
        print(f"{'─'*70}")

        # 모델마다 ECG-FM 새로 로드 (LoRA 주입이 in-place라 공유 불가)
        print("  ECG-FM 로드 중...", end=" ", flush=True)
        backbone = load_ecgfm(args.ecgfm_ckpt, device)
        print("완료")

        backbone, head = load_model(model_name, backbone, device)

        print(f"  {'DB':15s} {'AUROC':>7} {'F1@0.5':>8} {'Sens@95Sp':>10}")
        print(f"  {'-'*15} {'-'*7} {'-'*8} {'-'*10}")

        for db, loader in loaders.items():
            auroc, f1, sens = evaluate(backbone, head, loader, device)
            print(f"  {DB_LABELS[db]:15s} {auroc:7.4f} {f1:8.4f} {sens:10.4f}")
            results.append({
                "model": MODEL_DISPLAY[model_name],
                "db"   : DB_LABELS[db],
                "auroc": round(auroc, 4),
                "f1"   : round(f1,    4),
                "sens_at_95sp": round(sens, 4),
            })

        del backbone, head
        torch.cuda.empty_cache()

    # CSV 저장
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model","db","auroc","f1","sens_at_95sp"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n결과 저장: {args.out_csv}")
    print("=" * 70)


if __name__ == "__main__":
    main()
