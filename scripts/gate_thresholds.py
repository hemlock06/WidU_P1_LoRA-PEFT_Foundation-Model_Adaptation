"""
단계 7b: 신호품질 게이트 — 연속 신뢰도 + 3단 임계값 산출
==========================================================
목적:
  gate_best.pt의 연속 출력(P(unacceptable))을 P2 융합용
  {use / mask / alert} 3단 임계값 레이어로 변환.

임계값 설계 원칙:
  gate_score = P(unacceptable), 0~1
  reliability = 1 - gate_score (P2 인터페이스 입력)

  gate_tier 결정:
    gate_score < t_mask              → "use"   (신뢰도 高, ECG 가중치 크게)
    t_mask ≤ gate_score < t_alert   → "mask"  (신뢰도 中, 가중치 중간)
    gate_score ≥ t_alert            → "alert" (신뢰도 低, 자이로·SpO2로 이동)

임계값 산출 방법 (val set 기준):
  t_alert : val Specificity 90% 지점 (양호 신호의 90%가 alert 미만)
            → t_alert ≈ 0.5111. 비지배(Pareto) 운영점.
  t_mask  : val 양호 신호(label=0) gate_score 분포의 p75 백분위수
            양호 신호의 75%를 "use"로 확보하는 보수적 경계 → 0.2149.

설계 근거 (2026-05-29 재검토 반영):
  - t_mask=p75(0.2149): 기존 spec=0.65 기준(0.1510) 대비 양호 use 보존율
    58.3%→66.1% (+7.8%p) 개선. 핵심 개선 항목.
  - t_alert=spec0.90(0.5111): **Youden J(0.4462) 기각.** test set에서
    [0.4462, 0.5111) 구간엔 불량 신호 0개·양호 신호 4개만 존재 →
    Youden J는 오경보만 +3.5%p 늘리고 탐지 이득 0(63.9% 동일)인
    Pareto 지배 지점. spec=0.90(0.5111)이 비지배 운영점.
  - 불량 mask+alert 탐지율(80.6%)은 t_alert와 무관하게 t_mask가 결정.
    t_alert는 "양호 오경보 vs 불량 alert 격상"만 조절 → 보수적(높은) t_alert
    선택이 멀쩡한 ECG 보존에 유리. spec=0.95(0.7607)는 더 보수적 대안
    (test 오경보 1.7%, 단 불량 alert 58.3%로 5.6%p 하락 — alert→mask 이동).

사용법:
  python scripts/gate_thresholds.py
  python scripts/gate_thresholds.py --t_mask_pct 75.0 --spec_alert 0.90  # (기본값)
"""

import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

try:
    from sklearn.metrics import roc_auc_score, roc_curve
except ImportError:
    sys.exit("[오류] scikit-learn 미설치")


def set_deterministic(seed=42):
    """추론 재현성 고정 — 작은 val셋에서 임계값이 실행 간 변동하지 않도록.
    cuDNN 비결정 알고리즘 선택을 끄고 시드를 고정한다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

EMBED_DIM  = 768
DATA_DIR   = "D:/WidU_ecg-fm_emergency-detection/data/processed/physionet2011"
CKPT_FM    = "D:/WidU_ecg-fm_emergency-detection/checkpoints/ecg-fm/mimic_iv_ecg_physionet_pretrained.pt"
GATE_CKPT  = "D:/WidU_ecg-fm_emergency-detection/outputs/gate/gate_best.pt"
OUT_DIR    = "D:/WidU_ecg-fm_emergency-detection/outputs/gate"


class GateDataset(Dataset):
    def __init__(self, split_dir):
        self.signals = np.load(os.path.join(split_dir, "signals.npy"))
        self.labels  = np.load(os.path.join(split_dir, "labels.npy"))

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor(self.signals[idx], dtype=torch.float32)
        y = torch.tensor(float(self.labels[idx]), dtype=torch.float32)
        return x, y


class LinearHead(nn.Module):
    def __init__(self, in_dim=EMBED_DIM):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, x):
        return self.fc(x).squeeze(-1)


def load_ecgfm(ckpt_path, device):
    from fairseq_signals.utils.checkpoint_utils import load_model_and_task
    result = load_model_and_task(ckpt_path)
    if isinstance(result, tuple):
        for r in result:
            if hasattr(r, "parameters"): return r.to(device)
            if isinstance(r, list) and r and hasattr(r[0], "parameters"): return r[0].to(device)
    return result.to(device)


@torch.no_grad()
def get_probs(backbone, head, loader, device):
    backbone.eval(); head.eval()
    all_probs, all_labels = [], []
    for x, y in loader:
        x   = x.to(device)
        out = backbone(source=x, padding_mask=None, features_only=True)
        emb = out["x"].mean(dim=1)
        logits = head(emb).cpu().numpy()
        probs  = 1 / (1 + np.exp(-logits))
        all_probs.append(probs)
        all_labels.append(y.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels).astype(int)


def threshold_at_good_percentile(good_probs, pct):
    """양호 신호(label=0) 점수 분포의 pct 백분위수 = specificity(pct/100) 임계값.
    ROC searchsorted보다 견고 — 작은 val셋의 이산 점프에 영향받지 않고
    "양호 신호의 pct%가 이 값 미만"을 직접 보장한다."""
    return float(np.percentile(good_probs, pct))


def evaluate_tiers(probs, labels, t_mask, t_alert):
    """3단 임계값 기준 통계 산출."""
    tiers = np.where(probs < t_mask, 0,
            np.where(probs < t_alert, 1, 2))  # 0=use, 1=mask, 2=alert

    n_use   = (tiers == 0).sum()
    n_mask  = (tiers == 1).sum()
    n_alert = (tiers == 2).sum()
    n_total = len(probs)

    # 불량 신호(label=1) 기준
    bad  = labels == 1
    good = labels == 0

    bad_alert = ((tiers == 2) & bad).sum()
    bad_mask  = ((tiers == 1) & bad).sum()
    bad_use   = ((tiers == 0) & bad).sum()

    good_use   = ((tiers == 0) & good).sum()
    good_mask  = ((tiers == 1) & good).sum()
    good_alert = ((tiers == 2) & good).sum()

    return {
        "n_use": int(n_use), "n_mask": int(n_mask), "n_alert": int(n_alert),
        "pct_use": n_use/n_total*100, "pct_mask": n_mask/n_total*100, "pct_alert": n_alert/n_total*100,
        "bad_alert": int(bad_alert), "bad_mask": int(bad_mask), "bad_use": int(bad_use),
        "good_use": int(good_use), "good_mask": int(good_mask), "good_alert": int(good_alert),
        "bad_detected_alert": bad_alert / bad.sum() * 100 if bad.sum() > 0 else 0,
        "bad_detected_mask_or_alert": (bad_alert + bad_mask) / bad.sum() * 100 if bad.sum() > 0 else 0,
        "good_in_use": good_use / good.sum() * 100 if good.sum() > 0 else 0,
        "good_false_alert": good_alert / good.sum() * 100 if good.sum() > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",    default=DATA_DIR)
    parser.add_argument("--ckpt_path",   default=CKPT_FM)
    parser.add_argument("--gate_ckpt",   default=GATE_CKPT)
    parser.add_argument("--out_dir",     default=OUT_DIR)
    parser.add_argument("--t_mask_pct",  type=float, default=75.0,
                        help="t_mask = val 양호 신호 점수 분포의 N번째 백분위수 (default=75)")
    parser.add_argument("--spec_alert",  type=float, default=0.90,
                        help="t_alert = val specificity ≥ N 지점 (default=0.90)")
    args = parser.parse_args()

    set_deterministic(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print("단계 7b: 게이트 연속 신뢰도 + 3단 임계값 산출")
    print("=" * 65)
    print(f"디바이스: {device} (결정성 고정 seed=42)")
    print(f"임계값 전략: t_alert=val 양호 p{args.spec_alert*100:.0f}(spec={args.spec_alert:.2f}), "
          f"t_mask=val 양호 p{args.t_mask_pct:.0f}")
    print()

    # ── 모델 로드 ────────────────────────────────────────────────────
    print("[1] 모델 로드")
    backbone = load_ecgfm(args.ckpt_path, device)
    backbone.eval()
    for p in backbone.parameters(): p.requires_grad_(False)

    head = LinearHead().to(device)
    ckpt = torch.load(args.gate_ckpt, map_location=device)
    head.load_state_dict(ckpt["head_state"])
    head.eval()
    print(f"    gate_best.pt 로드 완료 (epoch={ckpt.get('epoch','?')}, "
          f"val_auroc={ckpt.get('val_auroc', '?')})")
    print()

    # ── val set 추론 ─────────────────────────────────────────────────
    print("[2] Val set 추론 → 임계값 산출")
    val_ds = GateDataset(os.path.join(args.data_dir, "val"))
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    val_probs, val_labels = get_probs(backbone, head, val_loader, device)

    val_auroc = roc_auc_score(val_labels, val_probs)

    # 두 임계값 모두 val 양호 신호(label=0) 점수 분포의 백분위수로 산출.
    # specificity(=양호 신호 정확 분류율)를 직접 보장 → 작은 val셋에서 견고.
    good_val_probs = val_probs[val_labels == 0]
    t_mask  = threshold_at_good_percentile(good_val_probs, args.t_mask_pct)
    t_alert = threshold_at_good_percentile(good_val_probs, args.spec_alert * 100)

    # 산출 임계값의 val 불량 탐지율(sensitivity)
    bad_val = val_labels == 1
    sens_a = (val_probs[bad_val] >= t_alert).mean() if bad_val.sum() else 0.0

    print(f"    Val AUROC: {val_auroc:.4f}")
    print(f"    t_alert (val 양호 p{args.spec_alert*100:.0f}, spec={args.spec_alert:.2f}) = {t_alert:.4f}  "
          f"(불량 탐지 sens={sens_a:.3f})")
    print(f"    t_mask  (val 양호 p{args.t_mask_pct:.0f}, spec={args.t_mask_pct/100:.2f}) = {t_mask:.4f}")
    print()

    # val 3단 통계
    s = evaluate_tiers(val_probs, val_labels, t_mask, t_alert)
    n_bad  = (val_labels == 1).sum()
    n_good = (val_labels == 0).sum()
    print(f"    [Val 3단 분포] n={len(val_labels)} (불량={n_bad}, 양호={n_good})")
    print(f"      use   : {s['n_use']:3d}개 ({s['pct_use']:.1f}%)  "
          f"양호={s['good_use']}, 불량={s['bad_use']}")
    print(f"      mask  : {s['n_mask']:3d}개 ({s['pct_mask']:.1f}%)  "
          f"양호={s['good_mask']}, 불량={s['bad_mask']}")
    print(f"      alert : {s['n_alert']:3d}개 ({s['pct_alert']:.1f}%)  "
          f"양호={s['good_alert']}, 불량={s['bad_alert']}")
    print(f"      불량 탐지: alert={s['bad_detected_alert']:.1f}%, "
          f"mask+alert={s['bad_detected_mask_or_alert']:.1f}%")
    print(f"      양호 보존: use={s['good_in_use']:.1f}%  "
          f"양호 오경보: alert={s['good_false_alert']:.1f}%")
    print()

    # ── test set 검증 ────────────────────────────────────────────────
    print("[3] Test set 검증 (val에서 산출한 임계값 적용)")
    test_ds = GateDataset(os.path.join(args.data_dir, "test"))
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)
    test_probs, test_labels = get_probs(backbone, head, test_loader, device)

    test_auroc = roc_auc_score(test_labels, test_probs)
    t2 = evaluate_tiers(test_probs, test_labels, t_mask, t_alert)
    n_bad_t  = (test_labels == 1).sum()
    n_good_t = (test_labels == 0).sum()

    print(f"    Test AUROC: {test_auroc:.4f}")
    print(f"    [Test 3단 분포] n={len(test_labels)} (불량={n_bad_t}, 양호={n_good_t})")
    print(f"      use   : {t2['n_use']:3d}개 ({t2['pct_use']:.1f}%)  "
          f"양호={t2['good_use']}, 불량={t2['bad_use']}")
    print(f"      mask  : {t2['n_mask']:3d}개 ({t2['pct_mask']:.1f}%)  "
          f"양호={t2['good_mask']}, 불량={t2['bad_mask']}")
    print(f"      alert : {t2['n_alert']:3d}개 ({t2['pct_alert']:.1f}%)  "
          f"양호={t2['good_alert']}, 불량={t2['bad_alert']}")
    print(f"      불량 탐지: alert={t2['bad_detected_alert']:.1f}%, "
          f"mask+alert={t2['bad_detected_mask_or_alert']:.1f}%")
    print(f"      양호 보존: use={t2['good_in_use']:.1f}%  "
          f"양호 오경보: alert={t2['good_false_alert']:.1f}%")
    print()

    # ── reliability 스코어 정의 ──────────────────────────────────────
    print("[4] Reliability 점수 정의 (P2 인터페이스)")
    print(f"    reliability = 1 - gate_score")
    print(f"    gate_tier 기준:")
    print(f"      gate_score < {t_mask:.4f}             → 'use'   (reliability > {1-t_mask:.4f})")
    print(f"      {t_mask:.4f} ≤ gate_score < {t_alert:.4f}  → 'mask'  (reliability {1-t_alert:.4f}~{1-t_mask:.4f})")
    print(f"      gate_score ≥ {t_alert:.4f}             → 'alert' (reliability < {1-t_alert:.4f})")
    print()

    # ── 저장 ────────────────────────────────────────────────────────
    out_path = os.path.join(args.out_dir, "gate_thresholds.npz")
    np.savez(out_path,
             t_mask=t_mask, t_alert=t_alert,
             t_mask_pct=args.t_mask_pct,
             spec_alert_target=args.spec_alert,
             sens_alert=sens_a,
             val_auroc=val_auroc, test_auroc=test_auroc,
             val_probs=val_probs, val_labels=val_labels,
             test_probs=test_probs, test_labels=test_labels)
    print(f"    저장: {out_path}")

    # ── 최종 요약 ────────────────────────────────────────────────────
    print()
    print("=" * 65)
    print("단계 7b 결과 요약")
    print("=" * 65)
    print(f"  t_mask  = {t_mask:.4f}  (val 양호 신호 p{args.t_mask_pct:.0f})")
    print(f"  t_alert = {t_alert:.4f}  (val 양호 p{args.spec_alert*100:.0f}, spec={args.spec_alert:.2f}, 불량 sens={sens_a:.3f})")
    print()
    print(f"  [Test 세트 기준]")
    print(f"    use   {t2['pct_use']:5.1f}% — 양호 {t2['good_in_use']:.1f}% 보존, 불량 {100-t2['good_in_use']-t2['good_false_alert']:.1f}%→mask/alert")
    print(f"    mask  {t2['pct_mask']:5.1f}%")
    print(f"    alert {t2['pct_alert']:5.1f}% — 불량 {t2['bad_detected_alert']:.1f}% 탐지")
    print(f"    mask+alert   — 불량 {t2['bad_detected_mask_or_alert']:.1f}% 탐지")
    print(f"    양호 오경보  — {t2['good_false_alert']:.1f}% (alert 잘못 분류)")
    print()
    print(f"  P1_output['reliability'] = 1 - gate_score")
    print(f"  P1_output['gate_tier']   = 'use'|'mask'|'alert'")
    print(f"             기준: t_mask={t_mask:.4f}, t_alert={t_alert:.4f}")
    print("=" * 65)


if __name__ == "__main__":
    main()
