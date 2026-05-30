"""
P1 cardiac 로깅 어댑터 — 로깅 §2 스키마 레코드 생성
=======================================================
목적: P1 cardiac 추론 + physio(hr/rhythm)를 통합 로깅 스키마(§2 ecg_raw+ecg_outputs)
      레코드로 변환. 로깅 파이프라인(parquet writer + 마스터클럭 동기)이 소비할 단일 진입점.

사용 (로깅 파이프라인에서):
    from p1_cardiac_logging_adapter import CardiacLoggingAdapter
    ad = CardiacLoggingAdapter()                 # P1CardiacChannel 내부 로드
    rec = ad.to_record(ecg_2lead, master_clock_ms=t)   # → dict (§2 스키마)
    # rec를 parquet writer에 append (raw는 별도 store로 분리 가능: include_raw=False)

입력: ecg = (2,5000)[II,V2] 또는 (12,5000). 2리드(II→slot1, V2→slot7).
정직 캐비엇: 모든 cardiac 검증은 단일리드(II) 기준. 2리드(II+V2) 성능은 ≥ 기대되나
            reliability·임계는 단일리드 학습 → 2리드 실데이터로 재검증 필요.
"""
from __future__ import annotations
import os, sys
import numpy as np
from scipy.signal import find_peaks

sys.path.insert(0, os.path.dirname(__file__))
from p1_cardiac_channel import P1CardiacChannel, LEAD   # LEAD=1 (slot II)

V2_SLOT = 7
FS = 500
CARDIAC_PROB_NAMES = ["nsr", "af", "isch", "cond", "ecto"]
MODEL_VERSION = "lora_multitask_snr_a07+relhead_v1"


def estimate_physio(lead_ii: np.ndarray, fs: int = FS):
    """R-peak 기반 hr_bpm·rhythm_regularity (lead II). 미검출 시 (nan,nan)."""
    lead = np.asarray(lead_ii, dtype=np.float32)
    height = max(float(lead.max()) * 0.3, 0.1)
    peaks, _ = find_peaks(lead, height=height, distance=int(fs * 0.3))
    if len(peaks) >= 2:
        rr = np.diff(peaks) / fs
        hr = float(60.0 / rr.mean())
        cv = rr.std() / (rr.mean() + 1e-6)
        rhythm = float(np.clip(1.0 - cv * 3, 0.0, 1.0))
        return hr, rhythm
    return float("nan"), float("nan")


class CardiacLoggingAdapter:
    """cardiac 추론 + physio → §2 로깅 레코드. P2 parquet writer가 소비."""

    def __init__(self, channel: P1CardiacChannel = None, **channel_kw):
        self.ch = channel if channel is not None else P1CardiacChannel(**channel_kw)

    def _prep(self, ecg: np.ndarray):
        """ecg → (추론용 단일리드 12ch, lead_II, lead_V2) 반환.

        ★정책: 실시간 추론은 **단일리드 II만**(검증된 경로).
          V2는 로깅 전용(Phase-3 2리드 재검증·허혈 개선용). 2리드 동시추론 임계는 미검증.
        """
        ecg = np.asarray(ecg, dtype=np.float32)
        if ecg.ndim == 2 and ecg.shape[0] == 12:
            lead_ii, lead_v2 = ecg[LEAD], ecg[V2_SLOT]
        elif ecg.ndim == 2 and ecg.shape[0] == 2:        # [II, V2]
            lead_ii, lead_v2 = ecg[0], ecg[1]
        else:
            raise ValueError(f"ecg shape {ecg.shape} — (2,5000)[II,V2] 또는 (12,5000) 필요")
        # 추론 입력: II만 slot1에 (단일리드, 검증). V2 슬롯은 0 유지.
        sig12 = np.zeros((12, lead_ii.shape[-1]), dtype=np.float32)
        sig12[LEAD] = lead_ii
        return sig12, lead_ii, lead_v2

    def to_record(self, ecg: np.ndarray, master_clock_ms: int,
                  lead_on: bool = True, include_raw: bool = True) -> dict:
        """단일 10s 윈도우 → §2 레코드 dict (master_clock_ms 키).
        추론=단일리드 II(검증), raw 로깅=II+V2 둘 다(Phase-3 재검증용)."""
        sig12, lead_ii, lead_v2 = self._prep(ecg)
        out = self.ch.infer(sig12)                       # single-lead II → 스칼라/벡터
        hr, rhythm = estimate_physio(lead_ii)
        rec = {
            "master_clock_ms":   int(master_clock_ms),
            "ecg_lead_on":       bool(lead_on),
            # ── ecg_outputs (P1 cardiac) ──
            "emergency_score":   float(out["emergency_score"]),
            **{f"cardiac_p_{n}": float(out["cardiac_probs"][i])
               for i, n in enumerate(CARDIAC_PROB_NAMES)},
            "reliability":       float(out["reliability"]),
            "effective_cardiac": float(out["effective_cardiac"]),
            "benign_flag":       bool(out["benign_flag"]),
            "hr_bpm":            hr,
            "rhythm_regularity": rhythm,
            "model_version":     MODEL_VERSION,
        }
        if include_raw:                                   # ecg_raw (별도 store 권장)
            rec["ecg_lead_II"] = lead_ii.astype(np.float32)
            rec["ecg_lead_V2"] = lead_v2.astype(np.float32)
        return rec


# ── 자가 데모/검증: CACHET 몇 윈도우 → 레코드 생성 + (가능 시) parquet ──────────
if __name__ == "__main__":
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
    cs = np.load(r"D:\WidU_ecg-fm_emergency-detection\data\processed\cachet\signals.npy")
    ad = CardiacLoggingAdapter()
    recs = [ad.to_record(cs[i], master_clock_ms=1000 * i, include_raw=False)
            for i in range(3)]
    print("P1 cardiac 로깅 레코드 데모 (CACHET 3 윈도우, raw 제외)")
    for r in recs:
        print(f"  t={r['master_clock_ms']}ms eff={r['effective_cardiac']:.3f} "
              f"rel={r['reliability']:.3f} hr={r['hr_bpm']:.0f} "
              f"p_af={r['cardiac_p_af']:.2f} benign={r['benign_flag']}")
    print("  레코드 키:", list(recs[0].keys()))
    # raw 포함 1개 → 크기 확인
    r_raw = ad.to_record(cs[0], master_clock_ms=0, include_raw=True)
    raw_kb = (r_raw["ecg_lead_II"].nbytes + r_raw["ecg_lead_V2"].nbytes) / 1024
    print(f"  raw 포함 시 윈도우당 +{raw_kb:.0f}KB (II+V2 float32) → 별도 store 권장")
    # parquet 쓰기 데모 (pandas 가용 시)
    try:
        import pandas as pd
        df = pd.DataFrame([{k: v for k, v in r.items()} for r in recs])
        out = r"D:\WidU_ecg-fm_emergency-detection\outputs\gate\_cardiac_log_demo.parquet"
        df.to_parquet(out)
        print(f"  parquet 데모 저장: {out} ({len(df)} rows, {df.shape[1]} cols)")
    except Exception as e:
        print(f"  [parquet 데모 건너뜀: {e}]")
