"""
단계 9 전처리: INCART 외부검증 데이터 준비
==========================================
목적:
  12-lead 30분 홀터 ECG (INCART) → 단계 9 추론용.
  AF 리듬 레코드 = 응급(1), 동리듬 지배 레코드 = 정상(0).

데이터:
  PhysioNet St.-Petersburg INCART 12-lead Arrhythmia Database
  75 레코드 (I01~I75), 12-lead, 257Hz, ~30분/레코드
  어노테이션: .atr (beat-level + 일부 rhythm '+' 어노테이션)

라벨 전략:
  ① AFIB/WPWAF 레코드 ('+' aux_note에 (AFIB 또는 (WPWAF 포함):
     → 해당 레코드 전 윈도우 = 응급(1)
  ② 정상 레코드 (rhythm 어노테이션 없고 N 비트 지배):
     → N 비트 ≥ 90% 윈도우 = 정상(0)
  ③ 기타 (부분 AFIB, PREX, VT 등): 제외

처리 과정:
  1. wfdb 어노테이션 확인 → AFIB 레코드 분류
  2. scipy.signal.resample: 257Hz → 500Hz
  3. 비중첩 10s 윈도우 (5,000 샘플 @500Hz)
  4. 비트 어노테이션을 500Hz 공간으로 스케일 → 윈도우 라벨 결정

출력:
  --out_dir/signals.npy   : float32 (N, 12, 5000)
  --out_dir/labels.npy    : int8    (N,) 0=정상, 1=응급
  --out_dir/record_ids.npy: str     (N,) '레코드명_w윈도우인덱스'

사용법:
  python scripts/preprocess_incart.py
"""

import argparse
import os
import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

try:
    import wfdb
except ImportError:
    sys.exit("[오류] wfdb 미설치 — pip install wfdb")

try:
    from scipy.signal import resample as sp_resample
except ImportError:
    sys.exit("[오류] scipy 미설치 — pip install scipy")

FS_IN       = 257
FS_OUT      = 500
N_LEADS     = 12
SEG_LEN_OUT = FS_OUT * 10   # 5,000

# 응급 리듬 어노테이션 (aux_note 값)
EMERGENCY_RHYTHMS = {"(AFIB", "(WPWAF"}

# 정상 윈도우 판정: N 비트 비율 임계값
NORMAL_N_RATIO = 0.90


def is_afib_record(ann_path):
    """
    .atr 파일에서 AFIB/WPWAF 리듬 어노테이션 유무 확인.
    반환: (bool, list_of_rhythms)
    """
    ann = wfdb.rdann(ann_path, "atr")
    rhythm_notes = [
        ann.aux_note[i]
        for i, s in enumerate(ann.symbol)
        if s == "+"
    ]
    emg_rhythms = [r for r in rhythm_notes if r.strip() in EMERGENCY_RHYTHMS]
    return bool(emg_rhythms), rhythm_notes


def classify_windows(ann, n_windows, fs_ratio):
    """
    beat 어노테이션으로 각 윈도우 라벨 결정 (정상 전용 — AFIB 레코드는 이 함수 미사용).

    ann     : wfdb annotation 객체
    n_windows: 총 윈도우 수
    fs_ratio : FS_OUT / FS_IN (어노테이션 샘플 → 500Hz 스케일 계수)

    반환: list of (window_idx, label)  label=0(정상) only; -1=제외
    """
    # beat 샘플 → 500Hz 공간 → 윈도우 인덱스
    beat_samples_scaled = np.array(ann.sample, dtype=float) * fs_ratio
    beat_wins = (beat_samples_scaled / SEG_LEN_OUT).astype(int)

    # 윈도우별 beat type 집계
    win_beats = [[] for _ in range(n_windows)]
    for i, w in enumerate(beat_wins):
        if 0 <= w < n_windows:
            win_beats[w].append(ann.symbol[i])

    result = []
    for w in range(n_windows):
        beats = win_beats[w]
        if len(beats) < 2:      # 비트가 너무 적음 → 제외
            result.append(-1)
            continue

        n_normal = sum(1 for b in beats if b == "N")
        ratio = n_normal / len(beats)

        if ratio >= NORMAL_N_RATIO:
            result.append(0)    # 정상
        else:
            result.append(-1)   # 혼합/이상 → 제외

    return result


def process_record(rec_path, label_override=None):
    """
    단일 레코드 처리.
    label_override=1 : AFIB 레코드 → 전 윈도우 응급
    label_override=None : 비트 어노테이션으로 per-window 라벨

    반환: (windows list, labels list)  각 (12, 5000) float32
    """
    # 신호 로드
    try:
        rec = wfdb.rdrecord(rec_path)
    except Exception as e:
        print(f"  [경고] {os.path.basename(rec_path)}: 로드 실패 — {e}")
        return [], []

    if rec.fs != FS_IN:
        print(f"  [경고] {os.path.basename(rec_path)}: fs={rec.fs} (기대={FS_IN}) — 스킵")
        return [], []

    sig = rec.p_signal
    if sig is None:
        return [], []

    sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # 리샘플: (T_in, 12) → (T_out, 12)
    T_in = sig.shape[0]
    T_out = int(T_in * FS_OUT / FS_IN)
    sig_rs = sp_resample(sig, T_out, axis=0).astype(np.float32)

    # 전치: (T_out, 12) → (12, T_out)
    sig_rs = sig_rs.T

    n_windows = T_out // SEG_LEN_OUT

    # 라벨 결정
    if label_override is not None:
        win_labels = [label_override] * n_windows
    else:
        ann = wfdb.rdann(rec_path, "atr")
        fs_ratio = FS_OUT / FS_IN
        win_labels = classify_windows(ann, n_windows, fs_ratio)

    windows_out, labels_out = [], []
    for w in range(n_windows):
        lbl = win_labels[w]
        if lbl < 0:
            continue
        seg = sig_rs[:, w * SEG_LEN_OUT: (w + 1) * SEG_LEN_OUT]
        windows_out.append(seg)
        labels_out.append(lbl)

    return windows_out, labels_out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_dir",
        default="data/raw/incart",
    )
    parser.add_argument(
        "--out_dir",
        default="data/processed/incart",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 65)
    print("단계 9 전처리: INCART (12-lead Holter, AF/정상)")
    print("=" * 65)
    print(f"입력: {args.data_dir}")
    print(f"출력: {args.out_dir}")
    print()

    # ── 1. 레코드 분류 ────────────────────────────────────────────────
    print("[1] 레코드 분류 (AFIB 유무)")
    atr_files = sorted([f[:-4] for f in os.listdir(args.data_dir) if f.endswith(".atr")])

    afib_recs   = []   # 전 윈도우 → 응급
    normal_recs = []   # N 지배 윈도우 → 정상

    for rec_name in atr_files:
        ann_path = os.path.join(args.data_dir, rec_name)
        is_afib, rhythms = is_afib_record(ann_path)
        if is_afib:
            afib_recs.append(rec_name)
            print(f"  AFIB: {rec_name}  rhythms={rhythms}")
        elif not rhythms:   # 리듬 어노테이션 없는 레코드 = 정상 후보
            normal_recs.append(rec_name)

    # PREX/기타 리듬 레코드는 자동 제외됨 (위 elif 조건)
    excluded = len(atr_files) - len(afib_recs) - len(normal_recs)
    print()
    print(f"  AFIB 레코드:   {len(afib_recs)}개 → 전 윈도우 응급")
    print(f"  정상 후보:     {len(normal_recs)}개 → N 비율로 필터")
    print(f"  기타 제외:     {excluded}개 (PREX/WPWAF/기타 리듬)")
    print()

    # ── 2. 신호 처리 ──────────────────────────────────────────────────
    print("[2] 신호 로드 → 리샘플 (257Hz→500Hz) → 윈도우 → 라벨")
    print("-" * 65)

    all_signals, all_labels, all_rec_ids = [], [], []

    for rec_name in afib_recs:
        rec_path = os.path.join(args.data_dir, rec_name)
        windows, labels = process_record(rec_path, label_override=1)
        for w_idx, (win, lbl) in enumerate(zip(windows, labels)):
            all_signals.append(win)
            all_labels.append(lbl)
            all_rec_ids.append(f"{rec_name}_w{w_idx:04d}")
        print(f"  {rec_name} [AFIB]: {len(windows)}윈도우 → 전부 응급")

    for rec_name in normal_recs:
        rec_path = os.path.join(args.data_dir, rec_name)
        windows, labels = process_record(rec_path, label_override=None)
        n_normal = labels.count(0) if labels else 0
        for w_idx, (win, lbl) in enumerate(zip(windows, labels)):
            all_signals.append(win)
            all_labels.append(lbl)
            all_rec_ids.append(f"{rec_name}_w{w_idx:04d}")
        if windows:
            print(f"  {rec_name}: {len(windows)}윈도우  정상={n_normal}")

    # ── 3. 저장 ───────────────────────────────────────────────────────
    sig_arr = np.stack(all_signals).astype(np.float32)
    lab_arr = np.array(all_labels, dtype=np.int8)
    rid_arr = np.array(all_rec_ids, dtype=object)

    np.save(os.path.join(args.out_dir, "signals.npy"),    sig_arr)
    np.save(os.path.join(args.out_dir, "labels.npy"),     lab_arr)
    np.save(os.path.join(args.out_dir, "record_ids.npy"), rid_arr)

    n_emg = int((lab_arr == 1).sum())
    n_nrm = int((lab_arr == 0).sum())

    print()
    print("=" * 65)
    print("전처리 완료")
    print("=" * 65)
    print(f"  총 윈도우: {len(lab_arr)}개")
    print(f"    응급(AFIB): {n_emg}")
    print(f"    정상:       {n_nrm}")
    print(f"  출력 shape:  {sig_arr.shape}  dtype={sig_arr.dtype}")
    print(f"  저장: {args.out_dir}")
    print()
    print("다음 단계:")
    print("  → python scripts/eval_external.py --db incart")
    print("=" * 65)


if __name__ == "__main__":
    main()
