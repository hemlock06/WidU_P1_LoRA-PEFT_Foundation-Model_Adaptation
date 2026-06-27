"""
단계 9 전처리: LTST 외부검증 데이터 준비
=========================================
목적:
  LTST-DB (Long-Term ST) → 단계 9 추론용.
  ischemic ST-change episode 포함 윈도우 = 응급(1),
  ST 정상 윈도우 = 정상(0).

데이터:
  PhysioNet Long-Term ST Database (ltstdb)
  86 레코드 (s20011~s30801), 2-lead (ML2·MV2), 250Hz, 21~45h/레코드
  어노테이션: .stb — ST reference + ischemic episode 마커

Lead 처리:
  ML2 → 12-lead 슬롯 index 1 (Lead II)
  MV2 → 12-lead 슬롯 index 7 (Lead V2)
  나머지 10개 lead 0-fill (RLM 학습 관례와 동일)

라벨 전략:
  ① 허혈 에피소드 윈도우 (stb의 '(rtst...' ~ 'rtst...)' 사이):
     → 응급(1)
  ② 정상 윈도우 (|LRST| < 10 × 1/100mV = ±0.1mV):
     → 정상(0)
  ③ 에피소드 경계 ±1 윈도우, 고편차(|LRST| ≥ 10) 비에피소드: 제외

처리 과정:
  1. .stb 어노테이션 파싱 → 에피소드 구간 [start_sample, end_sample) 목록
  2. 250Hz → 500Hz 리샘플 (scipy.signal.resample)
  3. 비중첩 10s 윈도우 (5,000 샘플 @500Hz)
  4. 윈도우 라벨 결정 (에피소드 교차 여부 + LRST 범위)

출력:
  --out_dir/signals.npy   : float32 (N, 12, 5000)
  --out_dir/labels.npy    : int8    (N,) 0=정상, 1=응급
  --out_dir/record_ids.npy: str     (N,) '레코드명_w윈도우인덱스'

사용법:
  python scripts/preprocess_ltst.py
  python scripts/preprocess_ltst.py --data_dir .../raw/ltst --out_dir .../processed/ltst
"""

import argparse
import os
import re
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
    from scipy.signal import resample_poly
except ImportError:
    sys.exit("[오류] scipy 미설치 — pip install scipy")

FS_IN = 250
FS_OUT = 500
N_LEADS = 12
SEG_LEN_OUT = FS_OUT * 10  # 5,000 샘플 @500Hz
SEG_LEN_IN = FS_IN * 10  # 2,500 샘플 @250Hz

# Lead 매핑: ML2 → index 1 (Lead II), MV2 → index 7 (Lead V2)
LEAD_MAP = {"ML2": 1, "MV2": 7}

# 정상 판정 기준: 양쪽 lead 모두 |LRST| < 10 (단위: 1/100 mV → 0.1 mV)
NORMAL_LRST_THRESH = 10


# ── 어노테이션 파싱 ──────────────────────────────────────────────────

_EPISODE_START = re.compile(r"^\(rtst", re.IGNORECASE)
_EPISODE_END = re.compile(r"rtst.*\)$", re.IGNORECASE)
_LRST_PAT = re.compile(r"^LRST(\d)([+-]\d+)$", re.IGNORECASE)


def parse_episodes(ann):
    """
    .stb 어노테이션에서 허혈 에피소드 구간 파싱.
    반환: list of (start_sample, end_sample) — 250Hz 원본 좌표
    """
    episodes = []
    ep_start = None

    for i in range(len(ann.symbol)):
        note = ann.aux_note[i].strip()
        sample = ann.sample[i]

        if _EPISODE_START.match(note):
            ep_start = sample
        elif _EPISODE_END.search(note) and ep_start is not None:
            episodes.append((ep_start, sample))
            ep_start = None

    return episodes


def parse_lrst_values(ann):
    """
    .stb 어노테이션에서 LRST 편차값 파싱.
    반환: dict {sample: {lead_idx: value}} — 250Hz 좌표
    """
    lrst_map = {}
    for i in range(len(ann.symbol)):
        note = ann.aux_note[i].strip()
        m = _LRST_PAT.match(note)
        if m:
            lead_idx = int(m.group(1))
            value = int(m.group(2))
            sample = ann.sample[i]
            if sample not in lrst_map:
                lrst_map[sample] = {}
            lrst_map[sample][lead_idx] = value
    return lrst_map


# ── 윈도우 라벨 결정 ─────────────────────────────────────────────────


def label_windows(n_windows, episodes, lrst_map, episode_buf=1):
    """
    n_windows  : 총 윈도우 수
    episodes   : [(start, end)] in 250Hz samples
    lrst_map   : {sample: {lead: value}} in 250Hz samples
    episode_buf: 에피소드 경계 양쪽 ±buf 윈도우 제외 (노이즈 완충)

    반환: list[int]  1=응급, 0=정상, -1=제외
    """
    labels = [-1] * n_windows  # 기본: 제외

    # 에피소드 윈도우 집합
    ep_wins = set()
    buf_wins = set()
    for es, ee in episodes:
        w_start = es // SEG_LEN_IN
        w_end = ee // SEG_LEN_IN
        for w in range(max(0, w_start), min(n_windows, w_end + 1)):
            ep_wins.add(w)
        for w in range(
            max(0, w_start - episode_buf), min(n_windows, w_end + 1 + episode_buf)
        ):
            buf_wins.add(w)

    # LRST 기반 정상 윈도우 판정
    # 가장 가까운 LRST 어노테이션을 각 윈도우에 대응
    lrst_samples = sorted(lrst_map.keys())
    win_lrst = {}  # {win_idx: max_abs_lrst}
    for s in lrst_samples:
        w = s // SEG_LEN_IN
        if 0 <= w < n_windows:
            vals = lrst_map[s].values()
            max_abs = max(abs(v) for v in vals) if vals else 0
            if w not in win_lrst:
                win_lrst[w] = max_abs
            else:
                win_lrst[w] = max(win_lrst[w], max_abs)

    for w in range(n_windows):
        if w in ep_wins:
            labels[w] = 1  # 응급
        elif w in buf_wins:
            labels[w] = -1  # 에피소드 경계 완충 → 제외
        elif w in win_lrst:
            if win_lrst[w] < NORMAL_LRST_THRESH:
                labels[w] = 0  # 정상 (LRST 낮음)
            else:
                labels[w] = -1  # 고편차 비에피소드 → 제외
        # else: LRST 어노테이션 없는 윈도우 → 제외 유지

    return labels


# ── 신호 처리 ────────────────────────────────────────────────────────


def process_record(rec_path, sig_names):
    """
    단일 LTST 레코드 → 리샘플. 12-lead 슬롯 배치는 윈도우 단위로 이후 수행.
    (전체 (12, T_out) array를 만들지 않아 긴 레코드 OOM 회피)

    반환: (sig_rs, slot_indices, n_windows)
      sig_rs        : float32 (T_out, n_sig) — 2-lead 원본 유지
      slot_indices  : list[(ch_idx, slot)] — ML2→1, MV2→7 등 매핑 정보
      n_windows     : T_out // SEG_LEN_OUT
    """
    rec = wfdb.rdrecord(rec_path)

    if rec.fs != FS_IN:
        raise ValueError(f"fs={rec.fs} (기대={FS_IN})")

    sig = rec.p_signal  # (T_in, n_sig)
    sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # 250Hz → 500Hz 리샘플 (resample_poly: polyphase 필터, FFT 대비 메모리 효율적)
    sig_rs = resample_poly(sig, up=2, down=1, axis=0).astype(
        np.float32
    )  # (T_out, n_sig)
    T_out = sig_rs.shape[0]

    # 12-lead 슬롯 매핑 정보만 미리 산출 (실제 배치는 윈도우 루프에서)
    slot_indices = []
    for ch_idx, name in enumerate(sig_names):
        slot = LEAD_MAP.get(name.strip(), None)
        if slot is not None and ch_idx < sig_rs.shape[1]:
            slot_indices.append((ch_idx, slot))

    n_windows = T_out // SEG_LEN_OUT
    return sig_rs, slot_indices, n_windows


def make_window_12lead(sig_rs, slot_indices, w):
    """
    윈도우 인덱스 w에 해당하는 (12, SEG_LEN_OUT) 슬롯 배치 array 생성.
    윈도우당 240 KB 정도 — 전체 (12, T_out) 폭발 회피.
    """
    seg = np.zeros((N_LEADS, SEG_LEN_OUT), dtype=np.float32)
    start = w * SEG_LEN_OUT
    end = start + SEG_LEN_OUT
    for ch_idx, slot in slot_indices:
        seg[slot] = sig_rs[start:end, ch_idx]
    return seg


# ── 메인 ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data_dir", default="data/raw/ltst")
    parser.add_argument("--out_dir", default="data/processed/ltst")
    parser.add_argument(
        "--episode_buf",
        type=int,
        default=1,
        help="에피소드 경계 ±buf 윈도우 제외 (기본 1)",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 65)
    print("단계 9 전처리: LTST (2-lead 장기 ST, 허혈 에피소드)")
    print("=" * 65)
    print(f"입력: {args.data_dir}")
    print(f"출력: {args.out_dir}")
    print()

    # 레코드 목록 (헤더 파일 기준)
    hea_files = sorted(f[:-4] for f in os.listdir(args.data_dir) if f.endswith(".hea"))
    print(f"발견된 레코드: {len(hea_files)}개")
    print()

    all_signals, all_labels, all_rec_ids = [], [], []
    stat = {"ep_win": 0, "norm_win": 0, "exc_win": 0, "skipped": 0}

    for rec_name in hea_files:
        rec_path = os.path.join(args.data_dir, rec_name)

        # .stb 어노테이션 확인
        stb_path = os.path.join(args.data_dir, rec_name + ".stb")
        if not os.path.exists(stb_path):
            print(f"  [건너뜀] {rec_name}: .stb 없음")
            stat["skipped"] += 1
            continue

        try:
            ann = wfdb.rdann(rec_path, "stb")
            episodes = parse_episodes(ann)
            lrst_map = parse_lrst_values(ann)

            hdr = wfdb.rdheader(rec_path)
            sig_rs, slot_indices, n_windows = process_record(rec_path, hdr.sig_name)

            win_labels = label_windows(
                n_windows, episodes, lrst_map, episode_buf=args.episode_buf
            )

        except Exception as e:
            print(f"  [오류] {rec_name}: {e}")
            stat["skipped"] += 1
            continue

        n_ep = win_labels.count(1)
        n_norm = win_labels.count(0)
        n_exc = win_labels.count(-1)
        stat["ep_win"] += n_ep
        stat["norm_win"] += n_norm
        stat["exc_win"] += n_exc

        # 윈도우 단위 12-lead 슬롯 배치 (메모리 효율)
        for w, lbl in enumerate(win_labels):
            if lbl < 0:
                continue
            seg = make_window_12lead(sig_rs, slot_indices, w)
            all_signals.append(seg)
            all_labels.append(lbl)
            all_rec_ids.append(f"{rec_name}_w{w:05d}")

        # 레코드 단위 신호 메모리 즉시 해제 (다음 레코드 메모리 확보)
        del sig_rs

        print(
            f"  {rec_name}: total={n_windows}  응급={n_ep}  정상={n_norm}  제외={n_exc}  "
            f"에피소드={len(episodes)}개"
        )

    # ── 저장 ──────────────────────────────────────────────────────────
    if not all_signals:
        print("[오류] 유효 윈도우 없음 — 저장 중단")
        return

    sig_arr = np.stack(all_signals).astype(np.float32)
    lab_arr = np.array(all_labels, dtype=np.int8)
    rid_arr = np.array(all_rec_ids, dtype=object)

    np.save(os.path.join(args.out_dir, "signals.npy"), sig_arr)
    np.save(os.path.join(args.out_dir, "labels.npy"), lab_arr)
    np.save(os.path.join(args.out_dir, "record_ids.npy"), rid_arr)

    n_emg = int((lab_arr == 1).sum())
    n_nrm = int((lab_arr == 0).sum())

    print()
    print("=" * 65)
    print("전처리 완료")
    print("=" * 65)
    print(
        f"  처리 레코드: {len(hea_files) - stat['skipped']}개  (건너뜀 {stat['skipped']}개)"
    )
    print(f"  총 윈도우:  {len(lab_arr)}개")
    print(f"    응급(허혈 ST): {n_emg}")
    print(f"    정상:          {n_nrm}")
    print(f"    제외:          {stat['exc_win']}")
    print("  Lead 배치: ML2→슬롯1(II), MV2→슬롯7(V2), 나머지 0-fill")
    print(f"  출력 shape: {sig_arr.shape}  dtype={sig_arr.dtype}")
    print(f"  저장: {args.out_dir}")
    print()
    print("다음 단계:")
    print("  → python scripts/eval_external.py --dbs ltst")
    print("=" * 65)


if __name__ == "__main__":
    main()
