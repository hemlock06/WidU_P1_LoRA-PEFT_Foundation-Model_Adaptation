"""
단계 9 전처리: STAFF-III 외부검증 데이터 준비
=============================================
목적:
  관상동맥 풍선성형술(PTCA) ECG → 단계 9 추론용.
  balloon inflation 중 급성 허혈 = 응급(1), 시술 전 기준선 = 정상(0).

데이터:
  PhysioNet STAFF-III: 104 patients × multiple segments/patient
  레코드 명명: NNNx (NNN=환자번호, x=구간 문자)
    - 'a' : 시술 전 기준선 (baseline)    → 정상(0)
    - 'b' : 풍선 팽창 중 (acute ischemia) → 응급(1)
    - 'c','d','e','f' : 회복/추가 팽창    → 제외 (ambiguous)
  포맷: 9-lead [V1,V2,V3,V4,V5,V6,I,II,III], 1000Hz, ~5분 (~300,000 샘플)
  주의: 일부 레코드는 .hea 없이 .dat만 존재 → 자동 스킵

처리 과정:
  1. 파일명 끝 문자로 라벨 결정 (a→0, b→1)
  2. .hea 없는 레코드 스킵
  3. scipy.signal.resample: 1000Hz → 500Hz
  4. 비중첩 10s 윈도우 (5,000 샘플 @500Hz)
  5. 9-lead → 표준 12-lead 재배열 + aVR/aVL/aVF 0-fill
  6. NaN/Inf → 0 클리핑

리드 매핑:
  STAFF-III 순서: [V1, V2, V3, V4, V5, V6, I, II, III]
  표준 12-lead:   [I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6]
    idx 6(I)  → slot 0,  idx 7(II) → slot 1,  idx 8(III) → slot 2
    slots 3,4,5 (aVR, aVL, aVF) → 0-fill
    idx 0(V1) → slot 6, ..., idx 5(V6) → slot 11

출력:
  --out_dir/signals.npy   : float32 (N, 12, 5000)
  --out_dir/labels.npy    : int8    (N,) 0=정상, 1=응급
  --out_dir/record_ids.npy: str     (N,) '레코드명_w윈도우인덱스'

사용법:
  python scripts/preprocess_staffiii.py
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

FS_IN       = 1000
FS_OUT      = 500
N_LEADS     = 12
SEG_LEN_OUT = FS_OUT * 10   # 5,000

# 라벨 결정 규칙 (파일명 끝 문자)
SEG_LABEL = {"a": 0, "b": 1}   # 'a'=정상, 'b'=응급; 나머지 제외

# STAFF-III lead → 표준 12-lead 슬롯 매핑
# STAFF-III 순서: V1(0), V2(1), V3(2), V4(3), V5(4), V6(5), I(6), II(7), III(8)
LEAD_REMAP = {
    6:  0,   # I   → slot 0
    7:  1,   # II  → slot 1
    8:  2,   # III → slot 2
    # slots 3,4,5 (aVR,aVL,aVF) = 0
    0:  6,   # V1  → slot 6
    1:  7,   # V2  → slot 7
    2:  8,   # V3  → slot 8
    3:  9,   # V4  → slot 9
    4: 10,   # V5  → slot 10
    5: 11,   # V6  → slot 11
}


def remap_leads(sig_t9):
    """
    sig_t9: (T, 9) STAFF-III 순서
    반환:  (12, T) 표준 12-lead 슬롯 (aVR/aVL/aVF=0)
    """
    T = sig_t9.shape[0]
    out = np.zeros((N_LEADS, T), dtype=np.float32)
    for src_idx, dst_slot in LEAD_REMAP.items():
        out[dst_slot] = sig_t9[:, src_idx]
    return out   # (12, T)


def load_record(rec_path):
    """
    wfdb 레코드 로드. 실패 시 None 반환.
    반환: (T, 9) float32 또는 None
    """
    try:
        rec = wfdb.rdrecord(rec_path)
    except Exception as e:
        print(f"  [경고] {os.path.basename(rec_path)}: 로드 실패 — {e}")
        return None

    if rec.fs != FS_IN:
        print(f"  [경고] {os.path.basename(rec_path)}: fs={rec.fs} (기대={FS_IN}) — 스킵")
        return None

    sig = rec.p_signal
    if sig is None or np.isnan(sig).all():
        print(f"  [경고] {os.path.basename(rec_path)}: 신호 없음 — 스킵")
        return None

    # NaN/Inf → 0 (일부 lead가 부분적으로 NaN일 수 있음)
    sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return sig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_dir",
        default="data/raw/staffiii",
    )
    parser.add_argument(
        "--out_dir",
        default="data/processed/staffiii",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 65)
    print("단계 9 전처리: STAFF-III (PCI 허혈 ECG)")
    print("=" * 65)
    print(f"입력: {args.data_dir}")
    print(f"출력: {args.out_dir}")
    print()

    # ── 1. 레코드 목록 수집 ───────────────────────────────────────────
    print("[1] 레코드 스캔 (a=정상, b=응급, 나머지 제외)")
    hea_files = sorted([f[:-4] for f in os.listdir(args.data_dir) if f.endswith(".hea")])

    counts = {"a": 0, "b": 0, "other": 0}
    selected = []   # (rec_name, label)

    for name in hea_files:
        seg_char = name[-1].lower()
        if seg_char in SEG_LABEL:
            selected.append((name, SEG_LABEL[seg_char]))
            counts[seg_char] += 1
        else:
            counts["other"] += 1

    print(f"  .hea 파일: {len(hea_files)}개")
    print(f"    'a' (정상): {counts['a']}개")
    print(f"    'b' (응급): {counts['b']}개")
    print(f"    기타 제외:  {counts['other']}개")
    print()

    # ── 2. 신호 처리 ──────────────────────────────────────────────────
    print("[2] 신호 로드 → 리샘플 (1000Hz→500Hz) → 윈도우 → 리드 재배열")
    print("-" * 65)

    signals_out, labels_out, rec_ids_out = [], [], []
    skip_rec = 0

    for rec_name, label in selected:
        rec_path = os.path.join(args.data_dir, rec_name)
        sig = load_record(rec_path)
        if sig is None:
            skip_rec += 1
            continue

        T_in, n_ch = sig.shape
        if n_ch != 9:
            print(f"  [경고] {rec_name}: {n_ch} leads (기대=9) — 스킵")
            skip_rec += 1
            continue

        # 리샘플: (T_in, 9) → (T_out, 9)
        T_out = int(T_in * FS_OUT / FS_IN)
        sig_rs = sp_resample(sig, T_out, axis=0).astype(np.float32)

        # 리드 재배열: (T_out, 9) → (12, T_out)
        sig_12 = remap_leads(sig_rs)   # (12, T_out)

        # 비중첩 10s 윈도우
        n_windows = T_out // SEG_LEN_OUT
        for w in range(n_windows):
            seg = sig_12[:, w * SEG_LEN_OUT: (w + 1) * SEG_LEN_OUT]   # (12, 5000)
            signals_out.append(seg)
            labels_out.append(label)
            rec_ids_out.append(f"{rec_name}_w{w:03d}")

    # ── 3. 저장 ───────────────────────────────────────────────────────
    sig_arr = np.stack(signals_out).astype(np.float32)
    lab_arr = np.array(labels_out, dtype=np.int8)
    rid_arr = np.array(rec_ids_out, dtype=object)

    np.save(os.path.join(args.out_dir, "signals.npy"),    sig_arr)
    np.save(os.path.join(args.out_dir, "labels.npy"),     lab_arr)
    np.save(os.path.join(args.out_dir, "record_ids.npy"), rid_arr)

    n_emg = int((lab_arr == 1).sum())
    n_nrm = int((lab_arr == 0).sum())

    print()
    print("=" * 65)
    print("전처리 완료")
    print("=" * 65)
    print(f"  처리 레코드: {len(selected) - skip_rec}개  (스킵: {skip_rec}개)")
    print(f"  총 윈도우:   {len(lab_arr)}개")
    print(f"    응급 ('b'): {n_emg}")
    print(f"    정상 ('a'): {n_nrm}")
    print(f"  출력 shape:  {sig_arr.shape}  dtype={sig_arr.dtype}")
    print(f"  저장: {args.out_dir}")
    print()
    print("다음 단계:")
    print("  → python scripts/eval_external.py --db staffiii")
    print("=" * 65)


if __name__ == "__main__":
    main()
