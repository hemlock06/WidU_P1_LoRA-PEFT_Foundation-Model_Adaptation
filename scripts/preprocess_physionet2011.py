"""
단계 7 전처리: PhysioNet 2011 신호품질 게이트 데이터 준비
==========================================================
목적:
  PhysioNet/CinC Challenge 2011 set-a (1,000개 레코드, 998개 라벨 확인)를
  신호품질 게이트 학습용 이진 라벨로 변환·저장.

라벨 규약:
  1 = unacceptable (품질 불량, positive class — 게이트가 탐지해야 할 대상)
  0 = acceptable   (품질 양호, negative class)

근거:
  - 게이트 목적 = 불량 신호 탐지 후 거부. 불량이 탐지 대상(positive).
  - Sensitivity@95%Sp = 5% 오거부율에서 불량 신호 탐지율 → 게이트 핵심 지표
  - 불량(225)이 소수 클래스 → train_gate.py 에서 pos_weight=3.44로 보정

데이터 출처:
  PhysioNet CinC Challenge 2011 set-a
  - RECORDS-acceptable  : 품질 양호 레코드 ID 목록
  - RECORDS-unacceptable: 품질 불량 레코드 ID 목록
  - 각 레코드: 12-lead, 500Hz, 5000샘플 (10초) — ECG-FM 입력과 완벽 일치

Split:
  record-level random 70/15/15, seed=42

출력 (--out_dir/{train,val,test}/):
  signals.npy   : float32 (N, 12, 5000)
  labels.npy    : int8    (N,)  0=양호, 1=불량
  record_ids.npy: str array (N,)

사용법:
  python scripts/preprocess_physionet2011.py \\
      --data_dir D:/WidU_ecg-fm_emergency-detection/data/raw/physionet2011/set-a \\
      --out_dir  D:/WidU_ecg-fm_emergency-detection/data/processed/physionet2011
"""

import argparse
import os
import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

FS_REQUIRED = 500
N_LEADS     = 12
N_SAMPLES   = 5000   # 10s × 500Hz


def read_record_list(fpath: str) -> list:
    """RECORDS-acceptable / RECORDS-unacceptable 파일에서 레코드 ID 목록 읽기."""
    ids = []
    if not os.path.exists(fpath):
        return ids
    with open(fpath, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.append(line)
    return ids


def load_signal(record_path: str) -> "np.ndarray | None":
    """
    wfdb 레코드 로드.
    record_path: 확장자 없는 경로 (예: .../set-a/1002603)
    반환: float32 (12, 5000) 또는 None
    """
    try:
        import wfdb
    except ImportError:
        sys.exit("[오류] wfdb 미설치 — pip install wfdb")

    try:
        record = wfdb.rdrecord(record_path)
    except Exception as e:
        print(f"  [경고] {os.path.basename(record_path)} 읽기 실패: {e}")
        return None

    if record.fs != FS_REQUIRED:
        print(f"  [경고] {os.path.basename(record_path)}: fs={record.fs} (기대=500) — 스킵")
        return None

    sig = record.p_signal   # (T, 12), mV
    if sig is None:
        print(f"  [경고] {os.path.basename(record_path)}: p_signal None — 스킵")
        return None

    T, C = sig.shape
    if C != N_LEADS:
        print(f"  [경고] {os.path.basename(record_path)}: {C}리드 (기대=12) — 스킵")
        return None

    # 길이 통일 (앞 5000샘플 / 부족 시 우측 zero-pad)
    if T >= N_SAMPLES:
        sig = sig[:N_SAMPLES, :]
    else:
        pad = np.zeros((N_SAMPLES - T, C), dtype=np.float32)
        sig = np.concatenate([sig, pad], axis=0)

    sig = sig.T.astype(np.float32)   # (12, 5000)

    if np.isnan(sig).any() or np.isinf(sig).any():
        print(f"  [경고] {os.path.basename(record_path)}: NaN/Inf — 스킵")
        return None

    return sig


def split_records(record_ids, labels_dict, seed=42, train_ratio=0.70, val_ratio=0.15):
    """record-level random split."""
    rng = np.random.RandomState(seed)
    ids = np.array(sorted(record_ids))
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    return (ids[:n_train],
            ids[n_train:n_train + n_val],
            ids[n_train + n_val:])


def save_split(out_dir, split_name, signals, labels, rec_ids):
    split_dir = os.path.join(out_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)

    sig_arr = np.stack(signals).astype(np.float32)
    lab_arr = np.array(labels, dtype=np.int8)
    rec_arr = np.array(rec_ids, dtype=object)

    np.save(os.path.join(split_dir, "signals.npy"),    sig_arr)
    np.save(os.path.join(split_dir, "labels.npy"),     lab_arr)
    np.save(os.path.join(split_dir, "record_ids.npy"), rec_arr)

    n_bad  = int((lab_arr == 1).sum())
    n_good = int((lab_arr == 0).sum())
    print(f"  {split_name:5s}: {len(lab_arr):4d} records "
          f"| 불량(1)={n_bad} ({100*n_bad/len(lab_arr):.1f}%) "
          f"| 양호(0)={n_good} ({100*n_good/len(lab_arr):.1f}%)")
    print(f"         → {split_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_dir",
        default="D:/WidU_ecg-fm_emergency-detection/data/raw/physionet2011/set-a",
        help="set-a 폴더 경로 (RECORDS-* 파일과 .dat/.hea 파일 위치)",
    )
    parser.add_argument(
        "--out_dir",
        default="D:/WidU_ecg-fm_emergency-detection/data/processed/physionet2011",
        help="전처리 결과 저장 폴더",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 65)
    print("단계 7 전처리: PhysioNet 2011 신호품질 게이트 데이터")
    print("=" * 65)
    print(f"입력: {args.data_dir}")
    print(f"출력: {args.out_dir}")
    print()

    # ── 1. 레코드 목록 읽기 ──────────────────────────────────────────
    print("[1] 레코드 목록 로드")
    acc_path   = os.path.join(args.data_dir, "RECORDS-acceptable")
    unacc_path = os.path.join(args.data_dir, "RECORDS-unacceptable")

    acc_ids   = read_record_list(acc_path)
    unacc_ids = read_record_list(unacc_path)

    if not acc_ids and not unacc_ids:
        sys.exit(f"[오류] RECORDS-acceptable / RECORDS-unacceptable 파일 없음: {args.data_dir}")

    print(f"  acceptable   (label=0): {len(acc_ids)}개")
    print(f"  unacceptable (label=1): {len(unacc_ids)}개")
    print(f"  합계: {len(acc_ids)+len(unacc_ids)}개")
    print()

    # label dict
    label_map = {}
    for rid in acc_ids:
        label_map[rid] = 0   # 양호
    for rid in unacc_ids:
        label_map[rid] = 1   # 불량 (positive, 게이트 탐지 대상)

    all_ids = list(label_map.keys())

    # ── 2. Split ─────────────────────────────────────────────────────
    print("[2] record-level random split (seed=42, 70/15/15)")
    train_ids, val_ids, test_ids = split_records(all_ids, label_map, seed=args.seed)
    print(f"  학습: {len(train_ids)}, 검증: {len(val_ids)}, 테스트: {len(test_ids)}")
    print()

    # ── 3. 신호 읽기 + 저장 ──────────────────────────────────────────
    print("[3] wfdb 신호 읽기 및 저장")
    print("-" * 65)

    skip_total = 0
    for split_name, ids in [("train", train_ids),
                             ("val",   val_ids),
                             ("test",  test_ids)]:
        signals, labels, rec_ids = [], [], []
        skip_count = 0

        for rid in ids:
            rec_path = os.path.join(args.data_dir, rid)
            sig = load_signal(rec_path)
            if sig is None:
                skip_count += 1
                skip_total += 1
                continue
            signals.append(sig)
            labels.append(label_map[rid])
            rec_ids.append(rid)

        if not signals:
            print(f"  [경고] {split_name}: 유효 레코드 0개 — 저장 스킵")
            continue
        if skip_count:
            print(f"  [참고] {split_name}: {skip_count}건 스킵")

        save_split(args.out_dir, split_name, signals, labels, rec_ids)

    # ── 4. 요약 ──────────────────────────────────────────────────────
    print()
    print("=" * 65)
    print("전처리 완료")
    if skip_total:
        print(f"  전체 스킵: {skip_total}건")
    print()
    print("다음 단계:")
    print("  → 단계 7 훈련: python scripts/train_gate.py")
    print("=" * 65)


if __name__ == "__main__":
    main()
