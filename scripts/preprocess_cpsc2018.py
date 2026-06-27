"""
단계 4: CPSC 2018 데이터 전처리 (.hea SNOMED-CT 코드 버전)
============================================================
목적:
  CPSC 2018 Challenge 데이터를 읽어 이진 응급 분류 레이블로 변환하고
  ECG-FM 입력 형식 (float32, 12ch, 5000샘플) 으로 저장한다.

라벨 출처:
  REFERENCE.csv 대신 각 .hea 파일의 '#Dx:' 라인에서 SNOMED-CT 코드 직접 파싱.
  (PhysioNet Challenge 2020 cpsc_2018 서브셋 표준 포맷)

라벨 매핑 (옵션 A — 엄격 이진):
  응급(1):  AF(164889003), STD(429622005), STE(164931005)
  정상(0):  Normal(426783006)
  제외(-1): I-AVB(270492004), LBBB(164909002), RBBB(59118001),
            PAC(284470004/63593006), PVC(427172004/17338001)
  다중 라벨: 하나라도 응급 → 응급. 모두 정상 → 정상. 제외만 → 제외.

Split:
  record-level random 70/15/15, seed=42
  (CPSC 2018은 patient ID 미제공 → record-level split 사용. decisions.md 참조)

입력 규격:
  - 500 Hz, 12-lead, 10초(5000샘플)
  - normalize=False (cfg['model']['normalize']=False 확인됨)

출력 (--out_dir/{train,val,test}/):
  signals.npy   : float32 (N, 12, 5000)
  labels.npy    : int8    (N,)
  record_ids.npy: str array (N,)

사용법:
  python scripts/preprocess_cpsc2018.py \\
      --data_dir data/raw/cpsc2018 \\
      --out_dir  data/processed/cpsc2018
"""

import argparse
import os
import sys

import numpy as np

FS_REQUIRED = 500
N_LEADS = 12
N_SAMPLES = 5000  # 10s × 500Hz

# ── SNOMED-CT 코드 → 클래스 매핑 ─────────────────────────────────────
# PhysioNet Challenge 2020 cpsc_2018 서브셋 기준
SNOMED_EMERGENCY = {
    164889003,  # Atrial fibrillation (AF)
    429622005,  # ST-segment depression (STD)
    164931005,  # ST-segment elevation (STE)
}
SNOMED_NORMAL = {
    426783006,  # Normal sinus rhythm
}
SNOMED_EXCLUDE = {
    270492004,  # First-degree AV block (I-AVB)
    164909002,  # Left bundle branch block (LBBB)
    59118001,  # Right bundle branch block (RBBB)
    284470004,  # Premature atrial contraction (PAC)
    63593006,  # PAC 대체 코드
    427172004,  # Premature ventricular contraction (PVC)
    17338001,  # PVC 대체 코드
}

# 디버그 출력용 이름
SNOMED_NAMES = {
    164889003: "AF",
    429622005: "STD",
    164931005: "STE",
    426783006: "Normal",
    270492004: "I-AVB",
    164909002: "LBBB",
    59118001: "RBBB",
    284470004: "PAC",
    63593006: "PAC(alt)",
    427172004: "PVC",
    17338001: "PVC(alt)",
}


def parse_dx_from_hea(hea_path: str) -> set:
    """
    .hea 파일에서 '#Dx:' 라인을 찾아 SNOMED-CT 코드 집합 반환.
    지원 포맷:
      # Dx: 426783006
      #Dx: 164889003,270492004
      # Dx: 164889003, 270492004
    코드가 없으면 빈 set 반환.
    """
    codes = set()
    try:
        with open(hea_path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                # '#Dx:' 또는 '# Dx:' 모두 처리
                if line.startswith("#") and "Dx" in line and ":" in line:
                    _, _, rest = line.partition(":")
                    for token in rest.split(","):
                        token = token.strip()
                        if token.isdigit():
                            codes.add(int(token))
    except Exception:
        pass
    return codes


def map_label_option_a(snomed_codes: set) -> int:
    """
    옵션 A 이진 매핑.
    반환: 1(응급), 0(정상), -1(제외 또는 미인식)
    """
    if snomed_codes & SNOMED_EMERGENCY:
        return 1
    if snomed_codes & SNOMED_NORMAL and not (snomed_codes & SNOMED_EXCLUDE):
        return 0
    return -1


def scan_records(data_dir: str) -> dict:
    """
    data_dir (및 하위 폴더) 에서 .hea 파일을 모두 찾아
    {rec_stem: hea_path} 딕셔너리 반환.
    rec_stem = 파일명에서 확장자 제거 (예: 'A0001').
    """
    records = {}
    for root, _, files in os.walk(data_dir):
        for fname in files:
            if fname.endswith(".hea"):
                stem = fname[:-4]
                records[stem] = os.path.join(root, fname)
    return records


def load_signal(hea_path: str) -> "np.ndarray | None":
    """
    WFDB 형식(.hea + .mat/.dat) 레코드 읽기.
    반환: float32 (12, 5000) 또는 None (오류 시)
    """
    try:
        import wfdb
    except ImportError:
        sys.exit("[오류] wfdb 미설치 — pip install wfdb")

    # wfdb.rdrecord는 확장자 없는 경로를 받음
    rec_path = hea_path[:-4]
    try:
        record = wfdb.rdrecord(rec_path)
    except Exception as e:
        print(f"  [경고] {os.path.basename(rec_path)} 읽기 실패: {e}")
        return None

    if record.fs != FS_REQUIRED:
        print(
            f"  [경고] {os.path.basename(rec_path)}: fs={record.fs} (기대={FS_REQUIRED}) — 스킵"
        )
        return None

    sig = record.p_signal  # (T, n_leads), mV
    if sig is None:
        print(f"  [경고] {os.path.basename(rec_path)}: p_signal None — 스킵")
        return None

    T, C = sig.shape
    if C != N_LEADS:
        print(f"  [경고] {os.path.basename(rec_path)}: {C}리드 (기대=12) — 스킵")
        return None

    # 길이 통일 (앞 5000샘플 사용 / 부족하면 우측 zero-pad)
    if T >= N_SAMPLES:
        sig = sig[:N_SAMPLES, :]
    else:
        pad = np.zeros((N_SAMPLES - T, C), dtype=np.float32)
        sig = np.concatenate([sig, pad], axis=0)

    sig = sig.T.astype(np.float32)  # (12, 5000)

    if np.isnan(sig).any() or np.isinf(sig).any():
        print(f"  [경고] {os.path.basename(rec_path)}: NaN/Inf — 스킵")
        return None

    return sig


def split_records(record_ids, seed=42, train_ratio=0.70, val_ratio=0.15):
    """record-level random split."""
    rng = np.random.RandomState(seed)
    ids = np.array(sorted(record_ids))
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return ids[:n_train], ids[n_train : n_train + n_val], ids[n_train + n_val :]


def save_split(out_dir: str, split: str, signals, labels, record_ids):
    split_dir = os.path.join(out_dir, split)
    os.makedirs(split_dir, exist_ok=True)

    sig_arr = np.stack(signals).astype(np.float32)
    lab_arr = np.array(labels, dtype=np.int8)
    rec_arr = np.array(record_ids, dtype=object)

    np.save(os.path.join(split_dir, "signals.npy"), sig_arr)
    np.save(os.path.join(split_dir, "labels.npy"), lab_arr)
    np.save(os.path.join(split_dir, "record_ids.npy"), rec_arr)

    n_emg = int((lab_arr == 1).sum())
    n_norm = int((lab_arr == 0).sum())
    print(
        f"  {split:5s}: {len(lab_arr):4d} records "
        f"| 응급={n_emg} ({100 * n_emg / len(lab_arr):.1f}%) "
        f"| 정상={n_norm} ({100 * n_norm / len(lab_arr):.1f}%)"
    )
    print(f"         → {split_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_dir",
        default="data/raw/cpsc2018",
        help=".hea/.mat 파일이 있는 폴더",
    )
    parser.add_argument(
        "--out_dir",
        default="data/processed/cpsc2018",
        help="전처리 결과 저장 폴더",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--show_unknown", action="store_true", help="미인식 SNOMED 코드 목록 출력"
    )
    args = parser.parse_args()

    print("=" * 65)
    print("단계 4: CPSC 2018 전처리 (.hea SNOMED-CT 코드 버전)")
    print("=" * 65)
    print(f"입력: {args.data_dir}")
    print(f"출력: {args.out_dir}")
    print()

    # ── 1. .hea 파일 스캔 ─────────────────────────────────────────────
    print("[1] .hea 파일 스캔")
    records = scan_records(args.data_dir)
    if not records:
        sys.exit(f"[오류] .hea 파일 없음: {args.data_dir}")
    print(f"  발견: {len(records)}개 레코드")
    print()

    # ── 2. SNOMED-CT 파싱 + 옵션 A 매핑 ──────────────────────────────
    print("[2] .hea '#Dx:' 파싱 → 옵션 A 이진 매핑")
    label_map = {}
    unknown_map = {}  # rec_id → 미인식 코드
    label_dist = {1: 0, 0: 0, -1: 0}

    all_known = SNOMED_EMERGENCY | SNOMED_NORMAL | SNOMED_EXCLUDE

    for rec_id, hea_path in sorted(records.items()):
        codes = parse_dx_from_hea(hea_path)
        lbl = map_label_option_a(codes)
        label_map[rec_id] = lbl
        label_dist[lbl] += 1

        unknown = codes - all_known
        if unknown:
            unknown_map[rec_id] = unknown

    print(f"  응급(1):  {label_dist[1]:4d} 건 — AF/STD/STE")
    print(f"  정상(0):  {label_dist[0]:4d} 건 — Normal")
    print(f"  제외(-1): {label_dist[-1]:4d} 건 — I-AVB/LBBB/RBBB/PAC/PVC + 미인식")

    if unknown_map:
        all_unknown_codes = set()
        for v in unknown_map.values():
            all_unknown_codes |= v
        print(
            f"  [참고] 미인식 SNOMED 코드 {len(all_unknown_codes)}종 "
            f"({len(unknown_map)}개 레코드) — 제외 처리됨"
        )
        if args.show_unknown:
            for code in sorted(all_unknown_codes):
                print(f"    {code}")
    print()

    # ── 3. Split ──────────────────────────────────────────────────────
    print("[3] record-level random split (seed=42, 70/15/15)")
    valid_ids = [r for r, l in label_map.items() if l != -1]
    train_ids, val_ids, test_ids = split_records(valid_ids, seed=args.seed)
    print(f"  학습: {len(train_ids)}, 검증: {len(val_ids)}, 테스트: {len(test_ids)}")
    print()

    # ── 4. 신호 읽기 + 저장 ───────────────────────────────────────────
    print("[4] WFDB 신호 읽기 및 저장")
    print("-" * 65)

    skip_total = 0
    for split_name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        signals, labels, rec_ids = [], [], []
        skip_count = 0

        for rec_id in ids:
            sig = load_signal(records[rec_id])
            if sig is None:
                skip_count += 1
                skip_total += 1
                continue
            signals.append(sig)
            labels.append(label_map[rec_id])
            rec_ids.append(rec_id)

        if not signals:
            print(f"  [경고] {split_name}: 유효 레코드 0개 — 저장 스킵")
            continue
        if skip_count:
            print(f"  [참고] {split_name}: {skip_count}건 스킵")

        save_split(args.out_dir, split_name, signals, labels, rec_ids)

    # ── 5. 요약 ───────────────────────────────────────────────────────
    print()
    print("=" * 65)
    print("전처리 완료")
    if skip_total:
        print(f"  전체 스킵: {skip_total}건")
    print()
    print("다음 단계:")
    print("  → 단계 5: 베이스라인 선형 프로빙 (scripts/train_baseline.py)")
    print("=" * 65)


if __name__ == "__main__":
    main()
