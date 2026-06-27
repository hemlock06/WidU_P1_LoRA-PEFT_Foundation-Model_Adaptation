"""
단계 5e: CPSC 2018 다중분류 전처리 (5-class, multi-label)
========================================================
★2026-05-29 개정 (구 5b 단일라벨 → multi-label):
  - 근본원인 분석(records/01 §다중분류 약클래스)에 따라 두 데이터 결함 수정:
    (1) 164884008(ventricular ectopics) 603개 레코드 배제 → 이소성으로 복구
    (2) 단일라벨 우선순위 붕괴로 동반코드 이소성 라벨 박탈 → multi-hot 보존
  - 기존 단일라벨 데이터(cpsc2018_mc)는 보존, 본 스크립트는 신규 디렉터리 출력.

목적:
  CPSC 2018 9 클래스를 5개 심장 taxonomy multi-hot으로 매핑하여
  ECG-FM 입력 형식 (float32, 12ch, 5000샘플) 으로 저장한다.

라벨 매핑 (5-class, multi-hot — 동반 진단 독립 보존):
  0 = 정상 (NSR)     : 426783006
  1 = AF             : 164889003
  2 = 급성허혈 ST변화 : STD(429622005) + STE(164931005)
  3 = 전도장애        : I-AVB(270492004) + LBBB(164909002) + RBBB(59118001)
  4 = 이소성 박동     : PAC(284470004/63593006) + PVC(427172004/17338001)
                       + 심실이소성(164884008)  ← 신규 복구
  → 한 레코드가 여러 클래스 코드 보유 시 해당 비트 모두 1.
  → 인식 코드 전무 시 제외.

Split:
  record-level random 70/15/15, seed=42

입력 규격: 500 Hz, 12-lead, 10초(5000샘플) — ECG-FM 입력 호환

출력 (--out_dir/{train,val,test}/):
  signals.npy   : float32 (N, 12, 5000)
  labels_mc.npy : int8    (N, 5)  ★ multi-hot — 학습용
  labels.npy    : int8    (N,)    우선순위 단일라벨 — 혼동행렬/호환용
  labels_bin.npy: int8    (N,)    파생 이진 (1=AF 또는 허혈 존재)
  record_ids.npy: str array (N,)

사용법:
  python scripts/preprocess_cpsc2018_mc.py \\
      --data_dir data/raw/cpsc2018 \\
      --out_dir  data/processed/cpsc2018_mc_ml
"""

import argparse
import os
import sys

import numpy as np

FS_REQUIRED = 500
N_LEADS = 12
N_SAMPLES = 5000

# ── SNOMED-CT 코드 → 다중분류 매핑 ───────────────────────────────────
SNOMED_NORMAL = {426783006}
SNOMED_AF = {164889003}
SNOMED_ISCH = {429622005, 164931005}  # STD + STE
SNOMED_COND = {270492004, 164909002, 59118001}  # I-AVB + LBBB + RBBB
SNOMED_ECTO = {
    284470004,
    63593006,  # PAC + SVPB
    427172004,
    17338001,  # PVC + VPB
    164884008,
}  # 심실 이소성(ventricular ectopics)
# ★2026-05-29 추가: raw 603개 배제분 복구

ALL_KNOWN = SNOMED_NORMAL | SNOMED_AF | SNOMED_ISCH | SNOMED_COND | SNOMED_ECTO

CLASS_NAMES = {
    0: "정상(NSR)",
    1: "AF",
    2: "급성허혈(STD/STE)",
    3: "전도장애(I-AVB/LBBB/RBBB)",
    4: "이소성(PAC/PVC)",
}


def parse_dx_from_hea(hea_path: str) -> set:
    codes = set()
    try:
        with open(hea_path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") and "Dx" in line and ":" in line:
                    _, _, rest = line.partition(":")
                    for token in rest.split(","):
                        token = token.strip()
                        if token.isdigit():
                            codes.add(int(token))
    except Exception:
        pass
    return codes


_CLASS_CODESETS = [SNOMED_NORMAL, SNOMED_AF, SNOMED_ISCH, SNOMED_COND, SNOMED_ECTO]


def map_label_multihot(snomed_codes: set) -> np.ndarray:
    """
    multi-label 매핑: 한 레코드가 여러 클래스 코드를 가지면 모두 1로 표시.
    반환: shape (5,) int8 멀티핫 [NSR, AF, ISCH, COND, ECTO].
    인식 가능한 코드 하나도 없으면 전부 0 (호출부에서 제외 판정).

    ★2026-05-29 변경: 기존 단일라벨 우선순위 붕괴(map_label_mc)가 동반코드의
      이소성 라벨을 박탈하던 문제 해결 — 동반 진단을 독립 보존.
    """
    vec = np.zeros(5, dtype=np.int8)
    for k, codeset in enumerate(_CLASS_CODESETS):
        if snomed_codes & codeset:
            vec[k] = 1
    return vec


def map_label_mc(snomed_codes: set) -> int:
    """
    [참고/하위호환용] 중증도 우선순위 단일라벨: 2 > 1 > 3 > 4 > 0.
    인식 가능한 코드 하나도 없으면 -1 (제외).
    혼동행렬 등 단일라벨 리포트에만 사용. 학습은 multi-hot 사용.
    """
    if snomed_codes & SNOMED_ISCH:
        return 2
    if snomed_codes & SNOMED_AF:
        return 1
    if snomed_codes & SNOMED_COND:
        return 3
    if snomed_codes & SNOMED_ECTO:
        return 4
    if snomed_codes & SNOMED_NORMAL:
        return 0
    return -1  # 미인식 코드만 있는 레코드 → 제외


def scan_records(data_dir: str) -> dict:
    records = {}
    for root, _, files in os.walk(data_dir):
        for fname in files:
            if fname.endswith(".hea"):
                stem = fname[:-4]
                records[stem] = os.path.join(root, fname)
    return records


def load_signal(hea_path: str):
    try:
        import wfdb
    except ImportError:
        sys.exit("[오류] wfdb 미설치 — pip install wfdb")

    rec_path = hea_path[:-4]
    try:
        record = wfdb.rdrecord(rec_path)
    except Exception as e:
        print(f"  [경고] {os.path.basename(rec_path)} 읽기 실패: {e}")
        return None

    if record.fs != FS_REQUIRED:
        return None

    sig = record.p_signal
    if sig is None:
        return None

    T, C = sig.shape
    if C != N_LEADS:
        return None

    if T >= N_SAMPLES:
        sig = sig[:N_SAMPLES, :]
    else:
        pad = np.zeros((N_SAMPLES - T, C), dtype=np.float32)
        sig = np.concatenate([sig, pad], axis=0)

    sig = sig.T.astype(np.float32)

    if np.isnan(sig).any() or np.isinf(sig).any():
        return None

    return sig


def split_records(record_ids, seed=42, train_ratio=0.70, val_ratio=0.15):
    rng = np.random.RandomState(seed)
    ids = np.array(sorted(record_ids))
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return ids[:n_train], ids[n_train : n_train + n_val], ids[n_train + n_val :]


def save_split(out_dir: str, split: str, signals, multihots, singles, record_ids):
    split_dir = os.path.join(out_dir, split)
    os.makedirs(split_dir, exist_ok=True)

    sig_arr = np.stack(signals).astype(np.float32)
    mc_arr = np.stack(multihots).astype(np.int8)  # (N, 5) multi-hot — 학습용
    lab_arr = np.array(singles, dtype=np.int8)  # (N,) 우선순위 단일 — 리포트/호환용
    rec_arr = np.array(record_ids, dtype=object)
    # 파생 이진: AF(idx1) 또는 급성허혈(idx2) 존재 → 응급(1)
    bin_arr = ((mc_arr[:, 1] | mc_arr[:, 2]) > 0).astype(np.int8)

    np.save(os.path.join(split_dir, "signals.npy"), sig_arr)
    np.save(os.path.join(split_dir, "labels_mc.npy"), mc_arr)  # ★ 신규: multi-hot
    np.save(os.path.join(split_dir, "labels.npy"), lab_arr)  # 단일라벨(호환)
    np.save(os.path.join(split_dir, "labels_bin.npy"), bin_arr)
    np.save(os.path.join(split_dir, "record_ids.npy"), rec_arr)

    n = len(lab_arr)
    print(f"  {split:5s}: {n:4d} records")
    print("    [multi-hot 클래스별 양성 수 (중복 허용)]")
    for k in range(5):
        n_k = int(mc_arr[:, k].sum())
        print(f"    [{k}] {CLASS_NAMES[k]:30s}: {n_k:4d} ({100 * n_k / n:.1f}%)")
    n_multi = int((mc_arr.sum(axis=1) > 1).sum())
    n_emg = int(bin_arr.sum())
    print(f"    └ 다중라벨 레코드(≥2 클래스): {n_multi} ({100 * n_multi / n:.1f}%)")
    print(f"    └ 파생이진 응급(AF/허혈): {n_emg} ({100 * n_emg / n:.1f}%)")
    print(f"         → {split_dir}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data_dir", default="data/raw/cpsc2018")
    parser.add_argument("--out_dir", default="data/processed/cpsc2018_mc_ml")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 70)
    print("단계 5b: CPSC 2018 다중분류 전처리 (5-class)")
    print("=" * 70)
    print(f"입력: {args.data_dir}")
    print(f"출력: {args.out_dir}")
    print()

    print("[1] .hea 파일 스캔")
    records = scan_records(args.data_dir)
    if not records:
        sys.exit(f"[오류] .hea 파일 없음: {args.data_dir}")
    print(f"  발견: {len(records)}개 레코드")
    print()

    print("[2] .hea '#Dx:' 파싱 → 5-class multi-hot 매핑")
    mc_map = {}  # rec_id → multihot (5,)
    single_map = {}  # rec_id → 우선순위 단일라벨 (호환용)
    label_dist = {i: 0 for i in range(5)}
    n_excl = 0

    for rec_id, hea_path in sorted(records.items()):
        codes = parse_dx_from_hea(hea_path)
        vec = map_label_multihot(codes)
        if vec.sum() == 0:  # 인식 코드 전무 → 제외
            n_excl += 1
            continue
        mc_map[rec_id] = vec
        single_map[rec_id] = map_label_mc(codes)
        for k in range(5):
            label_dist[k] += int(vec[k])

    total = len(mc_map)
    pos_total = sum(label_dist.values())
    print(f"  유효 레코드: {total}건 (multi-hot 양성 총합 {pos_total})")
    for k in range(5):
        print(
            f"  [{k}] {CLASS_NAMES[k]:30s}: {label_dist[k]:4d} "
            f"({100 * label_dist[k] / total:.1f}% of records)"
        )
    print(f"  제외(미인식 코드만): {n_excl}건")
    print()

    print("[3] record-level random split (seed=42, 70/15/15)")
    valid_ids = list(mc_map.keys())
    train_ids, val_ids, test_ids = split_records(valid_ids, seed=args.seed)
    print(f"  학습: {len(train_ids)}, 검증: {len(val_ids)}, 테스트: {len(test_ids)}")
    print()

    print("[4] WFDB 신호 읽기 및 저장")
    print("-" * 70)

    skip_total = 0
    for split_name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        signals, multihots, singles, rec_ids = [], [], [], []
        skip_count = 0

        for rec_id in ids:
            sig = load_signal(records[rec_id])
            if sig is None:
                skip_count += 1
                skip_total += 1
                continue
            signals.append(sig)
            multihots.append(mc_map[rec_id])
            singles.append(single_map[rec_id])
            rec_ids.append(rec_id)

        if not signals:
            print(f"  [경고] {split_name}: 유효 레코드 0개")
            continue
        if skip_count:
            print(f"  [참고] {split_name}: {skip_count}건 스킵")

        save_split(args.out_dir, split_name, signals, multihots, singles, rec_ids)

    print()
    print("=" * 70)
    print("전처리 완료")
    if skip_total:
        print(f"  전체 스킵: {skip_total}건")
    print()
    print("다음 단계:")
    print("  → python scripts/train_lora_multiclass.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
