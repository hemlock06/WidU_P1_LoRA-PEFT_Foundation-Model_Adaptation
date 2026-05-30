"""
PTB-XL 전처리 (preprocess_ptbxl.py)
======================================
목적:
  PTB-XL 1.0.3 (21,837건, 12-lead 500Hz)을 ECG-FM 입력 형식으로 변환.
  이진 응급 파일럿용 — CPSC와 동일한 (N, 12, 5000) float32 포맷.

라벨 매핑 (superclass 기준, ptbxl_database.csv의 superdiagnostic 컬럼):
  응급(1): MI (심근경색), STTC (ST-T change)
  정상(0): NORM (정상 동율동)
  제외:    CD (전도장애), HYP (비대), 다중 superclass 혼재 시 MI/STTC 우선

Split: patient-level (ecg_id → patient_id 기준), seed=42, 70/15/15
  → CPSC와 달리 PTB-XL은 patient_id 제공 → patient-level split 필수 (누출 방지)

입력 규격:
  - 500Hz, 12-lead, 10초(5000샘플)
  - normalize=False (ECG-FM 사전학습 관례)
  - 짧으면 우측 zero-pad, 길면 앞 5000샘플

출력 (--out_dir/{train,val,test}/):
  signals.npy   : float32 (N, 12, 5000)
  labels.npy    : int8    (N,)
  record_ids.npy: str array (N,)

사용법:
  python scripts/preprocess_ptbxl.py
  python scripts/preprocess_ptbxl.py \\
      --data_dir D:/WidU_ecg-fm_emergency-detection/data/raw/ptbxl \\
      --out_dir  D:/WidU_ecg-fm_emergency-detection/data/processed/ptbxl
"""

import argparse
import ast
import os
import sys
import json

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    import pandas as pd
    import wfdb
except ImportError as e:
    sys.exit(f"[오류] {e} — pip install pandas wfdb")

FS_REQUIRED = 500
N_LEADS     = 12
N_SAMPLES   = 5000  # 10s × 500Hz
SEED        = 42

# PTB-XL 12-lead 순서: I II III aVR aVL aVF V1 V2 V3 V4 V5 V6
# wfdb 로드 시 이 순서 그대로 나옴 — 재배치 불필요

DATA_DEFAULT  = "D:/WidU_ecg-fm_emergency-detection/data/raw/ptbxl"
OUT_DEFAULT   = "D:/WidU_ecg-fm_emergency-detection/data/processed/ptbxl"


def parse_superclass(scp_str):
    """
    ptbxl_database.csv scp_codes 컬럼 파싱.
    예: "{'NDT': 0, 'NSR': 100.0}"  → dict
    superclass 매핑은 scp_statements.csv의 'diagnostic_class' 컬럼 사용.
    반환: set of superclass strings (e.g. {'NORM'}, {'MI', 'CD'})
    """
    try:
        codes = ast.literal_eval(scp_str)
        return codes  # {code: likelihood}
    except Exception:
        return {}


def map_label(scp_codes, scp_df):
    """
    scp_codes: dict {code_str: likelihood}
    반환: 1 (응급), 0 (정상), -1 (제외)
    우선순위: MI > STTC > NORM > CD/HYP → 제외
    """
    super_classes = set()
    for code, lkl in scp_codes.items():
        if lkl == 0:  # likelihood=0은 사용 안 함
            continue
        if code in scp_df.index:
            sc = scp_df.loc[code, "diagnostic_class"]
            if pd.notna(sc) and sc != "":
                super_classes.add(sc)

    if "MI" in super_classes or "STTC" in super_classes:
        return 1   # 응급
    elif "NORM" in super_classes and len(super_classes) == 1:
        return 0   # 순수 정상만
    elif "NORM" in super_classes:
        # NORM + 다른 진단 → 제외 (혼재)
        return -1
    else:
        return -1  # CD, HYP, 알 수 없음 → 제외


def load_record(rec_path, data_dir):
    """wfdb 레코드 로드 → (12, 5000) float32"""
    full_path = os.path.join(data_dir, rec_path)
    try:
        record = wfdb.rdrecord(full_path)
    except Exception as e:
        return None, f"읽기 실패: {e}"

    sig = record.p_signal   # (T, n_leads) float64
    if sig is None:
        return None, "신호 없음"

    # NaN/Inf 체크
    if not np.isfinite(sig).all():
        return None, "NaN/Inf 포함"

    # lead 수 확인
    if sig.shape[1] != N_LEADS:
        return None, f"lead 수 불일치 ({sig.shape[1]})"

    # (T, 12) → (12, T) 변환
    sig = sig.T.astype(np.float32)

    # 길이 조정
    T = sig.shape[1]
    if T >= N_SAMPLES:
        sig = sig[:, :N_SAMPLES]
    else:
        pad = np.zeros((N_LEADS, N_SAMPLES - T), dtype=np.float32)
        sig = np.concatenate([sig, pad], axis=1)

    return sig, None


def patient_split(df, seed=SEED):
    """patient_id 기준 train/val/test 70/15/15 분리"""
    rng = np.random.default_rng(seed)
    patients = df["patient_id"].unique()
    rng.shuffle(patients)
    n = len(patients)
    n_val  = int(n * 0.15)
    n_test = int(n * 0.15)
    val_p   = set(patients[:n_val])
    test_p  = set(patients[n_val:n_val + n_test])
    train_p = set(patients[n_val + n_test:])

    splits = {}
    splits["train"] = df[df["patient_id"].isin(train_p)]
    splits["val"]   = df[df["patient_id"].isin(val_p)]
    splits["test"]  = df[df["patient_id"].isin(test_p)]
    return splits


def process_split(split_df, data_dir, out_dir, split_name, scp_df):
    signals, labels, rec_ids = [], [], []
    skip = {"label_exc": 0, "read_err": 0, "ok": 0}

    for _, row in split_df.iterrows():
        scp_codes = parse_superclass(row["scp_codes"])
        lbl = map_label(scp_codes, scp_df)
        if lbl == -1:
            skip["label_exc"] += 1
            continue

        # PTB-XL 레코드 경로: filename_hr (500Hz) 컬럼
        rec_path = row["filename_hr"]
        sig, err = load_record(rec_path, data_dir)
        if sig is None:
            skip["read_err"] += 1
            continue

        signals.append(sig)
        labels.append(lbl)
        rec_ids.append(str(row["ecg_id"]))
        skip["ok"] += 1

    if not signals:
        print(f"  [{split_name}] 유효 샘플 없음 — 건너뜀")
        return

    sig_arr = np.stack(signals, axis=0).astype(np.float32)
    lbl_arr = np.array(labels, dtype=np.int8)
    id_arr  = np.array(rec_ids)

    split_out = os.path.join(out_dir, split_name)
    os.makedirs(split_out, exist_ok=True)
    np.save(os.path.join(split_out, "signals.npy"), sig_arr)
    np.save(os.path.join(split_out, "labels.npy"), lbl_arr)
    np.save(os.path.join(split_out, "record_ids.npy"), id_arr)

    n_pos = int((lbl_arr == 1).sum())
    n_neg = int((lbl_arr == 0).sum())
    print(f"  [{split_name}] 총 {len(sig_arr)}  응급(MI/STTC)={n_pos}  정상(NORM)={n_neg}")
    print(f"           제외(CD/HYP/혼재)={skip['label_exc']}  읽기 실패={skip['read_err']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", default=DATA_DEFAULT)
    parser.add_argument("--out_dir",  default=OUT_DEFAULT)
    args = parser.parse_args()

    print("=" * 65)
    print("PTB-XL 전처리 — 이진 응급 파일럿용")
    print("=" * 65)
    print(f"입력: {args.data_dir}")
    print(f"출력: {args.out_dir}")
    print()

    # ── CSV 로드 ──────────────────────────────────────────────────────────
    db_csv = os.path.join(args.data_dir, "ptbxl_database.csv")
    scp_csv = os.path.join(args.data_dir, "scp_statements.csv")
    if not os.path.exists(db_csv):
        sys.exit(f"[오류] {db_csv} 없음 — 다운로드 확인")
    if not os.path.exists(scp_csv):
        sys.exit(f"[오류] {scp_csv} 없음")

    df  = pd.read_csv(db_csv, index_col="ecg_id")
    df  = df.reset_index()   # ecg_id 컬럼으로
    scp = pd.read_csv(scp_csv, index_col=0)

    print(f"PTB-XL 총 레코드: {len(df):,}건")

    # superclass 분포 확인
    superclass_counts = {}
    for _, row in df.iterrows():
        codes = parse_superclass(row["scp_codes"])
        lbl = map_label(codes, scp)
        tag = "응급(MI/STTC)" if lbl == 1 else ("정상(NORM)" if lbl == 0 else "제외")
        superclass_counts[tag] = superclass_counts.get(tag, 0) + 1
    print("라벨 분포 (전체):")
    for k, v in sorted(superclass_counts.items()):
        print(f"  {k}: {v:,}")
    print()

    # ── patient-level split ───────────────────────────────────────────────
    splits = patient_split(df, seed=SEED)
    print("Patient-level split (seed=42):")
    for sp, sdf in splits.items():
        print(f"  {sp}: {len(sdf['patient_id'].unique())} 환자, {len(sdf)} 레코드")
    print()

    # ── 전처리 ───────────────────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    for split_name, split_df in splits.items():
        process_split(split_df, args.data_dir, args.out_dir, split_name, scp)

    print()
    print("=" * 65)
    print("전처리 완료")
    print("다음 단계: python scripts/train_pilot_ptbxl.py")
    print("=" * 65)


if __name__ == "__main__":
    main()
