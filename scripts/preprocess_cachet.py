"""
단계 9 전처리: CACHET-CADB 외부검증 데이터 준비
================================================
목적:
  웨어러블 ECG (CACHET-CADB Short Format) → 단계 9 추론용.
  AF = 응급(1), NSR = 정상(0), Noise/Others = 제외.

데이터 구조:
  HDF5 키:  signal (16,404,480,) @1024Hz + labels (16,404,480,)
  라벨 값:  1=AF, 2=NSR, 3=Noise, 4=Others
  세그먼트: 1,602개 × 10,240 샘플 (10s × 1024Hz)
  채널:     단일 채널 (wearable single-lead)

처리 과정:
  1. HDF5 로드 → (N, 10240) reshape
  2. per-segment 라벨: 세그먼트 첫 샘플 (동일 라벨 보장)
  3. scipy.signal.resample: 10,240 → 5,000 (1024Hz → 500Hz)
  4. 12-lead 0-fill: 단일 채널 → slot 1 (lead II)
  5. AF→1, NSR→0, 나머지 제외

출력:
  --out_dir/signals.npy  : float32 (N, 12, 5000)
  --out_dir/labels.npy   : int8    (N,) 0=정상, 1=응급

사용법:
  python scripts/preprocess_cachet.py
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
    import h5py
except ImportError:
    sys.exit("[오류] h5py 미설치 — pip install h5py")

try:
    from scipy.signal import resample as sp_resample
except ImportError:
    sys.exit("[오류] scipy 미설치 — pip install scipy")

FS_IN       = 1024
FS_OUT      = 500
N_LEADS     = 12
SEG_LEN_IN  = FS_IN  * 10   # 10,240
SEG_LEN_OUT = FS_OUT * 10   # 5,000
LEAD_SLOT   = 1              # 단일 채널 → lead II 슬롯

# CACHET 라벨 → 이진 응급
LABEL_MAP = {1: 1, 2: 0}    # AF→응급, NSR→정상; 3/4 제외


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hdf5_path",
        default="data/raw/cachet/"
                "cachet-cadb_short_format_without_context.hdf5",
    )
    parser.add_argument(
        "--out_dir",
        default="data/processed/cachet",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 65)
    print("단계 9 전처리: CACHET-CADB (웨어러블 AF/NSR)")
    print("=" * 65)
    print(f"입력: {args.hdf5_path}")
    print(f"출력: {args.out_dir}")
    print()

    # ── 1. HDF5 로드 ──────────────────────────────────────────────────
    print("[1] HDF5 로드")
    with h5py.File(args.hdf5_path, "r") as f:
        sig_flat = f["signal"][:]   # (16,404,480,)
        lbl_flat = f["labels"][:]   # (16,404,480,)

    N_total = len(sig_flat) // SEG_LEN_IN
    sig_flat = sig_flat[: N_total * SEG_LEN_IN]
    lbl_flat = lbl_flat[: N_total * SEG_LEN_IN]

    sig_segs = sig_flat.reshape(N_total, SEG_LEN_IN).astype(np.float32)
    lbl_segs = lbl_flat.reshape(N_total, SEG_LEN_IN)

    # per-segment 라벨: 첫 샘플 (세그먼트 내 라벨 균일)
    seg_labels = lbl_segs[:, 0].astype(int)

    from collections import Counter
    cnt = Counter(seg_labels.tolist())
    lbl_names = {1: "AF", 2: "NSR", 3: "Noise", 4: "Others"}
    print(f"  총 세그먼트: {N_total}")
    for k, v in sorted(cnt.items()):
        print(f"    라벨 {k} ({lbl_names.get(k, '?')}): {v}개")
    print()

    # ── 2. 변환 + 필터링 ──────────────────────────────────────────────
    print("[2] 리샘플 (1024Hz→500Hz) + 12-lead 0-fill + 라벨 필터링")
    signals_out, labels_out = [], []
    n_skip = 0

    for i in range(N_total):
        lbl = seg_labels[i]
        if lbl not in LABEL_MAP:
            n_skip += 1
            continue

        # NaN/Inf 검사
        seg = sig_segs[i]
        if not np.isfinite(seg).all():
            n_skip += 1
            continue

        # 리샘플: 10,240 → 5,000
        seg_rs = sp_resample(seg, SEG_LEN_OUT).astype(np.float32)

        # 12-lead 0-fill: slot 1 (II)
        x = np.zeros((N_LEADS, SEG_LEN_OUT), dtype=np.float32)
        x[LEAD_SLOT] = seg_rs

        signals_out.append(x)
        labels_out.append(LABEL_MAP[lbl])

    # ── 3. 저장 ───────────────────────────────────────────────────────
    sig_arr = np.stack(signals_out).astype(np.float32)
    lab_arr = np.array(labels_out, dtype=np.int8)

    np.save(os.path.join(args.out_dir, "signals.npy"), sig_arr)
    np.save(os.path.join(args.out_dir, "labels.npy"),  lab_arr)

    n_emg = int((lab_arr == 1).sum())
    n_nrm = int((lab_arr == 0).sum())

    print()
    print("=" * 65)
    print("전처리 완료")
    print("=" * 65)
    print(f"  입력:    {N_total}개 세그먼트")
    print(f"  제외:    {n_skip}개 (Noise/Others/NaN)")
    print(f"  유효:    {len(lab_arr)}개")
    print(f"    응급(AF):  {n_emg}")
    print(f"    정상(NSR): {n_nrm}")
    print(f"  출력 shape: {sig_arr.shape}  dtype={sig_arr.dtype}")
    print(f"  저장: {args.out_dir}")
    print()
    print("다음 단계:")
    print("  → python scripts/eval_external.py --db cachet")
    print("=" * 65)


if __name__ == "__main__":
    main()
