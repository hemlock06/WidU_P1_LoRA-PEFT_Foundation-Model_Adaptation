"""
CACHET-CADB Short Format 다운로드 + patient-level split 확인
=============================================================
출처: DTU Data (data.dtu.dk)
  DOI: 10.11583/DTU.14547330.v1
  URL: https://ndownloader.figshare.com/files/27917358

데이터셋 정보:
  - 1,602개 10초 ECG 샘플 (AF / NSR / noise / other)
  - 24명 환자, 자유보행 웨어러블 단채널, 1024 Hz
  - HDF5 포맷: cachet-cadb_short_format_without_context.hdf5
  - 용량: ~125 MB (zip)

주의:
  - 단채널(1-lead) → ECG-FM 입력 시 lead II 슬롯에 배치, 나머지 11채널 0-fill
  - 1024 Hz → 500 Hz 리샘플 필요 (10240 → 5000 samples)
  - patient-level split 필수 (24명 → train/held-out 분리)

사용법:
  python scripts/download_cachet.py
"""

import os
import sys
import urllib.request
import zipfile

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

DEST = "data/raw/cachet"
DOWNLOAD_URL = "https://ndownloader.figshare.com/files/27917358"
ZIP_NAME = "cachet-cadb_short_format.zip"
HDF5_NAME = "cachet-cadb_short_format_without_context.hdf5"


def download_with_progress(url, dest_path, label):
    def hook(count, block, total):
        if total > 0 and count % 100 == 0:
            pct = min(count * block / total * 100, 100)
            print(f"\r  {label}: {pct:.1f}%", end="", flush=True)

    urllib.request.urlretrieve(url, dest_path, reporthook=hook)
    print(f"\r  {label}: 100.0%")


def inspect_hdf5(hdf5_path):
    """HDF5 구조 탐색 + patient-level split 가능 여부 확인."""
    try:
        import h5py
    except ImportError:
        print("[경고] h5py 미설치 — pip install h5py")
        return

    print(f"\n[HDF5 구조 분석] {os.path.basename(hdf5_path)}")
    with h5py.File(hdf5_path, "r") as f:

        def print_tree(name, obj, depth=0):
            indent = "  " * depth
            shape = obj.shape if hasattr(obj, "shape") else ""
            dtype = obj.dtype if hasattr(obj, "dtype") else ""
            print(f"{indent}{name}: {shape} {dtype}")

        print("  최상위 키:", list(f.keys()))
        for key in list(f.keys())[:3]:
            obj = f[key]
            print(f"\n  [{key}]")
            if hasattr(obj, "keys"):
                for subkey in list(obj.keys())[:5]:
                    sub = obj[subkey]
                    shape = sub.shape if hasattr(sub, "shape") else ""
                    dtype = sub.dtype if hasattr(sub, "dtype") else ""
                    print(f"    {subkey}: shape={shape}, dtype={dtype}")
            elif hasattr(obj, "shape"):
                print(f"    shape={obj.shape}, dtype={obj.dtype}")
                print(f"    sample values: {obj[:3] if obj.ndim == 1 else obj[0, :3]}")

        # 환자 ID 탐색
        pid_candidates = [
            "patient_id",
            "patient",
            "subject",
            "pid",
            "PatientID",
            "patient_ids",
            "subjects",
        ]
        print("\n  [환자 ID 탐색]")
        for key in pid_candidates:
            if key in f:
                ids = f[key][:]
                import numpy as np

                unique = np.unique(ids)
                print(f"    '{key}': {len(ids)}개 샘플, {len(unique)}명 환자")
                print(f"    유니크 ID: {unique[:10]}")
                break
        else:
            # 최상위 키가 환자별로 나뉜 경우
            top_keys = list(f.keys())
            print(f"    최상위 키 {len(top_keys)}개: {top_keys[:8]}")

        # 라벨 탐색
        label_candidates = [
            "label",
            "labels",
            "rhythm",
            "annotation",
            "class",
            "classes",
            "rhythm_label",
        ]
        print("\n  [라벨 탐색]")
        for key in label_candidates:
            if key in f:
                labs = f[key][:]
                import numpy as np

                unique_labs = np.unique(labs)
                print(f"    '{key}': {len(labs)}개, 유니크={unique_labs}")
                break
        else:
            print("    라벨 키 미발견 — 최상위 키 직접 확인 필요")


def main():
    os.makedirs(DEST, exist_ok=True)
    hdf5_path = os.path.join(DEST, HDF5_NAME)
    zip_path = os.path.join(DEST, ZIP_NAME)

    print("=" * 60)
    print("CACHET-CADB Short Format 다운로드")
    print("=" * 60)
    print("출처: DTU Data (DOI: 10.11583/DTU.14547330.v1)")
    print("용량: ~125 MB")
    print(f"저장: {DEST}")
    print()

    # ── 다운로드 ──────────────────────────────────────────────
    if os.path.exists(hdf5_path):
        print(f"[스킵] HDF5 이미 존재: {hdf5_path}")
    else:
        if not os.path.exists(zip_path):
            print(f"[다운로드] {ZIP_NAME}")
            download_with_progress(DOWNLOAD_URL, zip_path, ZIP_NAME)
        else:
            print("[스킵] zip 이미 존재")

        print(f"[압축해제] {ZIP_NAME}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
            print(f"  zip 내 파일: {names}")
            zf.extractall(DEST)
        os.remove(zip_path)
        print("  완료")

    # ── 파일 크기 확인 ────────────────────────────────────────
    sz = os.path.getsize(hdf5_path)
    print(f"\n[확인] {HDF5_NAME}: {sz:,} bytes ({sz / 1e6:.1f} MB)")

    # ── HDF5 구조 탐색 ────────────────────────────────────────
    inspect_hdf5(hdf5_path)

    print("\n[완료] CACHET-CADB Short Format 다운로드")
    print("  다음 단계: scripts/split_cachet.py 로 patient-level split 생성")


if __name__ == "__main__":
    main()
