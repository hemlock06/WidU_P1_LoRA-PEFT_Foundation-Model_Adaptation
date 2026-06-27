"""
PTB-XL 다운로드 (download_ptbxl.py)
=====================================
PhysioNet ptb-xl/1.0.3 다운로드.
  - 21,837건, 12-lead, 500Hz (records500/) + 100Hz (records100/) 제공
  - 500Hz 버전만 사용 (ECG-FM 입력 규격)
  - 라벨: ptbxl_database.csv (superclass 컬럼 포함)

사용법:
  python scripts/download_ptbxl.py
  python scripts/download_ptbxl.py --out_dir data/raw/ptbxl
  python scripts/download_ptbxl.py --workers 32   # 병렬 다운로드 스레드 수
"""

import argparse
import os
import sys
import time
import urllib.request
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

BASE_URL = "https://physionet.org/files/ptb-xl/1.0.3/"
OUT_DEFAULT = "data/raw/ptbxl"

# 필요한 파일 목록 (500Hz 레코드 + 라벨만)
ESSENTIAL_FILES = [
    "ptbxl_database.csv",       # 라벨 + 메타데이터
    "scp_statements.csv",        # SCP → superclass 매핑
    "LICENSE.txt",
    "SHA256SUMS.txt",
]

BASE_RECORDS_URL = "https://physionet.org/files/ptb-xl/1.0.3/"

# 전역 카운터 (스레드 안전)
_lock = threading.Lock()
_done = 0
_skip = 0
_fail = 0
_total = 0


def download_file(url, dst_path, desc=""):
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    if os.path.exists(dst_path):
        print(f"  [건너뜀] {desc or os.path.basename(dst_path)} (이미 존재)")
        return
    print(f"  다운로드: {desc or os.path.basename(dst_path)} ...", end="", flush=True)
    try:
        urllib.request.urlretrieve(url, dst_path)
        size_mb = os.path.getsize(dst_path) / 1e6
        print(f" {size_mb:.1f}MB 완료")
    except Exception as e:
        print(f" 실패: {e}")
        raise


def _download_with_retry(url, dst, max_retries=4, base_delay=2.0):
    """지수 백오프 재시도 포함 단일 파일 다운로드"""
    for attempt in range(max_retries):
        try:
            urllib.request.urlretrieve(url, dst)
            return True
        except Exception as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)  # 2,4,8,16초
                time.sleep(delay)
            else:
                return False, str(e)
    return False, "max retries"


def _download_one_record(args_tuple):
    """단일 레코드(.hea + .dat) 다운로드 (ThreadPoolExecutor 용)"""
    global _done, _skip, _fail
    rec_path, out_dir = args_tuple
    rec_dir  = os.path.join(out_dir, os.path.dirname(rec_path))
    rec_base = os.path.join(out_dir, rec_path)
    os.makedirs(rec_dir, exist_ok=True)

    local_done = 0
    local_skip = 0
    local_fail = 0

    for ext in [".hea", ".dat"]:
        dst = rec_base + ext
        if os.path.exists(dst):
            local_skip += 1
            continue
        url = BASE_RECORDS_URL + rec_path + ext
        result = _download_with_retry(url, dst)
        if result is True:
            local_done += 1
        else:
            local_fail += 1
            err_msg = result[1] if isinstance(result, tuple) else ""
            with _lock:
                if _fail + local_fail <= 5:
                    print(f"\n  [실패] {os.path.basename(dst)}: {err_msg}", flush=True)

    with _lock:
        _done += local_done
        _skip += local_skip
        _fail += local_fail
        processed = _done + _skip
        if processed % 2000 == 0 and processed > 0:
            pct = processed / _total * 100
            print(f"  진행: {processed}/{_total} ({pct:.0f}%)  신규={_done}  skip={_skip}  실패={_fail}",
                  flush=True)


def download_records_direct(out_dir, n_workers=16):
    """
    RECORDS 파일을 먼저 파싱 후, 각 .hea/.dat를 ThreadPoolExecutor로 병렬 다운로드.
    wfdb.dl_database의 URL 버그(버전 이중 삽입) 우회.
    """
    global _done, _skip, _fail, _total
    _done = 0; _skip = 0; _fail = 0

    # RECORDS 파일 다운로드 (경로 목록)
    records_index_url = BASE_RECORDS_URL + "RECORDS"
    records_index_path = os.path.join(out_dir, "RECORDS")
    if not os.path.exists(records_index_path):
        print("  RECORDS 인덱스 다운로드...", end="", flush=True)
        urllib.request.urlretrieve(records_index_url, records_index_path)
        print(" 완료")
    else:
        print("  RECORDS 인덱스: 이미 존재")

    with open(records_index_path, "r") as f:
        all_records = [line.strip() for line in f if line.strip()]

    # 500Hz 레코드만 필터 (records500/로 시작하는 것)
    hr_records = [r for r in all_records if r.startswith("records500/")]
    _total = len(hr_records) * 2   # .hea + .dat
    print(f"  500Hz 레코드: {len(hr_records)}건 ({_total}개 파일)", flush=True)
    print(f"  병렬 스레드: {n_workers}개  (진행: 2000파일마다 출력)", flush=True)

    tasks = [(r, out_dir) for r in hr_records]

    completed_count = 0
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(_download_one_record, t) for t in tasks]
        for _ in as_completed(futures):
            completed_count += 1
            # 500 레코드마다 현황 출력 (로그 가시성)
            if completed_count % 500 == 0:
                with _lock:
                    pct = (_done + _skip) / max(_total, 1) * 100
                    print(f"  [레코드 {completed_count}/{len(hr_records)}] "
                          f"파일: {_done+_skip}/{_total} ({pct:.1f}%)  "
                          f"신규={_done}  skip={_skip}  실패={_fail}",
                          flush=True)

    print(f"\n  다운로드 완료: 신규={_done}, 기존={_skip}, 실패={_fail}")
    if _fail > 0:
        print(f"  [주의] {_fail}개 실패 — 재실행하면 건너뜀 없이 재시도")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out_dir", default=OUT_DEFAULT)
    parser.add_argument("--workers", type=int, default=8,
                        help="병렬 다운로드 스레드 수 (기본값 8, PhysioNet rate-limit 고려)")
    args = parser.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("PTB-XL 1.0.3 다운로드")
    print("=" * 60)
    print(f"저장 위치: {out_dir}")
    print(f"병렬 스레드: {args.workers}  (재시도: 4회, 지수 백오프 2-16초)")
    print()

    # 1) 메타 파일 먼저
    print("[1/2] 메타 파일 다운로드")
    for fname in ESSENTIAL_FILES:
        url = BASE_URL + fname
        dst = os.path.join(out_dir, fname)
        download_file(url, dst)

    # 2) 레코드 (병렬 다운로드)
    print()
    print("[2/2] 레코드 다운로드 (records500/)")
    download_records_direct(out_dir, n_workers=args.workers)

    # 3) 완료 확인
    print()
    print("=" * 60)
    csv_path = os.path.join(out_dir, "ptbxl_database.csv")
    if os.path.exists(csv_path):
        import csv
        with open(csv_path, encoding="utf-8") as f:
            rows = sum(1 for _ in csv.reader(f)) - 1
        print(f"라벨 파일: {csv_path} ({rows:,}건)")

    records_dir = os.path.join(out_dir, "records500")
    if os.path.isdir(records_dir):
        n_folders = len(os.listdir(records_dir))
        print(f"레코드 폴더: records500/ ({n_folders}개 서브폴더)")

    dat_count = sum(
        1 for _, _, fs in os.walk(records_dir) for f in fs if f.endswith(".dat")
    ) if os.path.isdir(records_dir) else 0
    print(f"다운로드된 .dat 파일: {dat_count:,}개 / 21,837개 목표")
    print("다음 단계: python scripts/preprocess_ptbxl.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
