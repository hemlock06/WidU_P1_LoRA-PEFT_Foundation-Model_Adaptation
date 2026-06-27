"""
CPSC 2018 데이터 다운로더
==========================
출처: PhysioNet Challenge 2020 training set — cpsc_2018 subset
URL:  https://physionet.org/files/challenge-2020/1.0.2/training/cpsc_2018/

폴더 구조:
  cpsc_2018/
    REFERENCE.csv
    g1/  A0001.hea  A0001.mat  ...
    g2/  ...
    g7/  ...

저장 위치: data/raw/cpsc2018/
  (REFERENCE.csv + 모든 .hea/.mat를 단일 폴더에 평탄화)

사용법:
  pip install requests tqdm
  python scripts/download_cpsc2018.py
  python scripts/download_cpsc2018.py --dest data/raw/cpsc2018
  python scripts/download_cpsc2018.py --workers 8   # 병렬 다운로드 수
"""

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from html.parser import HTMLParser
from urllib.parse import urljoin

import requests
from tqdm import tqdm

BASE_URL   = "https://physionet.org/files/challenge-2020/1.0.2/training/cpsc_2018/"
GROUPS     = [f"g{i}" for i in range(1, 8)]   # g1 ~ g7
DEFAULT_DEST = "data/raw/cpsc2018"
TIMEOUT    = 30   # 초 (단일 요청)
CHUNK_SIZE = 1024 * 64  # 64 KB


# ── HTML 디렉터리 파싱 ────────────────────────────────────────────────

class LinkParser(HTMLParser):
    """PhysioNet Apache 디렉터리 리스팅에서 파일 링크 추출."""

    def __init__(self):
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            for name, val in attrs:
                if name == "href" and val and not val.startswith("?") and not val.startswith("/"):
                    self.links.append(val)


def list_directory(url: str, session: requests.Session) -> list[str]:
    """URL 디렉터리 리스팅에서 href 목록 반환."""
    resp = session.get(url, timeout=TIMEOUT)
    resp.raise_for_status()
    parser = LinkParser()
    parser.feed(resp.text)
    return parser.links


# ── 파일 다운로드 ─────────────────────────────────────────────────────

def download_file(url: str, dest_path: str, session: requests.Session) -> tuple[str, str]:
    """
    단일 파일 다운로드. 이미 존재하고 크기가 0보다 크면 스킵.
    반환: ("ok"|"skip"|"err", 메시지)
    """
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        return "skip", dest_path

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    for attempt in range(1, 5):
        try:
            with session.get(url, timeout=TIMEOUT, stream=True) as r:
                r.raise_for_status()
                with open(dest_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                        f.write(chunk)
            return "ok", dest_path
        except Exception as e:
            if attempt < 4:
                time.sleep(2 ** attempt)   # 2s, 4s, 8s
            else:
                return "err", f"{url} → {e}"

    return "err", f"{url} → 최대 재시도 초과"


# ── 태스크 수집 ───────────────────────────────────────────────────────

def collect_tasks(dest_dir: str, session: requests.Session) -> list[tuple[str, str]]:
    """
    다운로드할 (url, dest_path) 목록 수집.
    REFERENCE.csv + 각 그룹 폴더의 .hea/.mat 파일.
    """
    tasks = []

    # REFERENCE.csv (루트)
    ref_url  = urljoin(BASE_URL, "REFERENCE.csv")
    ref_dest = os.path.join(dest_dir, "REFERENCE.csv")
    tasks.append((ref_url, ref_dest))

    print("디렉터리 목록 수집 중...")
    for group in GROUPS:
        group_url = urljoin(BASE_URL, group + "/")
        try:
            links = list_directory(group_url, session)
        except Exception as e:
            print(f"  [경고] {group}/ 목록 조회 실패: {e}")
            continue

        count = 0
        for link in links:
            # .hea 또는 .mat 파일만 수집
            if not (link.endswith(".hea") or link.endswith(".mat")):
                continue
            file_url  = urljoin(group_url, link)
            file_dest = os.path.join(dest_dir, link)   # 단일 폴더에 평탄화
            tasks.append((file_url, file_dest))
            count += 1

        print(f"  {group}/: {count}개 파일 발견")

    return tasks


# ── 메인 ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest",    default=DEFAULT_DEST,
                        help="저장 폴더 (기본: %(default)s)")
    parser.add_argument("--workers", type=int, default=4,
                        help="병렬 다운로드 수 (기본: %(default)s)")
    args = parser.parse_args()

    dest_dir = os.path.abspath(args.dest)
    os.makedirs(dest_dir, exist_ok=True)
    print("=" * 65)
    print("CPSC 2018 다운로더")
    print("=" * 65)
    print(f"출처: {BASE_URL}")
    print(f"저장: {dest_dir}")
    print(f"병렬: {args.workers}개")
    print()

    session = requests.Session()
    session.headers["User-Agent"] = "cpsc2018-downloader/1.0"

    # 1. 태스크 목록 수집
    tasks = collect_tasks(dest_dir, session)
    total = len(tasks)
    print(f"\n총 {total}개 파일 대상 (REFERENCE.csv 포함)\n")

    # 2. 병렬 다운로드
    ok_count   = 0
    skip_count = 0
    err_list   = []

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(download_file, url, dest, session): (url, dest)
            for url, dest in tasks
        }
        with tqdm(total=total, unit="파일", ncols=80, dynamic_ncols=True) as pbar:
            for future in as_completed(futures):
                status, msg = future.result()
                if status == "ok":
                    ok_count += 1
                elif status == "skip":
                    skip_count += 1
                else:
                    err_list.append(msg)
                pbar.update(1)
                pbar.set_postfix(ok=ok_count, skip=skip_count, err=len(err_list))

    # 3. 결과 요약
    print()
    print("=" * 65)
    print("다운로드 완료")
    print(f"  다운로드: {ok_count}개")
    print(f"  스킵(기존): {skip_count}개")
    print(f"  오류: {len(err_list)}개")
    if err_list:
        print("\n  [오류 목록]")
        for e in err_list[:10]:
            print(f"    {e}")
        if len(err_list) > 10:
            print(f"    ... 외 {len(err_list)-10}개")
    print()

    # 4. REFERENCE.csv 확인
    ref_path = os.path.join(dest_dir, "REFERENCE.csv")
    if os.path.exists(ref_path):
        import csv
        with open(ref_path, newline="") as f:
            rows = list(csv.reader(f))
        print(f"REFERENCE.csv: {len(rows)}개 레코드")
    else:
        print("[경고] REFERENCE.csv 다운로드 실패 — 수동 확인 필요")

    hea_count = sum(1 for f in os.listdir(dest_dir) if f.endswith(".hea"))
    mat_count = sum(1 for f in os.listdir(dest_dir) if f.endswith(".mat"))
    print(f".hea 파일: {hea_count}개,  .mat 파일: {mat_count}개")
    print()
    print("다음 단계:")
    print(f"  python scripts/preprocess_cpsc2018.py --data_dir \"{dest_dir}\"")
    print("=" * 65)


if __name__ == "__main__":
    main()
