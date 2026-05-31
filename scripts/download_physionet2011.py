"""
단계 7 사전: PhysioNet/CinC Challenge 2011 다운로드
====================================================
목적:
  신호품질 게이트(1D CNN) 학습용 gold label 확보.
  per-record quality grade (A/B/C/D/F 또는 유사) 포함 여부 확인.

데이터셋 정보:
  - 용량: set-a.tar.gz (~103 MB) + set-b.tar.gz (~51 MB)
  - 레코드 수: set-a 1000개(학습, 라벨 공개), set-b 500개(테스트)
  - 포맷: 12-lead, 500Hz, 10s (5000 samples) — ECG-FM 입력과 동일!
  - 라벨: set-a/RECORDS-acceptable / RECORDS-unacceptable 파일로 품질 분류

사용법:
  python scripts/download_physionet2011.py
"""

import os
import sys
import tarfile
import urllib.request

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

DEST = "data/raw/physionet2011"
BASE_URL = "https://physionet.org/files/challenge-2011/1.0.0"

FILES = [
    ("set-a.tar.gz", "set-a"),
    ("set-b.tar.gz", "set-b"),
]


def download_with_progress(url: str, dest_path: str, label: str):
    def hook(count, block, total):
        if total > 0 and count % 200 == 0:
            pct = min(count * block / total * 100, 100)
            print(f"\r  {label}: {pct:.1f}%", end="", flush=True)

    urllib.request.urlretrieve(url, dest_path, reporthook=hook)
    print(f"\r  {label}: 100.0%")


def main():
    os.makedirs(DEST, exist_ok=True)
    print("=" * 60)
    print("PhysioNet Challenge 2011 다운로드")
    print("=" * 60)
    print(f"저장 경로: {DEST}")
    print()

    for fname, subdir in FILES:
        url = f"{BASE_URL}/{fname}"
        dest_tar = os.path.join(DEST, fname)
        dest_dir = os.path.join(DEST, subdir)

        if os.path.isdir(dest_dir) and len(os.listdir(dest_dir)) > 10:
            print(f"[스킵] {subdir} 이미 존재 ({len(os.listdir(dest_dir))}개 파일)")
            continue

        print(f"[다운로드] {fname}")
        download_with_progress(url, dest_tar, fname)

        print(f"[압축해제] {fname} -> {dest_dir}")
        with tarfile.open(dest_tar, "r:gz") as tf:
            tf.extractall(DEST)
        os.remove(dest_tar)
        print(f"  완료: {len(os.listdir(dest_dir))}개 파일")

    # ── 라벨 구조 확인 ────────────────────────────────────────────
    print("\n[라벨 확인] set-a 품질 라벨")
    set_a_dir = os.path.join(DEST, "set-a")
    label_files = [f for f in os.listdir(set_a_dir)
                   if "RECORD" in f.upper() or "label" in f.lower() or ".tsv" in f]
    print(f"  라벨 관련 파일: {label_files}")

    for lf in label_files[:3]:
        path = os.path.join(set_a_dir, lf)
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        print(f"\n  {lf} (총 {len(lines)}줄, 처음 5줄):")
        for line in lines[:5]:
            print(f"    {line.rstrip()}")

    # .hea 파일에서 품질 정보 추출 가능 여부 확인
    hea_files = [f for f in os.listdir(set_a_dir) if f.endswith(".hea")]
    if hea_files:
        sample = os.path.join(set_a_dir, hea_files[0])
        with open(sample, encoding="utf-8", errors="replace") as f:
            content = f.read()
        print(f"\n  샘플 .hea ({hea_files[0]}):")
        for line in content.split("\n")[:8]:
            print(f"    {line}")

    print(f"\n[완료] PhysioNet 2011 다운로드. set-a={len(os.listdir(set_a_dir))}개 파일")


if __name__ == "__main__":
    main()
