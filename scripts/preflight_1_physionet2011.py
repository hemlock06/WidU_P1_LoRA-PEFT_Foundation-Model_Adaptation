"""
Pre-flight 1: PhysioNet/CinC Challenge 2011 per-channel 라벨 가용성 확인
=======================================================================
목적:
  게이트(신호품질 게이트) 학습에 필요한 per-lead(채널별) 라벨이
  공식 데이터 release에 포함되어 있는지 확인.

배경 (4페이지_리빌딩_최종정리.md 섹션 8-3):
  PhysioNet 2011 Challenge는 1,539개 12-lead ECG (10s @ 500Hz, 모바일 폰 수집).
  annotator들이 채널별로 A(0.95)/B(0.85)/C(0.75)/D(0.60)/F(0) grade를 매겼으며,
  이를 합쳐 acceptable/unacceptable로 분류.
  per-channel grade가 공식 release에 있으면 → 게이트 학습 gold label로 직접 사용.
  없으면 → NSTDB SNR-conditioned 합성 라벨 단독 진행.

실행 전 준비:
  1. PhysioNet 계정 생성: https://physionet.org/register/ (open access, CITI 불필요)
  2. 2011 Challenge Set A 다운로드:
       wget -r -N -c -np https://physionet.org/files/challenge-2011/1.0.0/set-a/ -P data/raw/physionet2011
     또는 wfdb 패키지:
       python -c "import wfdb; wfdb.dl_database('challenge-2011', 'data/raw/physionet2011', records=['set-a'])"
  3. 이 스크립트를 data/raw/physionet2011 에 set-a 폴더가 있는 상태에서 실행

사용법:
  python scripts/preflight_1_physionet2011.py --data_dir data/raw/physionet2011
"""

import argparse
import os
import glob
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default="data/raw/physionet2011",
        help="PhysioNet 2011 다운로드 폴더 경로"
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    print("=" * 60)
    print("Pre-flight 1: PhysioNet 2011 per-channel 라벨 확인")
    print("=" * 60)
    print(f"데이터 경로: {data_dir}")
    print()

    # 1. 폴더 존재 확인
    if not os.path.exists(data_dir):
        print(f"[FAIL] 폴더 없음: {data_dir}")
        print("  → 위 '실행 전 준비' 지시대로 데이터를 먼저 다운로드하세요.")
        return

    # 2. 폴더 구조 파악
    all_files = []
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            all_files.append(os.path.join(root, f))

    print(f"[INFO] 전체 파일 수: {len(all_files)}")

    # 확장자별로 분류
    ext_counts = {}
    for f in all_files:
        ext = os.path.splitext(f)[1].lower()
        ext_counts[ext] = ext_counts.get(ext, 0) + 1
    print("[INFO] 확장자별 파일 수:")
    for ext, cnt in sorted(ext_counts.items()):
        print(f"  {ext or '(없음)'}: {cnt}개")
    print()

    # 3. acceptable / unacceptable 레코드 목록 파일 확인
    # 공식 release에는 RECORDS-acceptable, RECORDS-unacceptable 이 있을 수 있음
    target_files = [
        "RECORDS-acceptable",
        "RECORDS-unacceptable",
        "RECORDS",
        "answers.txt",   # annotator 채점 파일 이름이 다를 수 있음
        "set-a.txt",
    ]
    print("[STEP 1] 레코드 목록 / 라벨 파일 검색:")
    for tf in target_files:
        found = glob.glob(os.path.join(data_dir, "**", tf), recursive=True)
        if found:
            print(f"  발견: {tf}")
            for f in found[:3]:
                print(f"       {f}")
        else:
            print(f"  [오류] 없음: {tf}")
    print()

    # 4. 채널별 grade 정보가 있을 수 있는 파일 검색
    # annotator grade는 .txt 또는 .csv 형태일 수 있음
    print("[STEP 2] per-channel grade 파일 후보 검색 (.csv, .txt, .ann):")
    candidate_exts = [".csv", ".ann", ".qrs", ".xws"]
    for ext in candidate_exts:
        found = glob.glob(os.path.join(data_dir, "**", f"*{ext}"), recursive=True)
        if found:
            print(f"  {ext} 파일 {len(found)}개 발견")
            # 첫 파일 내용 미리보기
            with open(found[0], "r", errors="replace") as fh:
                preview = fh.read(500)
            print(f"    첫 파일: {found[0]}")
            print(f"    내용 미리보기:\n{'='*40}")
            print(preview)
            print("=" * 40)
        else:
            print(f"  [오류] {ext} 파일 없음")
    print()

    # 5. .hea 헤더 파일에서 채널 이름 확인
    print("[STEP 3] .hea 헤더 파일 샘플 확인 (lead 이름·신호 정보):")
    hea_files = glob.glob(os.path.join(data_dir, "**", "*.hea"), recursive=True)
    if hea_files:
        print(f"  .hea 파일 총 {len(hea_files)}개 발견")
        with open(hea_files[0], "r", errors="replace") as fh:
            content = fh.read()
        print(f"  샘플 ({hea_files[0]}):")
        print(content[:600])
    else:
        print("  [오류] .hea 파일 없음")
    print()

    # 6. wfdb 로드 시도 (wfdb 패키지가 있을 때)
    print("[STEP 4] wfdb 패키지로 레코드 로드 시도:")
    try:
        import wfdb
        # set-a 폴더에서 첫 번째 레코드 찾기
        dat_files = glob.glob(os.path.join(data_dir, "**", "*.dat"), recursive=True)
        if dat_files:
            # 레코드 경로 (확장자 제거)
            record_path = os.path.splitext(dat_files[0])[0]
            record = wfdb.rdrecord(record_path)
            print(f"  레코드 로드 성공: {record_path}")
            print(f"    fs={record.fs}Hz, n_sig={record.n_sig}, sig_len={record.sig_len}")
            print(f"    채널 이름: {record.sig_name}")

            # annotation 로드 시도
            try:
                ann = wfdb.rdann(record_path, "ann")
                print(f"    annotation 로드 성공: symbol={ann.symbol[:5]}...")
            except Exception as e:
                print(f"    ℹ️  annotation 없음 또는 다른 확장자: {e}")

            # quality annotation 시도
            for ext in ["qrs", "xws", "atr"]:
                try:
                    ann2 = wfdb.rdann(record_path, ext)
                    print(f"    {ext} annotation 발견: {ann2}")
                except Exception:
                    pass
        else:
            print("  [오류] .dat 파일 없음 — 데이터가 아직 다운로드되지 않음")

    except ImportError:
        print("  ℹ️  wfdb 미설치. pip install wfdb 후 재실행하면 레코드 로드 확인 가능.")
    print()

    # 7. 최종 판정 가이드
    print("=" * 60)
    print("[결과 해석 가이드]")
    print()
    print("per-channel 라벨이 있는 경우:")
    print("  → 'RECORDS-acceptable/unacceptable' 파일에 채널별 기록 있음")
    print("  → 또는 .csv/.ann 파일에 annotator별 grade(A/B/C/D/F) 있음")
    print("  → decisions.md 업데이트: '게이트 학습 = PhysioNet 2011 per-channel grade 직접 사용'")
    print()
    print("[오류] per-channel 라벨이 없는 경우 (per-record binary 만 있음):")
    print("  → decisions.md 업데이트: '게이트 학습 = NSTDB SNR-conditioned 합성 라벨 단독'")
    print("  → 섹션 8-3의 'Per-channel grade 직접 사용' 경로를 합성 라벨로 대체")
    print()
    print("→ 이 결과를 decisions.md에 기록하세요.")
    print("=" * 60)


if __name__ == "__main__":
    main()
