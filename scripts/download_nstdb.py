"""
단계 6 보강: MIT-BIH Noise Stress Test Database (NSTDB) 다운로드
================================================================
목적:
  multi-SNR 모션 증강의 노이즈 소스 확보.
  NSTDB의 3종 노이즈 레코드(bw/em/ma)만 PhysioNet에서 내려받음.

노이즈 종류 (4페이지_리빌딩_최종정리.md 섹션 6 / 13):
  bw = baseline wander   (기저선 변동)
  ma = muscle artifact   (근전도 잡음)
  em = electrode motion  (전극 움직임 — 가장 까다로움, 느슨한 접촉의 proxy)

사양 (검증된 사실 원장):
  2채널, 360Hz 샘플링 (우리 신호는 500Hz → 증강 단계에서 리샘플)
  각 노이즈 레코드 약 30분 길이, calibrated SNR 주입용

사용법:
  python scripts/download_nstdb.py
"""

import os
import sys

# Windows 콘솔(cp949)에서 한글·em-dash 출력 시 UnicodeEncodeError 방지
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# 데이터는 _git 레포가 아니라 비-_git 디렉토리에 저장 (CPSC와 동일 관행)
DEST = "D:/WidU_ecg-fm_emergency-detection/data/raw/nstdb"
NOISE_RECORDS = ["bw", "em", "ma"]


def main():
    try:
        import wfdb
    except ImportError:
        sys.exit("[오류] wfdb 미설치 — pip install wfdb")

    os.makedirs(DEST, exist_ok=True)
    print("=" * 60)
    print("NSTDB 노이즈 레코드 다운로드")
    print("=" * 60)
    print(f"대상 레코드: {NOISE_RECORDS}")
    print(f"저장 경로:   {DEST}")
    print()

    # nstdb 전체 대신 노이즈 레코드(bw/em/ma)만 선택 다운로드
    wfdb.dl_database("nstdb", dl_dir=DEST, records=NOISE_RECORDS)

    # ── 검증: 받은 레코드를 실제로 로드해 사양 확인 ──────────────────
    print("\n[검증] 노이즈 레코드 로드 확인")
    print("-" * 60)
    for rec in NOISE_RECORDS:
        path = os.path.join(DEST, rec)
        sig, fields = wfdb.rdsamp(path)
        print(f"  {rec}: shape={sig.shape}, fs={fields['fs']}Hz, "
              f"units={fields['units']}, leads={fields['sig_name']}")

    print("\n[완료] NSTDB 노이즈 소스 확보 — multi-SNR 증강 준비됨")


if __name__ == "__main__":
    main()
