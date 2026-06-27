"""
단계 9 사전: 외부검증 3종 다운로드
====================================
STAFF-III, INCART, LTST 를 PhysioNet에서 다운로드.

사용법:
  python scripts/download_external_dbs.py              # 전체
  python scripts/download_external_dbs.py --db incart  # 개별
  python scripts/download_external_dbs.py --db staffiii
  python scripts/download_external_dbs.py --db ltst

용량 (PhysioNet 공시):
  STAFF-III : 3.2 GB  (520 records, 104 patients, 9 leads, 1000 Hz, ~5 min/record)
  INCART    : 563 MB  (75 records, 12 leads, 257 Hz, ~30 min/record)
  LTST      :  9.5 GB (86 records, 2 leads, 250 Hz, ~23 h/record)
"""

import argparse
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

DEST_BASE = "data/raw"

DB_CONFIG = {
    "staffiii": {
        "pn_name": "staffiii",      # wfdb.get_record_list에 쓰는 PhysioNet DB 이름
        "pn_dir": "staffiii/1.0.0",
        "dest": os.path.join(DEST_BASE, "staffiii"),
        "records_prefix": "data/",   # 레코드가 data/NNN 형태
        "size_note": "3.2 GB",
        "n_records": 520,
    },
    "incart": {
        "pn_name": "incartdb",
        "pn_dir": "incartdb",
        "dest": os.path.join(DEST_BASE, "incart"),
        "records_prefix": "",
        "size_note": "563 MB",
        "n_records": 75,
    },
    "ltst": {
        "pn_name": "ltstdb",
        "pn_dir": "ltstdb",
        "dest": os.path.join(DEST_BASE, "ltst"),
        "records_prefix": "",
        "size_note": "9.5 GB",
        "n_records": 86,
    },
}


def download_db(db_name: str):
    import wfdb

    cfg = DB_CONFIG[db_name]
    os.makedirs(cfg["dest"], exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"다운로드: {db_name.upper()}  ({cfg['size_note']})")
    print(f"저장 경로: {cfg['dest']}")
    print(f"{'=' * 60}")

    recs = wfdb.get_record_list(cfg["pn_name"])
    print(f"레코드 수: {len(recs)}")

    ok, skip, err = 0, 0, 0
    for i, rec in enumerate(recs, 1):
        # 로컬 저장 경로 (data/ 접두사 제거)
        local_name = rec.replace("data/", "").replace("/", "_")
        local_path = os.path.join(cfg["dest"], local_name)

        # 이미 .hea가 있으면 스킵
        if os.path.exists(local_path + ".hea"):
            skip += 1
            continue

        try:
            wfdb.dl_files(
                cfg["pn_name"],
                cfg["dest"],
                [rec + ".hea", rec + ".dat"],
                keep_subdirs=False,
            )
            # dl_files는 data/ 서브디렉토리를 유지할 수 있어 rename 처리
            src_hea = os.path.join(cfg["dest"], "data", os.path.basename(rec) + ".hea")
            if os.path.exists(src_hea):
                import shutil
                for ext in [".hea", ".dat"]:
                    s = os.path.join(cfg["dest"], "data", os.path.basename(rec) + ext)
                    d = os.path.join(cfg["dest"], os.path.basename(rec) + ext)
                    if os.path.exists(s):
                        shutil.move(s, d)
            ok += 1
        except Exception as e:
            err += 1
            if err <= 5:
                print(f"  [오류] {rec}: {e}")

        if i % 50 == 0 or i == len(recs):
            print(f"  진행: {i}/{len(recs)}  (성공={ok}, 스킵={skip}, 오류={err})")

    # annotation 파일 다운로드 (있으면)
    ann_exts = {
        "staffiii": ["atr"],
        "incart": ["atr"],
        "ltst": ["stf", "sth", "ari"],
    }
    for rec in recs:
        for ext in ann_exts.get(db_name, []):
            try:
                wfdb.dl_files(
                    cfg["pn_name"],
                    cfg["dest"],
                    [rec + "." + ext],
                    keep_subdirs=False,
                )
                # data/ 서브디렉토리 이동
                src = os.path.join(cfg["dest"], "data",
                                   os.path.basename(rec) + "." + ext)
                if os.path.exists(src):
                    import shutil
                    shutil.move(src, os.path.join(cfg["dest"],
                                                  os.path.basename(rec) + "." + ext))
            except Exception:
                pass

    # data/ 빈 디렉토리 정리
    data_dir = os.path.join(cfg["dest"], "data")
    try:
        if os.path.isdir(data_dir) and not os.listdir(data_dir):
            os.rmdir(data_dir)
    except Exception:
        pass

    print(f"\n[완료] {db_name.upper()}: 성공={ok}, 스킵={skip}, 오류={err}")
    print(f"       저장 경로: {cfg['dest']}")

    # 검증: 몇 개 레코드 로드 테스트
    print("\n[검증] 첫 3개 레코드 로드 테스트")
    loaded = 0
    for rec in recs[:5]:
        local = os.path.basename(rec)
        local_path = os.path.join(cfg["dest"], local)
        if os.path.exists(local_path + ".hea"):
            try:
                sig, fields = wfdb.rdsamp(local_path)
                print(f"  {local}: shape={sig.shape}, fs={fields['fs']}Hz")
                loaded += 1
                if loaded >= 3:
                    break
            except Exception as e:
                print(f"  {local}: 로드 실패 — {e}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", choices=["staffiii", "incart", "ltst", "all"],
                   default="all", help="다운로드할 DB (기본: all)")
    args = p.parse_args()

    try:
        import wfdb
    except ImportError:
        sys.exit("[오류] wfdb 미설치 — pip install wfdb")

    targets = list(DB_CONFIG.keys()) if args.db == "all" else [args.db]

    print("=" * 60)
    print("외부검증 데이터셋 다운로드")
    print("=" * 60)
    print(f"대상: {targets}")
    for name in targets:
        print(f"  {name}: {DB_CONFIG[name]['size_note']}")
    print()

    for name in targets:
        download_db(name)

    print("\n[전체 완료]")


if __name__ == "__main__":
    main()
