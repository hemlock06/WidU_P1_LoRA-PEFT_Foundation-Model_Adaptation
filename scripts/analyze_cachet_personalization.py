"""
CACHET-CADB 개인화 타당성 — per-subject AF 이벤트 분포 점검
==========================================================
가설: 개인 ECG baseline을 학습해 이탈을 이상으로 검출하는 within-subject 개인화가
      AF 검출을 개선하는가. 성립 조건 = 다수 subject가 사람당 충분한 AF 이벤트를 보유해야
      within-subject 시간분할(적응→평가)이 통계적으로 성립.

방법: CACHET-CADB 종단 주석(per-subject annotation.csv: Start,End,Class)을 집계.
      Class 매핑은 데이터 descriptor(Frontiers Cardiovasc. Med. 2022, Table 7) 기준:
        1=AF, 2=NSR, 3=Noise, 4=Others.
      16GB 신호는 미해제 — 주석 CSV만 직접 파싱(zip 또는 추출 디렉터리).

결과(측정): 총 1602 주석(논문 일치). AF(이상) ≥20건 충족 = 6/24 subject뿐(편중).
      → within-subject 개인화 적응·평가가 소수 subject에서만 성립 + AF 모집단 검출이 이미
        높은 수준(records/03)이라 한계이득 입증 곤란 → 개인화 학습 미진입(데이터 부족 결론).

사용:
  python scripts/analyze_cachet_personalization.py --src D:/Downloads/CACHETCADB
  python scripts/analyze_cachet_personalization.py --src D:/Downloads/CACHETCADB.zip
"""
from __future__ import annotations
import argparse, os, io, csv, re, collections, zipfile, sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

CLASS_NAME = {1: "AF", 2: "NSR", 3: "Noise", 4: "Others"}  # descriptor Table 7
AF = 1
MIN_EVENTS = 20  # within-subject 적응→평가에 필요한 사람당 최소 이상 이벤트(보수적)


def iter_annotation_csv(src: str):
    """(subject, reader_lines) 산출. src = .zip 또는 추출 디렉터리."""
    if src.lower().endswith(".zip"):
        z = zipfile.ZipFile(src)
        for n in z.namelist():
            if n.endswith("annotation.csv") and "__MACOSX" not in n:
                subj = re.search(r"annotations/([^/]+)/", n).group(1)
                txt = io.TextIOWrapper(z.open(n), encoding="utf-8", errors="replace")
                yield subj, txt
    else:
        for root, _, files in os.walk(src):
            if "__MACOSX" in root:
                continue
            for f in files:
                if f == "annotation.csv":
                    p = os.path.join(root, f)
                    m = re.search(r"annotations[\\/]+([^\\/]+)[\\/]", p)
                    if m:
                        yield m.group(1), open(p, encoding="utf-8", errors="replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="D:/Downloads/CACHETCADB",
                    help="CACHET-CADB 추출 디렉터리 또는 .zip 경로")
    args = ap.parse_args()

    per = collections.defaultdict(collections.Counter)   # subj -> Counter(class)
    for subj, fh in iter_annotation_csv(args.src):
        rd = csv.reader(fh); next(rd, None)              # 헤더 skip
        for row in rd:
            if len(row) >= 3 and row[2].strip().lstrip("-").isdigit():
                per[subj][int(row[2])] += 1
        fh.close()

    subs = sorted(per.keys())
    total = sum(sum(c.values()) for c in per.values())
    glob = collections.Counter()
    for c in per.values():
        glob.update(c)

    print("=" * 64)
    print(f"CACHET-CADB 개인화 타당성 점검  (src={args.src})")
    print("=" * 64)
    print(f"subject 수: {len(subs)}  | 총 주석 샘플: {total} (descriptor: 1602)")
    print("전역 클래스 분포:",
          {f"{k}={CLASS_NAME.get(k,k)}": v for k, v in sorted(glob.items())})
    print()
    af = {s: per[s].get(AF, 0) for s in subs}
    n_suff = sum(1 for v in af.values() if v >= MIN_EVENTS)
    n_any = sum(1 for v in af.values() if v >= 1)
    print(f"[AF(이상) per-subject]  ≥{MIN_EVENTS}건: {n_suff}/{len(subs)}명  | ≥1건: {n_any}/{len(subs)}명")
    print("  AF 보유 subject (내림차순):")
    for s in sorted(subs, key=lambda x: -af[x]):
        if af[s] > 0:
            print(f"    {s:10s} AF={af[s]:4d}")
    print()
    print("판정:")
    if n_suff >= len(subs) * 0.5:
        print(f"  → 다수({n_suff}/{len(subs)})가 사람당 충분 → 개인화 적응·평가 통계 성립 가능.")
    else:
        print(f"  → 소수({n_suff}/{len(subs)})만 사람당 충분 = 편중 → within-subject 개인화 평가 불성립.")
        print("    + AF 모집단 검출 이미 높음(records/03) → 한계이득 입증 곤란 → 개인화 미진입(데이터 부족).")


if __name__ == "__main__":
    main()
