"""
SPH (Shandong Provincial Hospital) 12-lead ECG → 5-class 멀티태스크 전처리
========================================================================
출처: Liu et al., "A large-scale multi-label 12-lead ECG database with standardized
      diagnostic statements", Scientific Data 2022 (Figshare collection 5779802).
      25,770 레코드 / 24,666 환자 / 500Hz / 10~60초 / AHA 진단코드(multi-label).
용도: 급성허혈(class 2) 보강용 추가 학습 데이터. PhysioNet 2021 미포함(ECG-FM 사전학습 비중복).
라이선스: 데이터셋 dictionary/mapping CC0 (Figshare). 연구 사용.

스펙 정합(records/01): 500Hz·12리드·10초(앞 5000샘플)·정규화 OFF·patient-level split(70/15/15, seed=42).

AHA Code → 5-class 매핑 (code.csv 기준):
  0 NSR       : 1 (Normal ECG)
  1 AF        : 50 (Atrial fibrillation)
  2 Ischemia★ : 145(ST deviation), 146(ST dev+T-change), 160/161/165/166(MI)
  3 Conduction: 82(prolonged PR=I-AVB), 83-88(AV blocks), 101/102(fascicular),
                104(LBBB), 105(incomplete RBBB), 106(RBBB)
  4 Ectopic   : 30/31(atrial premature), 36(junctional premature), 60(ventricular premature)
  제외(-1): 21/22/23(sinus tachy/brady/arrhythmia=rate variant), 51(flutter), 54(junc tachy),
           37(junctional escape), 80/81(short PR/AV ratio), 108(preexcitation),
           120/121/125(axis/low voltage), 140/142/143(enlargement/hypertrophy),
           147(T-wave abnormality only), 148(QT), 152(TU), 153(ST-T due to hypertrophy=비허혈),
           155(early repolarization), modifier(308~367). → 비대상/모호 코드는 라벨에서 제외.

multi-label 축약(SPH는 multi-label): 우선순위 Ischemia(2) > AF(1) > Conduction(3) > Ectopic(4) > NSR(0).
  보강 타깃 ischemia를 최대 보존. 대상 클래스(1~4) 전무 + Normal(1) → NSR(0). 매핑 코드 전무 → 제외.
labels_bin: 응급=1 (AF·Ischemia = class {1,2}), 정상=0. — CPSC mc 규약과 동일.
"""
from __future__ import annotations
import argparse, os, sys, csv, io, tarfile, collections, random

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

CLASS_NAMES = ["NSR", "AF", "Ischemia", "Conduction", "Ectopic"]
EMERGENCY = {1, 2}            # AF, Ischemia → labels_bin=1
TARGET_FS = 500
SEG_LEN = 5000               # 10초 @500Hz
PRIORITY = [2, 1, 3, 4]      # ischemia 우선, 그 다음 AF·conduction·ectopic; 마지막 NSR(0)

# AHA primary code → 5-class
AHA2CLASS = {
    1: 0,                                          # NSR
    50: 1,                                          # AF
    145: 2, 146: 2, 160: 2, 161: 2, 165: 2, 166: 2,  # Ischemia (ST dev / MI)
    82: 3, 83: 3, 84: 3, 85: 3, 86: 3, 87: 3, 88: 3,  # AV blocks (I-AVB 등)
    101: 3, 102: 3, 104: 3, 105: 3, 106: 3,          # fascicular / bundle branch
    30: 4, 31: 4, 36: 4, 60: 4,                       # ectopic (APC/junctional/VPC)
}


def parse_codes(field: str):
    out = []
    for tok in str(field).replace(",", ";").split(";"):
        tok = tok.strip()
        if tok.lstrip("-").isdigit():
            out.append(int(tok))
    return out


def reduce_label(codes):
    """AHA 코드 리스트 → 단일 5-class 라벨(우선순위) 또는 -1(제외)."""
    present = {AHA2CLASS[c] for c in codes if c in AHA2CLASS}
    for cls in PRIORITY:
        if cls in present:
            return cls
    if 0 in present:
        return 0
    return -1


def read_metadata(meta_path):
    rows = []
    with open(meta_path, encoding="utf-8", errors="replace") as f:
        rd = csv.DictReader(f)
        for r in rd:
            rows.append(r)
    return rows


def analyze(meta_path):
    rows = read_metadata(meta_path)
    n = len(rows)
    dist = collections.Counter()
    multi = 0                      # 대상 클래스(1~4) 2개 이상 보유(축약 전)
    excluded = 0
    pat_by_cls = collections.defaultdict(set)
    lens = collections.Counter()
    for r in rows:
        codes = parse_codes(r["AHA_Code"])
        present = {AHA2CLASS[c] for c in codes if c in AHA2CLASS}
        if len({c for c in present if c in (1, 2, 3, 4)}) >= 2:
            multi += 1
        lab = reduce_label(codes)
        dist[lab] += 1
        if lab == -1:
            excluded += 1
        else:
            pat_by_cls[lab].add(r["Patient_ID"])
        lens[int(r.get("N", 0))] += 1

    print("=" * 64)
    print(f"SPH 매핑 sanity  (records={n}, meta={meta_path})")
    print("=" * 64)
    print(f"{'class':<12} {'records':>8} {'patients':>9}")
    for c in range(5):
        print(f"  [{c}] {CLASS_NAMES[c]:<8} {dist.get(c,0):>8} {len(pat_by_cls[c]):>9}")
    print(f"  [-1] 제외      {dist.get(-1,0):>8}")
    used = n - dist.get(-1, 0)
    n_emerg = sum(dist.get(c, 0) for c in EMERGENCY)
    print("-" * 32)
    print(f"  사용 가능       {used:>8}  (전체의 {100*used/n:.1f}%)")
    print(f"  응급(AF+허혈)   {n_emerg:>8}  | 정상/기타 {used-n_emerg:>8}")
    print(f"  다중대상(축약 전 1~4 ≥2) {multi}  ({100*multi/n:.1f}%)")
    print(f"  레코드 길이 분포(N): " + ", ".join(f"{k}={v}" for k, v in sorted(lens.items())[:8]))
    print()
    print("주: 우선순위 축약 Ischemia>AF>Conduction>Ectopic>NSR. 비대상/모호 코드 제외(-1).")
    return dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sph_dir", default="data/raw/sph", help="metadata.csv·code.csv·records 위치")
    ap.add_argument("--records_tar", default="data/raw/sph/records.tar.gz")
    ap.add_argument("--records_dir", default="data/raw/sph/records",
                    help="추출된 *.h5 디렉터리 (full 모드)")
    ap.add_argument("--out_dir", default="data/processed/sph")
    ap.add_argument("--clip_mv", type=float, default=8.0,
                    help="진폭 클리핑 ±mV (SPH 극단 아티팩트 제거; CPSC p99≈7.6 포함). 0=비활성")
    ap.add_argument("--analyze_only", action="store_true",
                    help="metadata.csv만으로 매핑/분포 검증 (신호 미접근)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    meta_path = os.path.join(args.sph_dir, "metadata.csv")
    dist = analyze(meta_path)

    if args.analyze_only:
        return

    # ── full 전처리: tar에서 HDF5 직접 적재 → npy (메모리 안전: split별 사전할당·저장·해제) ──
    # 16GB RAM 환경 → 전체 list+stack(≈9GB) 금지. split별 np.empty 사전할당 후 in-place 채움.
    import h5py, tarfile, io as _io
    rows = read_metadata(meta_path)
    rng = np.random.default_rng(args.seed)

    # 라벨 가용 레코드 + 환자 단위 split
    recs = []
    for r in rows:
        lab = reduce_label(parse_codes(r["AHA_Code"]))
        if lab >= 0:
            recs.append((r["ECG_ID"], r["Patient_ID"], lab))
    pats = sorted({p for _, p, _ in recs})
    rng.shuffle(pats)
    n_tr = int(0.70 * len(pats)); n_va = int(0.15 * len(pats))
    split_of = {}
    for i, p in enumerate(pats):
        split_of[p] = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")
    by_split = {"train": [], "val": [], "test": []}
    for ecg_id, pid, lab in recs:
        by_split[split_of[pid]].append((ecg_id, lab))

    print(f"[tar] 인덱스 구축: {args.records_tar}")
    tf = tarfile.open(args.records_tar, "r")          # 실제 비압축 tar (auto-detect)
    name2m = {m.name: m for m in tf.getmembers() if m.isfile()}
    print(f"      멤버 {len(name2m)}개")

    os.makedirs(args.out_dir, exist_ok=True)
    for s in ("train", "val", "test"):
        items = by_split[s]; n = len(items)
        sig = np.empty((n, 12, SEG_LEN), dtype=np.float32)   # split별 사전할당
        mc  = np.empty(n, dtype=np.int64); bn = np.empty(n, dtype=np.int64); ids = []
        w = 0; miss = 0
        for ecg_id, lab in items:
            m = name2m.get(f"records/{ecg_id}.h5")
            if m is None:
                miss += 1; continue
            with h5py.File(_io.BytesIO(tf.extractfile(m).read()), "r") as hf:
                arr = np.asarray(hf["ecg"], dtype=np.float32)
            if arr.shape[0] != 12 and arr.shape[-1] == 12:
                arr = arr.T
            if arr.shape[1] >= SEG_LEN:
                arr = arr[:, :SEG_LEN]
            else:
                arr = np.pad(arr, ((0, 0), (0, SEG_LEN - arr.shape[1])))
            if args.clip_mv > 0:                         # SPH ~1.1% 병적 아티팩트(최대 452mV) 제거
                np.clip(arr, -args.clip_mv, args.clip_mv, out=arr)
            sig[w] = arr; mc[w] = lab; bn[w] = 1 if lab in EMERGENCY else 0
            ids.append(ecg_id); w += 1
        sig = sig[:w]; mc = mc[:w]; bn = bn[:w]
        d = os.path.join(args.out_dir, s); os.makedirs(d, exist_ok=True)
        np.save(os.path.join(d, "signals.npy"), sig)
        np.save(os.path.join(d, "labels.npy"), mc)
        np.save(os.path.join(d, "labels_bin.npy"), bn)
        np.save(os.path.join(d, "record_ids.npy"), np.array(ids))
        print(f"  {s}: {w}개 저장 (누락 {miss}) → {d}")
        del sig, mc, bn                                # 다음 split 전 메모리 해제
    tf.close()


if __name__ == "__main__":
    main()
