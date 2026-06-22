"""
CPSC mc + SPH 결합 학습셋 빌더 (급성허혈 보강 실험용)
=====================================================
단일 변인 = '학습 데이터에 SPH 추가'. 결합은 train에만 적용하고 val/test는 CPSC 그대로 둔다
(baseline=CPSC-only+mixed 와 동일 평가셋 → apples-to-apples).

SPH 구성: 비정상 클래스(AF·Ischemia·Conduction·Ectopic) 전량 + NSR은 --nsr_cap으로 제한
  (NSR 과다 희석 방지 + 16GB RAM 안전). 라벨/스케일은 preprocess_sph.py에서 이미 정합(clip ±8mV).
출력: data/processed/cpsc_sph/{train(결합), val(=CPSC), test(=CPSC)}
"""
from __future__ import annotations
import argparse, os, shutil, collections
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpsc_dir", default="data/processed/cpsc2018_mc")
    ap.add_argument("--sph_dir", default="data/processed/sph")
    ap.add_argument("--out_dir", default="data/processed/cpsc_sph")
    ap.add_argument("--nsr_cap", type=int, default=2000, help="SPH NSR(class 0) 최대 포함 수")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # ── SPH train 인덱스 선택 (비정상 전량 + NSR 캡) ──
    sph_mc = np.load(os.path.join(args.sph_dir, "train", "labels.npy"))
    sel = []
    for c in (1, 2, 3, 4):
        sel.append(np.where(sph_mc == c)[0])
    nsr_idx = np.where(sph_mc == 0)[0]
    if len(nsr_idx) > args.nsr_cap:
        nsr_idx = rng.permutation(nsr_idx)[:args.nsr_cap]
    sel.append(nsr_idx)
    sph_sel = np.sort(np.concatenate(sel))
    print(f"[SPH] train 선택 {len(sph_sel)} / {len(sph_mc)} (NSR cap={args.nsr_cap})")

    # ── 소스 로드 (CPSC 전량 + SPH 선택분) ──
    cs = np.load(os.path.join(args.cpsc_dir, "train", "signals.npy"))      # (Nc,12,5000) f32
    cm = np.load(os.path.join(args.cpsc_dir, "train", "labels.npy"))
    cb = np.load(os.path.join(args.cpsc_dir, "train", "labels_bin.npy"))
    ss_all = np.load(os.path.join(args.sph_dir, "train", "signals.npy"), mmap_mode="r")
    sm = np.load(os.path.join(args.sph_dir, "train", "labels.npy"))[sph_sel]
    sb = np.load(os.path.join(args.sph_dir, "train", "labels_bin.npy"))[sph_sel]

    nc, ns = len(cm), len(sph_sel)
    out_sig = np.empty((nc + ns, 12, 5000), dtype=np.float32)             # 사전할당(메모리 안전)
    out_sig[:nc] = cs
    del cs
    for i, j in enumerate(sph_sel):                                       # SPH는 선택분만 복사
        out_sig[nc + i] = np.asarray(ss_all[j], dtype=np.float32)
    out_mc = np.concatenate([cm, sm]).astype(np.int64)
    out_bin = np.concatenate([cb, sb]).astype(np.int64)

    d = os.path.join(args.out_dir, "train"); os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, "signals.npy"), out_sig)
    np.save(os.path.join(d, "labels.npy"), out_mc)
    np.save(os.path.join(d, "labels_bin.npy"), out_bin)

    # ── val/test = CPSC 그대로 복사 ──
    for s in ("val", "test"):
        sd = os.path.join(args.cpsc_dir, s); dd = os.path.join(args.out_dir, s)
        os.makedirs(dd, exist_ok=True)
        for fn in ("signals.npy", "labels.npy", "labels_bin.npy"):
            shutil.copy(os.path.join(sd, fn), os.path.join(dd, fn))

    # ── 분포 보고 ──
    dist = collections.Counter(out_mc.tolist())
    print(f"[결합 train] {nc + ns}개  (CPSC {nc} + SPH {ns})")
    names = ["NSR", "AF", "Ischemia", "Conduction", "Ectopic"]
    cdist = collections.Counter(cm.tolist()); sdist = collections.Counter(sm.tolist())
    print(f"  {'class':<12}{'CPSC':>7}{'SPH':>7}{'합계':>7}")
    for c in range(5):
        print(f"  [{c}] {names[c]:<8}{cdist.get(c,0):>7}{sdist.get(c,0):>7}{dist.get(c,0):>7}")
    print(f"  응급(bin=1): {int(out_bin.sum())} / {len(out_bin)}")
    print(f"  → {args.out_dir} (val/test=CPSC 복사)")


if __name__ == "__main__":
    main()
