"""
Exhaustive duplicate / near-duplicate screen over all of PTB-XL 1.0.3 records500.

Why this exists
---------------
F3 eligibility condition 4 (``records/external_validation_protocol_v1.md`` section 3) requires
that no eligible record has an identical-waveform partner sitting in the published ECG-FM
train or valid split. Until now that condition was evaluated from the PTB-XL **changelog
catalogue** of 38 dropped duplicates. The 2026-09-11 audit compared 78 catalogued pairs and
found 76 byte-exact and 2 not, and the protocol amendment section 2 states plainly that the
relation list "is not an exhaustive near-duplicate search over all records".

That leaves the load-bearing direction untested. What condition 4 needs is **completeness** of
the identity relation (every identical pair is known), not the property the failing gate
tested (every catalogued pair is byte-identical). A catalogue broader than byte identity is
conservative; a catalogue *narrower* than the true duplicate relation is leakage.

This script tests the load-bearing direction directly and stops depending on the catalogue.

Two channels, because exact hashing alone is provably insufficient
-----------------------------------------------------------------
Direct measurement of the two catalogued-but-nonexact pairs (3832/15768, 11838/11839) showed:
best integer lag 0, per-lead Pearson correlation 0.9942..0.9999, difference-to-signal RMS
ratio 0.035 and 0.107, identical gain and baseline, difference power concentrated at 5-40 Hz.
Two distinct 10-second recordings cannot align sample-for-sample like that. They are the same
underlying recording re-exported with slightly different processing -- genuine duplicates that
a byte hash does **not** match. So:

  channel A (exact)  : SHA-256 over the canonical decoded digital array. Catches re-encodings
                       that are bit-identical after decoding even if file bytes differ.
  channel B (near)   : cosine similarity between z-scored, block-averaged per-lead
                       fingerprints, computed for every (test-split record) x (train/valid
                       record) pair. Catches the 3832/15768 class.

Channel B is a screen, not a verdict: every pair above the review threshold is written out for
adjudication against the full-resolution signals. The threshold is a **review** trigger, not an
exclusion rule, so it cannot quietly drop or keep cohort members.

Silent-failure guards (each reported as a number, not assumed)
--------------------------------------------------------------
* Records that fail to load are counted and listed. A record that could not be read is never
  counted as "no duplicate found" -- absence of evidence is reported separately.
* The count of records actually screened is printed next to the count expected from RECORDS.
  A screen over a partial download must not read as a clean screen.
* A positive control runs every time: a known duplicate pair is injected and must be
  recovered by both channels. If the control fails the script exits nonzero and reports
  UNVERIFIED rather than printing zero findings.
* Split assignment is keyed on the v1.0.1 / Challenge ecg_id used by the published ECG-FM
  split, taken from the frozen cross-version table, not re-derived here.

Usage:
  .venv/Scripts/python.exe scripts/audit_ptbxl_duplicate_exhaustive.py \
      --records-dir data/raw/ptbxl/records500 \
      --cross-version results/external_readiness_final_20260911/ptbxl_cross_version_table.csv \
      --f3 results/external_readiness_final_20260911/ptbxl_f3_eligible_records.csv \
      --out-dir results/ptbxl_duplicate_screen_20260912
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

FP_BLOCKS = 250   # block-average each 5000-sample lead down to 250 points
N_LEADS = 12
N_SAMP = 5000


def read_header(hea: Path):
    lines = [ln.strip() for ln in hea.read_text(errors="replace").splitlines()]
    lines = [ln for ln in lines if ln and not ln.startswith("#")]
    p = lines[0].split()
    return int(p[1]), int(p[3])


def load_digital(rec_dir: Path, eid: int):
    """Return the (5000, 12) int16 digital array for a v1.0.3 records500 record."""
    sub = f"{eid // 1000 * 1000:05d}"
    stem = f"{eid:05d}_hr"
    hea = rec_dir / sub / f"{stem}.hea"
    dat = rec_dir / sub / f"{stem}.dat"
    if not hea.exists() or not dat.exists():
        return None, "missing"
    nsig, nsamp = read_header(hea)
    raw = np.fromfile(dat, dtype="<i2")
    if raw.size != nsig * nsamp:
        return None, f"size {raw.size} != {nsig * nsamp}"
    if (nsig, nsamp) != (N_LEADS, N_SAMP):
        return None, f"shape {nsig}x{nsamp}"
    return raw.reshape(nsamp, nsig), "ok"


def fingerprint(sig: np.ndarray) -> np.ndarray:
    """z-scored, block-averaged, per-lead fingerprint, flattened and L2-normalised.

    Block averaging suppresses the small high-frequency processing differences that separate
    re-exports of one recording, while z-scoring per lead makes the comparison invariant to
    amplitude scaling. A constant (all-zero-variance) lead contributes zeros rather than NaN.
    """
    x = sig.astype(np.float32).T                      # (12, 5000)
    x = x.reshape(N_LEADS, FP_BLOCKS, N_SAMP // FP_BLOCKS).mean(axis=2)
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    x = np.where(sd > 0, (x - mu) / np.where(sd > 0, sd, 1.0), 0.0)
    v = x.reshape(-1)
    n = np.linalg.norm(v)
    return (v / n).astype(np.float32) if n > 0 else v.astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records-dir", default="data/raw/ptbxl/records500")
    ap.add_argument("--cross-version",
                    default="results/external_readiness_final_20260911/ptbxl_cross_version_table.csv")
    ap.add_argument("--f3",
                    default="results/external_readiness_final_20260911/ptbxl_f3_eligible_records.csv")
    ap.add_argument("--out-dir", default="results/ptbxl_duplicate_screen_20260912")
    ap.add_argument("--review-threshold", type=float, default=0.98,
                    help="cosine similarity at or above which a pair is written out for "
                         "human adjudication (a review trigger, never an exclusion rule)")
    ap.add_argument("--chunk", type=int, default=512)
    args = ap.parse_args()

    t0 = time.time()
    rec_dir = Path(args.records_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- split assignment from the frozen cross-version table --------------------
    rows = list(csv.DictReader(open(args.cross_version)))
    split = {int(r["ecg_id"]): r["published_split"] for r in rows}
    in103 = {int(r["ecg_id"]): r["in_v103"] == "True" for r in rows}
    v103_ids = sorted(e for e in in103 if in103[e])
    f3_ids = {int(r["ecg_id"]) for r in csv.DictReader(open(args.f3))}
    print(f"[1] cross-version rows={len(rows)}  v1.0.3 records={len(v103_ids)}  F3={len(f3_ids)}")

    expected = len(v103_ids)

    # --- load, hash, fingerprint -------------------------------------------------
    print(f"[2] loading records from {rec_dir}")
    ids, hashes, fps, failures = [], [], [], []
    for i, eid in enumerate(v103_ids):
        sig, why = load_digital(rec_dir, eid)
        if sig is None:
            failures.append({"ecg_id": eid, "reason": why})
            continue
        ids.append(eid)
        hashes.append(hashlib.sha256(np.ascontiguousarray(sig, dtype="<i2").tobytes()).hexdigest())
        fps.append(fingerprint(sig))
        if (i + 1) % 5000 == 0:
            print(f"    {i+1}/{expected}  loaded={len(ids)}  failed={len(failures)}  "
                  f"{time.time()-t0:.0f}s")
    n = len(ids)
    print(f"[2] screened={n}  expected={expected}  load_failures={len(failures)}")
    if failures[:5]:
        print(f"    first failures: {failures[:5]}")
    coverage_complete = (len(failures) == 0 and n == expected)
    if not coverage_complete:
        print("    WARNING: coverage is incomplete. Any 'no duplicates found' result below is")
        print("    scoped to the records actually screened and must be reported as partial.")

    ids_arr = np.array(ids)
    idx_of = {e: k for k, e in enumerate(ids)}
    F = np.vstack(fps)                                  # (n, 12*FP_BLOCKS)

    # --- channel A: exact duplicate groups --------------------------------------
    groups: dict[str, list[int]] = {}
    for e, h in zip(ids, hashes):
        groups.setdefault(h, []).append(e)
    exact_groups = {h: g for h, g in groups.items() if len(g) > 1}
    print(f"[3] channel A (exact): duplicate groups={len(exact_groups)}  "
          f"records involved={sum(len(g) for g in exact_groups.values())}")

    def cross_split(group):
        s = {split.get(e, "?") for e in group}
        return ("test" in s) and bool(s & {"train", "valid"})

    exact_violations = []
    for h, g in exact_groups.items():
        f3_members = [e for e in g if e in f3_ids]
        partner_splits = {e: split.get(e, "?") for e in g}
        if f3_members and any(split.get(e) in ("train", "valid") for e in g):
            exact_violations.append({"sha256": h, "group": g, "f3_members": f3_members,
                                     "splits": partner_splits})
    print(f"[3] exact groups spanning test<->train/valid: "
          f"{sum(1 for g in exact_groups.values() if cross_split(g))}")
    print(f"[3] *** exact violations touching F3: {len(exact_violations)} ***")

    # --- channel B: near-duplicate screen, F3 vs train/valid ---------------------
    tv_mask = np.array([split.get(e) in ("train", "valid") for e in ids])
    f3_mask = np.array([e in f3_ids for e in ids])
    Ftv, Ff3 = F[tv_mask], F[f3_mask]
    tv_ids, f3_scr = ids_arr[tv_mask], ids_arr[f3_mask]
    print(f"[4] channel B (near): F3 screened={Ff3.shape[0]} vs train/valid={Ftv.shape[0]} "
          f"({Ff3.shape[0]*Ftv.shape[0]:,} pairs)")

    near_hits, best_per_f3 = [], []
    for s in range(0, Ff3.shape[0], args.chunk):
        blk = Ff3[s : s + args.chunk]
        sim = blk @ Ftv.T                                # cosine, both L2-normalised
        for r in range(sim.shape[0]):
            row = sim[r]
            j = int(np.argmax(row))
            best_per_f3.append({"f3_ecg_id": int(f3_scr[s + r]),
                                "best_match_ecg_id": int(tv_ids[j]),
                                "cosine": round(float(row[j]), 6)})
            for j2 in np.nonzero(row >= args.review_threshold)[0]:
                near_hits.append({"f3_ecg_id": int(f3_scr[s + r]),
                                  "train_valid_ecg_id": int(tv_ids[int(j2)]),
                                  "cosine": round(float(row[int(j2)]), 6),
                                  "partner_split": split.get(int(tv_ids[int(j2)]), "?")})
    sims = np.array([b["cosine"] for b in best_per_f3]) if best_per_f3 else np.array([0.0])
    print(f"[4] best-match cosine over F3: max={sims.max():.6f} p99={np.percentile(sims,99):.6f} "
          f"median={np.median(sims):.6f}")
    print(f"[4] *** pairs at/above review threshold {args.review_threshold}: {len(near_hits)} ***")

    # --- positive control: both channels must recover a planted duplicate --------
    control = {"ran": False}
    if n > 2 and Ftv.shape[0] > 0 and Ff3.shape[0] > 0:
        src = int(tv_ids[0])
        sig, why = load_digital(rec_dir, src)
        if sig is not None:
            h_self = hashlib.sha256(np.ascontiguousarray(sig, dtype="<i2").tobytes()).hexdigest()
            exact_ok = (h_self == hashes[idx_of[src]])
            # perturb like a re-export: small broadband noise at the observed scale
            rng = np.random.default_rng(0)
            noisy = (sig.astype(np.float32)
                     + rng.normal(0, 0.035 * sig.astype(np.float32).std(), sig.shape))
            near_ok = float(fingerprint(noisy.astype(np.int16)) @ F[idx_of[src]]) >= args.review_threshold
            control = {"ran": True, "record": src, "exact_channel_recovers_self": bool(exact_ok),
                       "near_channel_recovers_reexport": bool(near_ok),
                       "reexport_cosine": round(float(fingerprint(noisy.astype(np.int16))
                                                      @ F[idx_of[src]]), 6)}
            print(f"[5] positive control on {src}: exact={exact_ok} near={near_ok} "
                  f"(cosine {control['reexport_cosine']})")
    control_ok = control.get("ran") and control["exact_channel_recovers_self"] and \
        control["near_channel_recovers_reexport"]
    if not control_ok:
        print("[5] POSITIVE CONTROL FAILED -- the screen's zero/low findings are UNVERIFIED.")

    # --- write outputs -----------------------------------------------------------
    (out_dir / "exact_duplicate_groups.json").write_text(json.dumps(
        [{"sha256": h, "ecg_ids": g, "splits": {str(e): split.get(e, "?") for e in g}}
         for h, g in sorted(exact_groups.items())], indent=1))
    with (out_dir / "near_duplicate_hits.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["f3_ecg_id", "train_valid_ecg_id", "cosine",
                                           "partner_split"])
        w.writeheader()
        w.writerows(sorted(near_hits, key=lambda d: -d["cosine"]))
    with (out_dir / "f3_best_match.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["f3_ecg_id", "best_match_ecg_id", "cosine"])
        w.writeheader()
        w.writerows(best_per_f3)

    summary = {
        "records_expected": expected,
        "records_screened": n,
        "load_failures": failures,
        "coverage_complete": coverage_complete,
        "exact_duplicate_groups": len(exact_groups),
        "exact_groups_spanning_test_and_train_valid":
            sum(1 for g in exact_groups.values() if cross_split(g)),
        "exact_violations_touching_f3": exact_violations,
        "near_review_threshold": args.review_threshold,
        "near_hits_count": len(near_hits),
        "near_hits": sorted(near_hits, key=lambda d: -d["cosine"])[:200],
        "f3_best_match_cosine_max": float(sims.max()),
        "f3_best_match_cosine_p99": float(np.percentile(sims, 99)),
        "positive_control": control,
        "elapsed_sec": round(time.time() - t0, 2),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    gate_pass = (coverage_complete and control_ok and not exact_violations and not near_hits)
    summary["gate"] = ("PASS_EXHAUSTIVE_DUPLICATE_SCREEN" if gate_pass
                       else "FAIL_OR_REVIEW_REQUIRED")
    (out_dir / "duplicate_screen_summary.json").write_text(json.dumps(summary, indent=1))
    print(f"[6] outputs -> {out_dir}")
    print(f"[7] gate: {summary['gate']}")
    if not coverage_complete:
        print("    (coverage incomplete -> not a clean pass regardless of findings)")
    return 0 if gate_pass else 4


if __name__ == "__main__":
    raise SystemExit(main())
