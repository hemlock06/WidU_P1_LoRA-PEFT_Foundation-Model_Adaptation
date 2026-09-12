"""
PTB-XL 1.0.3 records500 acquisition via the AWS Open Data mirror, with verification.

Why a second downloader exists alongside ``download_ptbxl.py``
-------------------------------------------------------------
Measured on the lab PC, 2026-09-12:

  control endpoint (Cloudflare)      10_662_594 B/s
  physionet.org, 1 connection            23_075 B/s
  physionet.org, 6 connections           36_000 B/s aggregate
  physionet-open S3 mirror, 1 conn       96_436 B/s
  physionet-open S3 mirror, 6 conns     720_000 B/s aggregate

physionet.org throttles server-side and does not scale with parallelism, giving roughly a
20-hour transfer for records500. The AWS Open Data mirror is the same published project and
scales with connections. Bytes therefore come from the mirror; **integrity is never taken on
the mirror's word**.

Verification chain (all three are reported; none is assumed)
-----------------------------------------------------------
1. ``RECORDS`` is downloaded from the mirror and checked against the SHA-256 recorded in
   ``SHA256SUMS.txt``, which was fetched from physionet.org itself. This anchors the file
   list to the authoritative source.
2. The official ``SHA256SUMS.txt`` for 1.0.3 covers ``records100/`` and seven root files.
   It contains **no** ``records500/`` entries, so per-file official checksums are not
   available for the data this project uses. That absence is reported, not papered over.
3. Each downloaded record is verified in-band instead: every WFDB ``.hea`` stores a
   per-signal checksum (the 16-bit truncated sum of that lead's samples). The decoded
   ``.dat`` is recomputed against it. A record counts as verified only if every lead matches.

Additionally, ``--crosscheck-dir`` compares freshly downloaded bytes against records already
obtained directly from physionet.org in an earlier session, which measures mirror fidelity
directly rather than by assertion.

Output manifest records, per record: bytes, SHA-256 generated at receipt (required by the
acquisition standard when the provider ships no checksum), and the header-checksum verdict.

Usage:
  .venv/Scripts/python.exe scripts/download_ptbxl_mirror.py \
      --out-dir data/raw/ptbxl --workers 32 \
      --crosscheck-dir work/external_lineage_sources_20260910/ptbxl/v103
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

MIRROR = "https://physionet-open.s3.amazonaws.com/ptb-xl/1.0.3/"
AUTHORITATIVE = "https://physionet.org/files/ptb-xl/1.0.3/"

_lock = threading.Lock()
_stats = {"new": 0, "skip": 0, "fail": 0, "bytes": 0}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dst: Path, retries: int = 5, timeout: int = 60) -> tuple[bool, str]:
    """Download to a .part file and rename only on success.

    Writing straight to the final path is what makes an interrupted run indistinguishable
    from a complete one: the truncated file simply looks 'already present' to the next run.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_suffix(dst.suffix + ".part")
    delay = 1.0
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp, part.open("wb") as out:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
            part.replace(dst)
            return True, ""
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            if part.exists():
                try:
                    part.unlink()
                except OSError:
                    pass
            if attempt < retries - 1:
                time.sleep(delay)
                delay *= 2
            else:
                return False, f"{type(exc).__name__}: {exc}"
    return False, "max retries"


# --- WFDB header/signal verification -------------------------------------------------

_HEA_SIG = re.compile(
    r"^(?P<file>\S+)\s+(?P<fmt>\d+)(?:x\d+)?(?:\+\d+)?\s+"
    r"(?P<gain>[-\d.]+)(?:\((?P<baseline>[-\d]+)\))?(?:/\S+)?\s+"
    r"(?P<bitres>\d+)\s+(?P<zero>[-\d]+)\s+(?P<initval>[-\d]+)\s+(?P<checksum>[-\d]+)"
)


def parse_header(hea_path: Path):
    lines = [ln.strip() for ln in hea_path.read_text(errors="replace").splitlines()]
    lines = [ln for ln in lines if ln and not ln.startswith("#")]
    parts = lines[0].split()
    nsig, nsamp = int(parts[1]), int(parts[3])
    sigs = []
    for ln in lines[1 : 1 + nsig]:
        m = _HEA_SIG.match(ln)
        if not m:
            return None
        sigs.append({"fmt": int(m.group("fmt")), "checksum": int(m.group("checksum"))})
    return {"nsig": nsig, "nsamp": nsamp, "sigs": sigs}


def verify_record(hea_path: Path, dat_path: Path) -> tuple[bool, str]:
    """Recompute each lead's WFDB checksum from the .dat and compare against the .hea."""
    try:
        import numpy as np
    except ImportError:
        return False, "numpy unavailable"
    hdr = parse_header(hea_path)
    if hdr is None:
        return False, "header unparsed"
    if any(s["fmt"] != 16 for s in hdr["sigs"]):
        return False, f"unexpected format {[s['fmt'] for s in hdr['sigs']]}"
    raw = np.fromfile(dat_path, dtype="<i2")
    expected_n = hdr["nsig"] * hdr["nsamp"]
    if raw.size != expected_n:
        return False, f"sample count {raw.size} != {expected_n}"
    sig = raw.reshape(hdr["nsamp"], hdr["nsig"])
    for i, s in enumerate(hdr["sigs"]):
        # WFDB checksum: the sum of that lead's samples, wrapped to 16 bits. PTB-XL headers
        # are not consistent about the sign convention -- 15768_hr.hea writes 62102 while
        # HR03832.hea writes -24030 for the same kind of field -- so compare modulo 2**16
        # rather than picking one convention and silently failing half the database.
        total = int(sig[:, i].astype(np.int64).sum()) & 0xFFFF
        if (total - s["checksum"]) % (1 << 16) != 0:
            return False, f"lead {i} checksum {total} != {s['checksum']} (mod 2^16)"
    return True, "ok"


def do_record(rec: str, out_dir: Path, verify: bool):
    results = {}
    ok_all = True
    for ext in (".hea", ".dat"):
        dst = out_dir / (rec + ext)
        if dst.exists() and dst.stat().st_size > 0:
            with _lock:
                _stats["skip"] += 1
            results[ext] = "present"
            continue
        ok, err = fetch(MIRROR + rec + ext, dst)
        with _lock:
            if ok:
                _stats["new"] += 1
                _stats["bytes"] += dst.stat().st_size
            else:
                _stats["fail"] += 1
        results[ext] = "new" if ok else f"FAIL {err}"
        ok_all = ok_all and ok
    entry = {"record": rec, "files": results}
    if ok_all and verify:
        hea, dat = out_dir / (rec + ".hea"), out_dir / (rec + ".dat")
        good, why = verify_record(hea, dat)
        entry["header_checksum"] = "PASS" if good else f"FAIL {why}"
        entry["dat_bytes"] = dat.stat().st_size
        entry["dat_sha256"] = sha256_file(dat)
    elif not ok_all:
        entry["header_checksum"] = "NOT_ATTEMPTED_download_failed"
    return entry


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/raw/ptbxl")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--crosscheck-dir", default=None,
                    help="directory of records previously fetched from physionet.org directly")
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N records")
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # --- 1. authoritative checksum file ------------------------------------------
    sums_path = out_dir / "SHA256SUMS.txt"
    if not sums_path.exists():
        print("[1] fetching SHA256SUMS.txt from physionet.org (authoritative)")
        ok, err = fetch(AUTHORITATIVE + "SHA256SUMS.txt", sums_path, timeout=300)
        if not ok:
            print(f"    FAILED: {err}")
            return 2
    sums = {}
    for line in sums_path.read_text(errors="replace").splitlines():
        m = re.match(r"([0-9a-f]{64})\s+(.+)$", line.strip())
        if m:
            sums[m.group(2).strip()] = m.group(1)
    n500 = sum(1 for k in sums if k.startswith("records500/"))
    n100 = sum(1 for k in sums if k.startswith("records100/"))
    print(f"[1] SHA256SUMS.txt entries={len(sums)}  records100={n100}  records500={n500}")
    if n500 == 0:
        print("    NOTE: the provider ships no records500 checksums. Per-file SHA-256 is")
        print("    generated at receipt instead, and integrity is verified in-band against")
        print("    each WFDB header's per-lead checksum. Do not report this as")
        print("    'official checksum verified'.")

    # --- 2. RECORDS, verified against the authoritative sum ----------------------
    rec_path = out_dir / "RECORDS"
    if not rec_path.exists():
        ok, err = fetch(MIRROR + "RECORDS", rec_path, timeout=300)
        if not ok:
            print(f"[2] RECORDS download FAILED: {err}")
            return 2
    got = sha256_file(rec_path)
    exp = sums.get("RECORDS")
    verdict = "MATCH" if exp == got else ("NO_OFFICIAL_ENTRY" if exp is None else "MISMATCH")
    print(f"[2] RECORDS sha256={got[:16]}… official={str(exp)[:16]}… -> {verdict}")
    if verdict == "MISMATCH":
        print("    Mirror RECORDS disagrees with physionet.org. Stopping.")
        return 3

    all_recs = [ln.strip() for ln in rec_path.read_text().splitlines() if ln.strip()]
    recs500 = [r for r in all_recs if r.startswith("records500/")]
    if args.limit:
        recs500 = recs500[: args.limit]
    print(f"[3] records500 to acquire: {len(recs500)}  (workers={args.workers})")

    # --- 3. parallel download + in-band verification -----------------------------
    entries = []
    done = 0
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(do_record, r, out_dir, not args.no_verify): r for r in recs500}
        for fut in cf.as_completed(futs):
            entries.append(fut.result())
            done += 1
            if done % 1000 == 0 or done == len(recs500):
                el = time.time() - t0
                rate = _stats["bytes"] / el if el > 0 else 0
                print(f"    {done}/{len(recs500)}  new={_stats['new']} skip={_stats['skip']} "
                      f"fail={_stats['fail']}  {rate/1e6:.2f} MB/s  {el:.0f}s")

    passed = sum(1 for e in entries if e.get("header_checksum") == "PASS")
    failed = [e for e in entries if str(e.get("header_checksum", "")).startswith("FAIL")]
    notatt = [e for e in entries if str(e.get("header_checksum", "")).startswith("NOT_ATTEMPTED")]
    print(f"[4] header-checksum verification: PASS={passed}  FAIL={len(failed)}  "
          f"NOT_ATTEMPTED={len(notatt)}  of {len(entries)}")
    for e in failed[:10]:
        print(f"      FAIL {e['record']}: {e['header_checksum']}")
    for e in notatt[:10]:
        print(f"      NOT_ATTEMPTED {e['record']}: {e['files']}")

    # --- 4. mirror-fidelity cross-check against physionet.org-sourced bytes ------
    cross = {"checked": 0, "identical": 0, "differing": [], "absent": 0}
    if args.crosscheck_dir:
        cdir = Path(args.crosscheck_dir)
        for hea in sorted(cdir.glob("*_hr.hea")):
            stem = hea.stem  # e.g. 15768_hr
            eid = int(stem.split("_")[0])
            sub = f"{eid // 1000 * 1000:05d}"
            cand = out_dir / "records500" / sub / f"{stem}.dat"
            ref = cdir / f"{stem}.dat"
            if not cand.exists() or not ref.exists():
                cross["absent"] += 1
                continue
            cross["checked"] += 1
            if sha256_file(cand) == sha256_file(ref):
                cross["identical"] += 1
            else:
                cross["differing"].append(stem)
        print(f"[5] mirror fidelity vs physionet.org-sourced files: checked={cross['checked']} "
              f"identical={cross['identical']} differing={len(cross['differing'])} "
              f"absent={cross['absent']}")
        if cross["differing"]:
            print(f"      DIFFERING: {cross['differing'][:10]}")

    manifest = {
        "source_bytes": MIRROR,
        "authoritative_checksums": AUTHORITATIVE + "SHA256SUMS.txt",
        "records500_official_checksums_available": n500 > 0,
        "records_index_verdict": verdict,
        "requested": len(recs500),
        "new": _stats["new"], "skipped": _stats["skip"], "failed": _stats["fail"],
        "header_checksum_pass": passed,
        "header_checksum_fail": [e["record"] for e in failed],
        "header_checksum_not_attempted": [e["record"] for e in notatt],
        "mirror_crosscheck": cross,
        "elapsed_sec": round(time.time() - t0, 2),
        "entries": entries,
    }
    mpath = Path(args.manifest) if args.manifest else out_dir / "acquisition_manifest.json"
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(manifest, indent=1))
    print(f"[6] manifest -> {mpath}")

    complete = (_stats["fail"] == 0 and not failed and not notatt
                and len(entries) == len(recs500))
    print(f"[7] verdict: {'verified_complete' if complete else 'partial/unverified'}")
    return 0 if complete else 4


if __name__ == "__main__":
    raise SystemExit(main())
