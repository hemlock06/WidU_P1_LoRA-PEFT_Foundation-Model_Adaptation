"""Rebuild historical CPSC test signals without replacing any recovered array.

The original test identifiers define order; this is a preprocessing control, not
a new split or model evaluation. Validation reconstruction remains a separate gate.
"""
import argparse
import hashlib
import json
import platform
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from preprocess_cpsc2018_mc import load_signal, parse_dx_from_hea

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def historical_label(codes):
    # Stage5b code sets, before the May29 multi-label/164884008 expansion.
    for label, group in ((2, {429622005, 164931005}), (1, {164889003}),
                         (3, {270492004, 164909002, 59118001}),
                         (4, {284470004, 63593006, 427172004, 17338001}),
                         (0, {426783006})):
        if codes & group:
            return label
    return -1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.out_dir.resolve()
    # Refuse reuse, so failure evidence cannot be overwritten by another attempt.
    out.mkdir(parents=True, exist_ok=False)
    original = ROOT / "data/processed/cpsc2018_mc/test"
    raw = ROOT / "data/raw/cpsc2018"
    start = time.monotonic()
    report = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "python": platform.python_version(), "numpy": np.__version__,
              "status": "FAIL", "failures": [], "skipped": 0,
              "control_only": True, "validation_reconstructed": False,
              "id_provenance": "Recovered original test IDs retained, not independently regenerated",
              "sources": {p.name: sha(p) for p in original.glob("*.npy")},
              "code_sha256": {p.name: sha(p) for p in
                 (Path(__file__), ROOT / "scripts/preprocess_cpsc2018_mc.py")}}
    try:
        # The local recovered, hash-recorded object array is the historical ID asset.
        ids = np.load(original / "record_ids.npy", allow_pickle=True)
        reference = np.load(original / "signals.npy", mmap_mode="r")
        expected_labels = np.load(original / "labels.npy")
        if ids.shape != (936,) or len(set(ids)) != 936:
            raise ValueError("Historical936 unique-ID contract failed")
        if reference.shape != (936, 12, 5000) or reference.dtype != np.float32:
            raise ValueError("Historical signal shape/dtype contract failed")
        lookup = {}
        for header in raw.rglob("*.hea"):
            if header.stem in lookup:
                raise ValueError(f"Duplicate raw ID: {header.stem}")
            lookup[header.stem] = header
        signals, labels, inventory = [], [], []
        for index, record_id in enumerate(ids):
            header = lookup.get(str(record_id))
            if header is None:
                report["failures"].append({"id": str(record_id), "error": "missing header"})
                continue
            signal = load_signal(str(header))
            if signal is None:
                report["failures"].append({"id": str(record_id), "error": "load_signal rejected record"})
                continue
            label = historical_label(parse_dx_from_hea(str(header)))
            signals.append(signal)
            labels.append(label)
            inventory.append({"record_id": str(record_id), "index": index,
                              "hea_sha256": sha(header),
                              "mat_sha256": sha(header.with_suffix(".mat")),
                              "signal_equal": bool(np.array_equal(signal, reference[index])),
                              "label_equal": bool(label == expected_labels[index]),
                              "max_abs_difference": float(np.max(np.abs(signal-reference[index])))})
        (out / "raw_inventory.json").write_text(json.dumps(inventory, indent=2)+"\n")
        if report["failures"]:
            raise ValueError("Missing/rejected records; no skipped records are permitted")
        np.save(out / "signals.npy", np.stack(signals))
        np.save(out / "labels.npy", np.asarray(labels, dtype=np.int8))
        np.save(out / "labels_bin.npy", np.isin(labels, [1, 2]).astype(np.int8))
        shutil.copy2(original / "record_ids.npy", out / "record_ids.npy")
        report["comparisons"] = {name: {"original_sha256": sha(original/name),
                 "reconstructed_sha256": sha(out/name),
                 "byte_exact": sha(original/name) == sha(out/name)}
                 for name in ("signals.npy", "labels.npy", "labels_bin.npy", "record_ids.npy")}
        report["records"] = len(inventory)
        report["signal_mismatches"] = sum(not row["signal_equal"] for row in inventory)
        report["label_mismatches"] = sum(not row["label_equal"] for row in inventory)
        report["max_abs_difference"] = max(row["max_abs_difference"] for row in inventory)
        if not all(row["byte_exact"] for row in report["comparisons"].values()):
            raise ValueError("Byte-exact reconstruction control failed; external inference prohibited")
        report["status"] = "PASS_TEST_RECONSTRUCTION_ONLY"
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        report["error"] = repr(exc)
    report["elapsed_seconds"] = time.monotonic() - start
    (out / "control.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))
    if report["status"] == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
