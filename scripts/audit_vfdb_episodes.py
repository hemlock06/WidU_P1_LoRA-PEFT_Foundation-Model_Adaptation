"""Build VFDB rhythm episodes from raw markers under the official NOISE rule.

Official statement (PhysioNet vfdb/1.0.0): "The rhythm change annotations are
placed at the beginning of the episode of the indicated rhythm. The previous
rhythm continues during episodes marked by (NOISE; the noise ends at the time of
the next annotation." No beat labels exist. No model inference is performed here;
this fixes candidate episode/window rules and counts before any training design.
"""

import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
READY = ROOT / "results/vfdb_readiness_20260910"
OUT = ROOT / "results/vfdb_episodes_20260910"
FS = 250
WINDOW = 10 * FS
TARGET_SETS = {
    "VT_only": {"VT"},
    "VF_VFIB_VFL": {"VF", "VFIB", "VFL"},
    "malignant_ventricular_VT_VFL_VF_VFIB": {"VT", "VFL", "VF", "VFIB"},
    "ASYS_only": {"ASYS"},
}
DURATION_GRID_S = (2, 5, 10, 30)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_episodes(markers, sig_len):
    """markers: list of (sample, label) sorted by sample; label without '('.

    Returns (rhythm_spans, noise_spans). A NOISE marker opens a noise span that
    ends at the next marker and leaves the rhythm state unchanged. A rhythm marker
    sets the state from its sample until the next rhythm marker. Samples before the
    first rhythm marker are UNLABELED_START.
    """
    rhythm_spans, noise_spans = [], []
    state, state_start = "UNLABELED_START", 0
    for idx, (sample, label) in enumerate(markers):
        nxt = markers[idx + 1][0] if idx + 1 < len(markers) else sig_len
        if label == "NOISE":
            noise_spans.append((sample, nxt))
            continue
        if sample > state_start:
            rhythm_spans.append((state_start, sample, state))
        state, state_start = label, sample
    if sig_len > state_start:
        rhythm_spans.append((state_start, sig_len, state))
    merged = []
    for span in rhythm_spans:
        if merged and merged[-1][2] == span[2] and merged[-1][1] == span[0]:
            merged[-1] = (merged[-1][0], span[1], span[2])
        else:
            merged.append(span)
    return merged, noise_spans


def overlap(a0, a1, b0, b1):
    return max(0, min(a1, b1) - max(a0, b0))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ann = pd.read_csv(READY / "raw_annotations.csv")
    inv = pd.read_csv(READY / "record_inventory.csv")
    sig_len = dict(zip(inv.record_id.astype(str), inv.samples))
    if not (ann.symbol == "+").all():
        raise ValueError("VFDB annotations must all be rhythm-change markers")
    episodes, windows = [], []
    per_record = []
    for rec, group in ann.groupby("record_id"):
        rec = str(rec)
        group = group.sort_values("sample")
        markers = [(int(s), str(a)[1:]) for s, a in zip(group["sample"], group.aux_note)]
        if any(not str(a).startswith("(") for a in group.aux_note):
            raise ValueError(f"{rec}: aux note without '(' prefix")
        spans, noise = build_episodes(markers, sig_len[rec])
        for start, end, label in spans:
            noise_in = sum(overlap(start, end, n0, n1) for n0, n1 in noise)
            episodes.append({"record_id": rec, "rhythm": label, "start_sample": start,
                             "end_sample": end, "start_s": start / FS, "end_s": end / FS,
                             "duration_s": (end - start) / FS, "noise_s_inside": noise_in / FS,
                             "contains_noise": noise_in > 0})
        n_windows = sig_len[rec] // WINDOW
        for w in range(n_windows):
            w0, w1 = w * WINDOW, (w + 1) * WINDOW
            frac = Counter()
            for start, end, label in spans:
                frac[label] += overlap(start, end, w0, w1) / WINDOW
            noise_frac = sum(overlap(w0, w1, n0, n1) for n0, n1 in noise) / WINDOW
            dominant = max(frac.items(), key=lambda kv: kv[1])[0]
            row = {"record_id": rec, "window": w, "start_s": w0 / FS, "dominant_rhythm": dominant,
                   "dominant_fraction": round(frac[dominant], 4), "noise_fraction": round(noise_frac, 4),
                   "unlabeled_fraction": round(frac.get("UNLABELED_START", 0.0), 4)}
            for name, labels in TARGET_SETS.items():
                row[f"{name}_fraction"] = round(sum(frac[k] for k in labels), 4)
            windows.append(row)
        per_record.append({"record_id": rec, "markers": len(markers),
                           "noise_spans": len(noise), "noise_s": sum((b - a) for a, b in noise) / FS,
                           "unlabeled_start_s": next((e - s for s, e, l in spans if l == "UNLABELED_START"), 0) / FS,
                           "rhythms": ";".join(sorted({l for _, _, l in spans}))})
    ep = pd.DataFrame(episodes)
    win = pd.DataFrame(windows)
    ep.to_csv(OUT / "episodes.csv", index=False)
    win.to_csv(OUT / "window_labels_10s.csv", index=False)
    pd.DataFrame(per_record).to_csv(OUT / "record_summary.csv", index=False)

    summary = {}
    for name, labels in TARGET_SETS.items():
        sel = ep[ep.rhythm.isin(labels)]
        summary[name] = {
            "episodes": len(sel),
            "records_with_episode": int(sel.record_id.nunique()),
            "total_seconds": float(sel.duration_s.sum()),
            "median_seconds": float(sel.duration_s.median()) if len(sel) else None,
            "episodes_with_noise_inside": int(sel.contains_noise.sum()),
            "episodes_at_least": {f"{d}s": int((sel.duration_s >= d).sum()) for d in DURATION_GRID_S},
            "windows_fraction_1.0": int((win[f"{name}_fraction"] >= 0.999).sum()),
            "windows_fraction_ge_0.5": int((win[f"{name}_fraction"] >= 0.5).sum()),
            "windows_fraction_between_0_and_0.5": int(((win[f"{name}_fraction"] > 0) & (win[f"{name}_fraction"] < 0.5)).sum()),
        }
    negatives = win[(win.dominant_rhythm.isin(["N", "NSR"])) & (win.dominant_fraction >= 0.999)
                    & (win.noise_fraction == 0)]
    result = {
        "status": "episode_and_window_rules_audited_not_a_model_result",
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "rule": "Rhythm marker sets state until next rhythm marker; NOISE keeps the prior rhythm and ends at the next marker; pre-first-marker samples are UNLABELED_START.",
        "records": int(ann.record_id.nunique()), "markers": len(ann),
        "rhythm_episodes_total": len(ep),
        "episodes_by_rhythm": {k: int(v) for k, v in ep.rhythm.value_counts().sort_index().items()},
        "seconds_by_rhythm": {k: float(v) for k, v in ep.groupby("rhythm").duration_s.sum().sort_index().items()},
        "noise_spans": int(sum(r["noise_spans"] for r in per_record)),
        "noise_seconds": float(sum(r["noise_s"] for r in per_record)),
        "records_with_unlabeled_start": int(sum(r["unlabeled_start_s"] > 0 for r in per_record)),
        "unlabeled_start_seconds_total": float(sum(r["unlabeled_start_s"] for r in per_record)),
        "windows_10s_total": len(win),
        "pure_sinus_noise_free_windows": len(negatives),
        "records_with_pure_sinus_windows": int(negatives.record_id.nunique()),
        "target_set_summaries": summary,
        "patient_identity": "VFDB provides no subject identifiers; record-level grouping only, and distinct-subject status across records is unverified.",
        "limitations": [
            "Marker-derived episodes are not adjudicated clinical events.",
            "Window fractions depend on the fixed 10 s grid; other grids change counts.",
            "Generic ECG/ECG channels cannot be mapped to standard leads.",
            "No pulse/hemodynamic information: VT here is an ECG rhythm label only.",
        ],
        "source_hashes": {"raw_annotations.csv": sha(READY / "raw_annotations.csv"),
                          "record_inventory.csv": sha(READY / "record_inventory.csv")},
        "script_sha256": sha(Path(__file__)),
        "model_inferences": 0, "new_training_runs": 0,
    }
    (OUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    with (OUT / "target_set_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["target_set", "episodes", "records", "total_seconds", "ge_10s", "ge_30s",
                         "windows_full", "windows_ge_half"])
        for name, s in summary.items():
            writer.writerow([name, s["episodes"], s["records_with_episode"], round(s["total_seconds"], 1),
                             s["episodes_at_least"]["10s"], s["episodes_at_least"]["30s"],
                             s["windows_fraction_1.0"], s["windows_fraction_ge_0.5"]])
    print(json.dumps({k: result[k] for k in (
        "episodes_by_rhythm", "noise_spans", "records_with_unlabeled_start",
        "unlabeled_start_seconds_total", "windows_10s_total", "pure_sinus_noise_free_windows",
        "records_with_pure_sinus_windows", "target_set_summaries")}, indent=2))


if __name__ == "__main__":
    main()
