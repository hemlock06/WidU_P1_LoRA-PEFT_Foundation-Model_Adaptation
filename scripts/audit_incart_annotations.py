"""INCART original-versus-Challenge scope, annotation content and prior-preprocess replay.

Uses only official header, annotation and metadata files (no waveform .dat, no
model inference). Replays the historical `preprocess_incart.py` window labelling
from annotations alone to state exactly what the earlier binary evaluation used.
"""

import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import wfdb

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "work/external_lineage_sources_20260910/incart"
DX = ROOT / "work/external_lineage_sources_20260910/challenge_dx"
OUT = ROOT / "results/external_cohort_lineage_20260910"
FS = 257
WINDOW_SAMPLES_257 = 2570  # 10 s at 257 Hz; equals the historical 5000 @ 500 Hz window.
FS_OUT, SEG_LEN_OUT = 500, 5000
EMERGENCY_RHYTHMS = {"(AFIB", "(WPWAF"}
NORMAL_N_RATIO = 0.90
BEAT_NAMES = {"N": "normal", "V": "pvc", "A": "apc", "F": "fusion", "R": "rbbb_beat",
              "L": "lbbb_beat", "j": "junctional_escape", "n": "supraventricular_escape",
              "S": "supraventricular_premature", "E": "ventricular_escape",
              "Q": "unclassifiable", "|": "isolated_artifact", "+": "rhythm_change",
              "~": "signal_quality_change", "x": "non_conducted_p", "B": "bbb_beat",
              "a": "aberrated_apc", "J": "nodal_premature", "e": "atrial_escape",
              "f": "fusion_paced_normal", "/": "paced"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_patients(text):
    patients, current = {}, None
    lines = [line.rstrip() for line in text.splitlines()]
    i = 0
    while i < len(lines):
        m = re.match(r"^patient (\d+)$", lines[i])
        if m:
            current = int(m.group(1))
            recs = lines[i + 1].split()
            diag = lines[i + 2].strip() if i + 2 < len(lines) and not lines[i + 2].startswith("patient") else ""
            patients[current] = {"records": recs, "diagnosis": diag}
            i += 3 if diag else 2
            continue
        i += 1
    return patients


def parse_orig_header(path):
    text = path.read_text(errors="replace")
    age = sex = None
    diagnoses = ""
    patient = None
    description = ""
    for line in text.splitlines():
        if line.startswith("#<age>"):
            m = re.search(r"<age>:\s*(\d+)\s*<sex>:\s*(\w)", line)
            if m:
                age, sex = int(m.group(1)), m.group(2)
            d = re.search(r"<diagnoses>\s*(.*)$", line)
            if d:
                diagnoses = d.group(1).strip()
        elif line.startswith("# patient"):
            patient = int(line.split()[-1])
        elif line.startswith("#"):
            description = line[1:].strip()
    return age, sex, diagnoses, patient, description


def parse_challenge_header(path):
    fields = {}
    for line in path.read_text().splitlines():
        if line.startswith("# "):
            key, _, value = line[2:].partition(":")
            fields[key.strip()] = value.strip()
    return fields


def rhythm_intervals(ann, sig_len):
    """Rhythm '+' markers persist until the next '+' marker (no NOISE special case in INCART)."""
    marks = [(int(s), a.rstrip("\x00")) for s, sym, a in zip(ann.sample, ann.symbol, ann.aux_note)
             if sym == "+"]
    intervals = []
    for idx, (start, label) in enumerate(marks):
        end = marks[idx + 1][0] if idx + 1 < len(marks) else sig_len
        intervals.append((start, end, label))
    return intervals


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    sums = {}
    for line in (SRC / "SHA256SUMS.txt").read_text().splitlines():
        digest, name = line.split()
        sums[name] = digest
    verified = 0
    for path in sorted((SRC / "orig").glob("*")):
        digest = sha(path)
        if sums.get(path.name) != digest:
            raise ValueError(f"Official SHA256SUMS mismatch for {path.name}")
        verified += 1
        manifest.append({"path": str(path.relative_to(ROOT)).replace("\\", "/"),
                         "bytes": path.stat().st_size, "sha256": digest,
                         "official_sha256_verified": True})
    for path in sorted(SRC.glob("*")):
        if path.is_file():
            manifest.append({"path": str(path.relative_to(ROOT)).replace("\\", "/"),
                             "bytes": path.stat().st_size, "sha256": sha(path),
                             "official_sha256_verified": sums.get(path.name) == sha(path)})
    for path in sorted((SRC / "challenge").glob("*")) + sorted(DX.glob("*.csv")):
        manifest.append({"path": str(path.relative_to(ROOT)).replace("\\", "/"),
                         "bytes": path.stat().st_size, "sha256": sha(path),
                         "official_sha256_verified": False})

    orig_records = (SRC / "RECORDS").read_text().split()
    chal_records = (SRC / "challenge/RECORDS").read_text().split()
    chal_to_orig = {name: "I" + name[3:] for name in chal_records}
    if len(orig_records) != 75 or len(chal_records) != 74:
        raise ValueError("Unexpected record counts")
    missing_from_challenge = sorted(set(orig_records) - set(chal_to_orig.values()))
    patients = parse_patients((SRC / "files-patients-diagnoses.txt").read_text())
    record_patient = {r: p for p, info in patients.items() for r in info["records"]}
    if set(record_patient) != set(orig_records) or len(patients) != 32:
        raise ValueError("Patient-record mapping does not cover 75 records / 32 patients")

    dx = pd.concat([pd.read_csv(DX / "dx_mapping_scored.csv"),
                    pd.read_csv(DX / "dx_mapping_unscored.csv")])
    dx_name = dict(zip(dx.SNOMEDCTCode.astype(str), dx.Abbreviation))

    rows, intervals_out, window_rows = [], [], []
    replay = {"afib_records": [], "normal_candidates": [], "excluded": [],
              "afib_windows": 0, "normal_windows": 0}
    for rec in orig_records:
        base = SRC / "orig" / rec
        header = wfdb.rdheader(str(base))
        ann = wfdb.rdann(str(base), "atr")
        if header.fs != FS or header.n_sig != 12:
            raise ValueError(f"{rec}: unexpected fs/leads")
        age, sex, diagnoses, patient, description = parse_orig_header(base.with_suffix(".hea"))
        if patient != record_patient[rec]:
            raise ValueError(f"{rec}: header patient {patient} != metadata {record_patient[rec]}")
        symbols = Counter(ann.symbol)
        ivals = rhythm_intervals(ann, header.sig_len)
        rhythm_labels = sorted({label for _, _, label in ivals})
        for start, end, label in ivals:
            intervals_out.append({"record_id": rec, "patient": patient, "start_sample": start,
                                  "end_sample": end, "start_s": start / FS, "end_s": end / FS,
                                  "rhythm": label})
        n_windows = int(header.sig_len * FS_OUT / FS) // SEG_LEN_OUT
        beat_win = ((ann.sample.astype(float) * (FS_OUT / FS)) / SEG_LEN_OUT).astype(int)
        per_window = [Counter() for _ in range(n_windows)]
        for w, sym in zip(beat_win, ann.symbol):
            if 0 <= w < n_windows:
                per_window[w][sym] += 1
        afib_fraction = []
        for w in range(n_windows):
            ws, we = w * WINDOW_SAMPLES_257, (w + 1) * WINDOW_SAMPLES_257
            covered = 0
            for start, end, label in ivals:
                if label in EMERGENCY_RHYTHMS:
                    covered += max(0, min(end, we) - max(start, ws))
            afib_fraction.append(covered / WINDOW_SAMPLES_257)
            c = per_window[w]
            beats = sum(v for k, v in c.items() if k not in ("+", "~", "|"))
            window_rows.append({"record_id": rec, "patient": patient, "window": w,
                                "afib_fraction": round(afib_fraction[-1], 4),
                                "beats": beats, "N": c.get("N", 0), "V": c.get("V", 0),
                                "A": c.get("A", 0), "F": c.get("F", 0), "R": c.get("R", 0),
                                "L": c.get("L", 0), "other": beats - sum(
                                    c.get(k, 0) for k in ("N", "V", "A", "F", "R", "L"))})
        # Historical preprocess replay (annotation-only): AF record -> every window positive.
        rhythm_notes = [a.rstrip("\x00") for s, a in zip(ann.symbol, ann.aux_note) if s == "+"]
        is_afib = any(r.strip() in EMERGENCY_RHYTHMS for r in rhythm_notes)
        if is_afib:
            replay["afib_records"].append(rec)
            replay["afib_windows"] += n_windows
            replay_label = "all_windows_emergency"
            n_used = n_windows
        elif not rhythm_notes:
            replay["normal_candidates"].append(rec)
            n_used = 0
            for c in per_window:
                beats = [k for k, v in c.items() for _ in range(v)]
                if len(beats) < 2:
                    continue
                if sum(1 for b in beats if b == "N") / len(beats) >= NORMAL_N_RATIO:
                    n_used += 1
            replay["normal_windows"] += n_used
            replay_label = "normal_windows_by_N_ratio"
        else:
            replay["excluded"].append(rec)
            replay_label = "excluded"
            n_used = 0
        rows.append({
            "record_id": rec, "challenge_id": f"I00{rec[1:]}",
            "in_challenge": rec in chal_to_orig.values(), "patient": patient,
            "patient_diagnosis": patients[patient]["diagnosis"], "age": age, "sex": sex,
            "header_diagnoses": diagnoses, "record_description": description,
            "sig_len": header.sig_len, "seconds": header.sig_len / FS,
            "gain_min": min(header.adc_gain), "gain_max": max(header.adc_gain),
            "annotations": len(ann.sample), "beat_N": symbols.get("N", 0),
            "beat_V": symbols.get("V", 0), "beat_A": symbols.get("A", 0),
            "beat_F": symbols.get("F", 0), "beat_R": symbols.get("R", 0),
            "beat_L": symbols.get("L", 0),
            "other_symbols": ";".join(f"{k}:{v}" for k, v in sorted(symbols.items())
                                      if k not in ("N", "V", "A", "F", "R", "L", "+")),
            "rhythm_markers": len(rhythm_notes), "rhythm_labels": ";".join(rhythm_labels),
            "afib_seconds": round(sum((e - s) / FS for s, e, l in ivals if l in EMERGENCY_RHYTHMS), 1),
            "windows_10s": n_windows,
            "windows_afib_full": int(sum(f >= 0.999 for f in afib_fraction)),
            "windows_afib_partial": int(sum(0 < f < 0.999 for f in afib_fraction)),
            "windows_with_V": int(sum(c.get("V", 0) > 0 for c in per_window)),
            "windows_with_A": int(sum(c.get("A", 0) > 0 for c in per_window)),
            "windows_with_R_or_L": int(sum(c.get("R", 0) + c.get("L", 0) > 0 for c in per_window)),
            "historical_replay": replay_label, "historical_windows_used": n_used,
            "challenge_dx": "", "challenge_dx_names": "", "challenge_age": "", "challenge_sex": "",
        })
    table = pd.DataFrame(rows).set_index("record_id")
    for name in chal_records:
        fields = parse_challenge_header(SRC / "challenge" / f"{name}.hea")
        rec = chal_to_orig[name]
        codes = [c.strip() for c in fields.get("Dx", "").split(",") if c.strip()]
        table.loc[rec, "challenge_dx"] = ";".join(codes)
        table.loc[rec, "challenge_dx_names"] = ";".join(dx_name.get(c, f"unknown:{c}") for c in codes)
        table.loc[rec, "challenge_age"] = fields.get("Age", "")
        table.loc[rec, "challenge_sex"] = fields.get("Sex", "")
    table.reset_index().to_csv(OUT / "incart_record_table.csv", index=False)
    pd.DataFrame(intervals_out).to_csv(OUT / "incart_rhythm_intervals.csv", index=False)
    pd.DataFrame(window_rows).to_csv(OUT / "incart_window_annotation_summary.csv", index=False)
    (OUT / "incart_source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    chal_af = table[table.challenge_dx_names.str.contains(r"\bAF\b|PAF", regex=True)].index.tolist()
    ann_af = table[table.rhythm_labels.str.contains(r"\(AFIB|\(WPWAF", regex=True)].index.tolist()
    # The Challenge abbreviations for ventricular ectopy are VEB/VPVC, not PVC; matching only
    # "PVC" returned 0 and read as an absence rather than as a pattern that never matches.
    chal_pvc = table[
        table.challenge_dx_names.str.contains(r"\bPVC\b|\bVEB\b|\bVPVC\b", regex=True)
    ].index.tolist()
    # ST change has no time-stamped annotation; it exists as record-level free text and as
    # record-level Challenge codes. Both routes are computed so the count is reproducible.
    st_text = table[table.record_description.str.contains("ST", na=False)].index.tolist()
    st_codes = table[
        table.challenge_dx_names.str.contains(r"\bSTD\b|\bSTE\b|STIAb", regex=True)
    ].index.tolist()
    st_patients = sorted({int(table.loc[r, "patient"]) for r in st_text})
    chal_bbb = table[table.challenge_dx_names.str.contains("BBB", regex=False)].index.tolist()
    age_mismatch = [r for r in chal_to_orig.values()
                    if str(table.loc[r, "age"]) != str(table.loc[r, "challenge_age"])]
    sex_map = {"M": "Male", "F": "Female"}
    sex_mismatch = [r for r in chal_to_orig.values()
                    if sex_map.get(table.loc[r, "sex"]) != table.loc[r, "challenge_sex"]]
    result = {
        "status": "annotation_scope_audited_not_an_evaluation",
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "original_records": len(orig_records), "original_patients": len(patients),
        "official_sha256_verified_files": verified,
        "challenge_records": len(chal_records),
        "missing_from_challenge": missing_from_challenge,
        "missing_from_challenge_patient": [record_patient[r] for r in missing_from_challenge],
        "record_seconds": sorted({float(v) for v in table.seconds}),
        "total_seconds": float(table.seconds.sum()),
        "adc_gain_range": [int(table.gain_min.min()), int(table.gain_max.max())],
        "total_annotations": int(table.annotations.sum()),
        "beat_symbol_totals": {k: int(table[f"beat_{k}"].sum()) for k in ("N", "V", "A", "F", "R", "L")},
        "rhythm_marker_records": int((table.rhythm_markers > 0).sum()),
        "rhythm_labels_observed": sorted({l for s in table.rhythm_labels for l in s.split(";") if l}),
        "afib_records_by_annotation": ann_af,
        "afib_partial_windows": {r: int(table.loc[r, "windows_afib_partial"]) for r in ann_af},
        "afib_full_windows": {r: int(table.loc[r, "windows_afib_full"]) for r in ann_af},
        "challenge_af_or_paf_records": chal_af,
        "challenge_ventricular_ectopy_records": len(chal_pvc),
        "st_change_records_by_description": st_text,
        "st_change_records_by_challenge_code": st_codes,
        "st_change_routes_agree": sorted(st_text) == sorted(st_codes),
        "st_change_patients": st_patients,
        "st_change_patient_count": len(st_patients),
        "challenge_bbb_records": chal_bbb,
        "challenge_vs_original_age_mismatch": age_mismatch,
        "challenge_vs_original_sex_mismatch": sex_mismatch,
        "historical_preprocess_replay": {
            "afib_records": replay["afib_records"], "afib_windows": replay["afib_windows"],
            "normal_candidate_records": len(replay["normal_candidates"]),
            "normal_windows": replay["normal_windows"],
            "excluded_records": replay["excluded"],
            "total_windows": replay["afib_windows"] + replay["normal_windows"],
            "matches_recorded_7811_540_7271": (
                replay["afib_windows"] == 540 and replay["normal_windows"] == 7271),
            "patients_in_afib_records": sorted({record_patient[r] for r in replay["afib_records"]}),
            "patients_in_normal_records": len({record_patient[r] for r in replay["normal_candidates"]}),
        },
        "four_group_label_availability": {
            "af": "temporal rhythm markers '(AFIB'/'(WPWAF' in .atr; window-level derivable",
            "ectopy": "beat-level V/A symbols in .atr; window-level derivable",
            "conduction": "no 1AVB/BBB rhythm markers; only record descriptions and beat symbols R/L if present",
            "st_change": "record descriptions only (ST elevation/depression text); no time-stamped ST annotations",
        },
        "limitations": [
            "Waveform .dat files were not downloaded here; prior windows are replayed from annotations only.",
            "Record descriptions are free text at record level, not time-resolved labels.",
            "Beat annotation locations were not manually corrected per the official README.",
            "No model outputs were inspected.",
        ],
        "new_inference_runs": 0, "new_training_runs": 0,
        "script_sha256": sha(Path(__file__)),
    }
    (OUT / "incart_annotation_audit.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps({k: result[k] for k in (
        "missing_from_challenge", "beat_symbol_totals", "rhythm_labels_observed",
        "afib_records_by_annotation", "afib_partial_windows", "challenge_af_or_paf_records",
        "challenge_bbb_records", "historical_preprocess_replay",
        "challenge_vs_original_age_mismatch", "challenge_vs_original_sex_mismatch")},
        indent=2, default=str))


if __name__ == "__main__":
    main()
