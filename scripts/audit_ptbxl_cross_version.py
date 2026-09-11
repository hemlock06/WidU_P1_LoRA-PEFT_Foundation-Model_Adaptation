"""PTB-XL cross-version identity, duplicate exposure and HR->ecg_id waveform check.

No model inference. Metadata joins use official PTB-XL 1.0.1/1.0.3 files, the
official changelogs and the pinned ECG-FM published split. A bounded waveform
sample (Challenge .mat versus PTB-XL 1.0.3 records500 .dat) is compared exactly.
"""

import argparse
import hashlib
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
import wfdb
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "work/external_lineage_sources_20260910/ptbxl"
PRIOR = ROOT / "work/exposure_sources_20260910"
OUT = ROOT / "results/external_cohort_lineage_20260910"
SPLIT_SHA = "a2d17b0485a8186d227e82f2200ceb8b2ba7423df23a4fd306275f2197b4d88d"
V103_SHA = "7600de9c1b27d181d850b3c6038a35d7c3ddb6bb33b702e3a20252a6859d216b"
SCP_SHA = "ad05b0b1fcae83bb1230755ad9cfc7c96f303feddc08a4a9ad5bdc9ca63bac8f"
CHALLENGE = "https://physionet.org/files/challenge-2021/1.0.3/training/ptb-xl/"
PTBXL = "https://physionet.org/files/ptb-xl/1.0.3/"
N_V101 = 21837
SAMPLE_SEED = 20260910
GROUPS = {
    "af": ("AFIB",),
    "st_change_surrogate": ("STD_", "STE_"),
    "conduction_restricted": ("1AVB", "CLBBB", "CRBBB"),
    "ectopy": ("PAC", "PVC"),
    "normal": ("NORM",),
}
OTHER_CD = ("ILBBB", "IRBBB", "IVCD", "LAFB", "LPFB", "WPW", "2AVB", "3AVB")
OTHER_RHYTHM = ("AFLT", "SVARR", "SVTAC", "PSVT", "SARRH", "STACH", "SBRAD", "PACE",
                "BIGU", "TRIGU", "SR")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def acquire(url, path):
    reused = path.exists()
    if not reused:
        path.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(url, timeout=120) as response:
            payload = response.read()
        with path.open("xb") as handle:
            handle.write(payload)
    return {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "url": url,
            "bytes": path.stat().st_size, "sha256": sha(path), "reused": reused}


def parse_changelog(text):
    """Return {dropped_id: kept_id} from '#### Change' blocks."""
    mapping = {}
    for block in re.split(r"^#### Change \d+", text, flags=re.MULTILINE)[1:]:
        dropped = [int(x) for x in re.findall(r"->drop row with ecg_id=(\d+)", block)]
        kept = [int(x) for x in re.findall(r"->keep row with ecg_id=(\d+)", block)]
        if not dropped and not kept:
            continue
        if len(kept) != 1:
            raise ValueError(f"Unexpected keep count in block: {block[:80]!r}")
        for item in dropped:
            if item in mapping:
                raise ValueError(f"Duplicate drop entry {item}")
            mapping[item] = kept[0]
    return mapping


def challenge_url(ecg_id, ext):
    return f"{CHALLENGE}g{ecg_id // 1000 + 1}/HR{ecg_id:05d}.{ext}"


def parse_challenge_header(path):
    age = sex = None
    for line in path.read_text().splitlines():
        if line.startswith("# Age:"):
            age = line.split(":", 1)[1].strip()
        elif line.startswith("# Sex:"):
            sex = line.split(":", 1)[1].strip()
    return age, sex


def records500_ids(text):
    return {int(m) for m in re.findall(r"records500/\d{5}/(\d{5})_hr", text)}


def scp_dict(text):
    return json.loads(text.replace("'", '"'))


def validate_drop_maps(v102_map, v103_map, dropped_expected):
    if len(v102_map) != 36 or len(v103_map) != 38 or set(v103_map) != set(dropped_expected):
        raise ValueError("Unexpected changelog drop cardinality or identity")
    if not set(v102_map) <= set(v103_map):
        raise ValueError("1.0.2 drops are not a subset of 1.0.3 drops")


def validate_filters(f1, f2, f3):
    for frame, expected in ((f1, (1685, 1676)), (f2, (1684, 1675)), (f3, (1682, 1673))):
        if (len(frame), frame.patient_id.nunique()) != expected:
            raise ValueError(f"Frozen cohort count changed: {expected}")
    if set(f1.ecg_id) - set(f2.ecg_id) != {2507}:
        raise ValueError("Historical-patient exclusion identity changed")
    if set(f2.ecg_id) - set(f3.ecg_id) != {13803, 15741}:
        raise ValueError("Duplicate-partner exclusion identity changed")


def age_compatibility(challenge_age, released_age, historical_age):
    """Age300 merges missing and privacy-masked ages; consult original metadata."""
    try:
        actual = float(challenge_age)
    except (TypeError, ValueError):
        return False, "unparseable_challenge_age"
    if pd.isna(historical_age):
        return bool(np.isnan(actual) and released_age == 300), "historically_missing"
    if released_age == 300:
        return bool(historical_age > 89 and actual == historical_age), "privacy_masked"
    return bool(actual == released_age), "numeric"


def validate_identity(frame):
    needed = ("digital_values_identical", "gain_match", "lead_names_match")
    if frame.empty or not frame[list(needed)].all(axis=None):
        raise ValueError("Waveform identity gate failed; inspect saved mismatch rows")


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="Use a new audit directory; preserve historical outputs")
    OUT = parser.parse_args().out_dir.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for name in ("ptbxl_database_v101.csv", "ptbxl_v102_changelog.txt",
                 "ptbxl_v103_changelog.txt", "RECORDS_v101", "RECORDS_v103"):
        path = SRC / name
        manifest.append({"path": str(path.relative_to(ROOT)).replace("\\", "/"),
                         "bytes": path.stat().st_size, "sha256": sha(path)})
    if sha(PRIOR / "meta_split_physionet.csv") != SPLIT_SHA:
        raise ValueError("Published split fingerprint changed")
    if sha(PRIOR / "ptbxl_database.csv") != V103_SHA:
        raise ValueError("PTB-XL 1.0.3 metadata fingerprint changed")
    if sha(PRIOR / "scp_statements.csv") != SCP_SHA:
        raise ValueError("SCP statements fingerprint changed")

    v101 = pd.read_csv(SRC / "ptbxl_database_v101.csv")
    v103 = pd.read_csv(PRIOR / "ptbxl_database.csv")
    if len(v101) != N_V101 or sorted(v101.ecg_id) != list(range(1, N_V101 + 1)):
        raise ValueError("Unexpected 1.0.1 record identity")
    if len(v103) != 21799 or v103.patient_id.nunique() != 18869:
        raise ValueError("Unexpected 1.0.3 counts")
    # The official RECORDS files join the last records100 line and the first
    # records500 line without a newline, so entries are parsed by pattern.
    hr_v101 = records500_ids((SRC / "RECORDS_v101").read_text())
    hr_v103 = records500_ids((SRC / "RECORDS_v103").read_text())
    if hr_v101 != set(v101.ecg_id) or hr_v103 != set(v103.ecg_id):
        raise ValueError("RECORDS lists disagree with metadata CSVs")

    dropped_expected = set(v101.ecg_id) - set(v103.ecg_id)
    v103_map = parse_changelog((SRC / "ptbxl_v103_changelog.txt").read_text())
    v102_map = parse_changelog((SRC / "ptbxl_v102_changelog.txt").read_text())
    if set(v103_map) != dropped_expected or len(dropped_expected) != 38:
        raise ValueError("Changelog drop set does not equal metadata difference")
    validate_drop_maps(v102_map, v103_map, dropped_expected)
    if any(k not in set(v103.ecg_id) for k in v103_map.values()):
        raise ValueError("A kept partner is absent from 1.0.3")

    upstream = pd.read_csv(PRIOR / "meta_split_physionet.csv")
    upstream = upstream[upstream.source_path.str.contains("/ptb-xl/")].copy()
    upstream["ecg_id"] = upstream.source_path.str.extract(r"/HR(\d+)$").astype(int)
    if upstream.ecg_id.duplicated().any():
        raise ValueError("Duplicate HR ids in published split")
    split = dict(zip(upstream.ecg_id, upstream.split))
    unlisted_hr = sorted(set(range(1, N_V101 + 1)) - set(split))

    pid101 = dict(zip(v101.ecg_id, v101.patient_id))
    pid103 = dict(zip(v103.ecg_id, v103.patient_id))
    fold103 = dict(zip(v103.ecg_id, v103.strat_fold))
    fold101 = dict(zip(v101.ecg_id, v101.strat_fold))
    patient_changed = [e for e in v103.ecg_id if pid101[e] != pid103[e]]
    fold_changed = [e for e in v103.ecg_id if fold101[e] != fold103[e]]
    kept_to_dropped = {}
    for d, k in v103_map.items():
        kept_to_dropped.setdefault(k, []).append(d)

    rows = []
    for e in range(1, N_V101 + 1):
        in_v103 = e in pid103
        dropped_partner = v103_map.get(e)
        dup_partners = kept_to_dropped.get(e, [])
        rows.append({
            "ecg_id": e, "in_v103": in_v103,
            "patient_id_v101": pid101[e],
            "patient_id_v103": pid103.get(e, ""),
            "strat_fold_v101": fold101[e],
            "published_split": split.get(e, "unlisted"),
            "dropped_in_v103": not in_v103,
            "kept_partner_if_dropped": dropped_partner if dropped_partner else "",
            "same_patient_as_partner": (pid101[e] == pid101[dropped_partner])
            if dropped_partner else "",
            "identical_waveform_dropped_partners": ";".join(map(str, dup_partners)),
            "identical_waveform_partner_splits": ";".join(
                split.get(d, "unlisted") for d in dup_partners),
        })
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "ptbxl_cross_version_table.csv", index=False)

    # Patient-level exposure using full 1.0.1 history (dropped records included).
    hist = v101[["ecg_id", "patient_id"]].copy()
    hist["split"] = hist.ecg_id.map(split).fillna("unlisted")
    full_all_test = hist.groupby("patient_id")["split"].agg(lambda s: bool(s.eq("test").all()))
    cur = v103[["ecg_id", "patient_id", "strat_fold", "scp_codes"]].copy()
    cur["split"] = cur.ecg_id.map(split).fillna("unlisted")
    cur["patient_all_current_test"] = cur.groupby("patient_id")["split"].transform(
        lambda s: s.eq("test").all())
    cur["patient_all_v101_history_test"] = cur.patient_id.map(full_all_test)
    cur["dup_partner_splits"] = cur.ecg_id.map(
        lambda e: ";".join(split.get(d, "unlisted") for d in kept_to_dropped.get(e, [])))
    cur["dup_partner_in_train_or_valid"] = cur.dup_partner_splits.str.contains(
        "train|valid")
    f0 = cur[cur.split == "test"]
    f1 = f0[f0.patient_all_current_test]
    f2 = f1[f1.patient_all_v101_history_test]
    f3 = f2[~f2.dup_partner_in_train_or_valid]
    if len(f1) != 1685 or f1.patient_id.nunique() != 1676:
        raise ValueError("Provisional filter no longer reproduces 1685/1676")
    validate_filters(f1, f2, f3)

    def counts(frame):
        return {"records": len(frame), "patients": int(frame.patient_id.nunique())}

    lost_by_history = f1[~f1.patient_all_v101_history_test]
    lost_by_dup = f2[f2.dup_partner_in_train_or_valid]

    scp = pd.read_csv(PRIOR / "scp_statements.csv", index_col=0)
    diag_codes = set(scp.index[scp.diagnostic == 1.0])

    def flags(codes, diag_min):
        out = {}
        for group, keys in GROUPS.items():
            present = []
            for key in keys:
                if key not in codes:
                    continue
                if key in diag_codes and codes[key] < diag_min:
                    continue
                present.append(key)
            out[group] = present
        out["other_cd"] = [k for k in OTHER_CD if k in codes and codes[k] >= diag_min]
        out["other_rhythm"] = [k for k in OTHER_RHYTHM if k in codes and k != "SR"]
        return out

    def exclusive(fl):
        # Mirror the historical CPSC priority 2>1>3>4>0 on the mapped groups.
        if fl["st_change_surrogate"]:
            return 2
        if fl["af"]:
            return 1
        if fl["conduction_restricted"]:
            return 3
        if fl["ectopy"]:
            return 4
        if fl["normal"]:
            return 0
        return -1

    label_summary = {}
    label_rows = []
    for name, frame in (("all_v103", cur), ("F0_test", f0), ("F3_eligible", f3)):
        for diag_min in (1, 50, 100):
            group_counts = Counter()
            excl = Counter()
            for e, text in zip(frame.ecg_id, frame.scp_codes):
                fl = flags(scp_dict(text), diag_min)
                for group in list(GROUPS) + ["other_cd", "other_rhythm"]:
                    if fl[group]:
                        group_counts[group] += 1
                excl[exclusive(fl)] += 1
                if name == "F3_eligible" and diag_min == 50:
                    label_rows.append({"ecg_id": e, "patient_id": pid103[e],
                                       **{g: ";".join(fl[g]) for g in fl},
                                       "exclusive_priority_label": exclusive(fl)})
            label_summary[f"{name}/diag_likelihood_min_{diag_min}"] = {
                "records": len(frame),
                "group_presence": dict(group_counts),
                "exclusive_priority_counts": {str(k): v for k, v in sorted(excl.items())},
            }
    pd.DataFrame(label_rows).to_csv(OUT / "ptbxl_f3_label_flags.csv", index=False)
    f3[["ecg_id", "patient_id", "strat_fold", "split"]].to_csv(
        OUT / "ptbxl_f3_eligible_records.csv", index=False)

    # Bounded waveform identity sample (exact digital comparison).
    rng = np.random.default_rng(SAMPLE_SEED)
    fixed = [1, N_V101, 138, 144, 2507, 11816, 3802]
    fixed = [e for e in fixed if e in pid103]
    pool_f3 = sorted(set(f3.ecg_id) - set(fixed))
    pool_other = sorted(set(cur.ecg_id) - set(f3.ecg_id) - set(fixed))
    sample = fixed + sorted(rng.choice(pool_f3, 24, replace=False).tolist())
    sample += sorted(rng.choice(pool_other, 8, replace=False).tolist())
    # Verify all38 reported duplicate relations, including both F3 exclusions.
    dropped_sample = sorted(v103_map)
    if unlisted_hr and unlisted_hr[0] in pid103:
        sample.append(unlisted_hr[0])
    fname = dict(zip(v103.ecg_id, v103.filename_hr))
    age103 = dict(zip(v103.ecg_id, v103.age))
    age101 = dict(zip(v101.ecg_id, v101.age))
    sex103 = dict(zip(v103.ecg_id, v103.sex))

    jobs = []
    for e in sample + dropped_sample:
        for ext in ("hea", "mat"):
            jobs.append((challenge_url(e, ext), SRC / "challenge" / f"HR{e:05d}.{ext}"))
    targets = sorted({v103_map.get(e, e) for e in sample + dropped_sample})
    for e in targets:
        for ext in ("hea", "dat"):
            rel = f"{fname[e]}.{ext}"
            jobs.append((PTBXL + rel, SRC / "v103" / Path(rel).name))
    failures = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [(url, pool.submit(acquire, url, path)) for url, path in jobs]
        for url, future in futures:
            try:
                manifest.append(future.result())
            except (OSError, ValueError) as exc:
                failures.append({"url": url, "error": repr(exc)})
    if failures:
        (OUT / "ptbxl_sample_failures.json").write_text(json.dumps(failures, indent=2) + "\n")
        raise RuntimeError(f"{len(failures)} sample acquisition failures")

    identity_rows = []
    for e in sample + dropped_sample:
        target = v103_map.get(e, e)
        chal = loadmat(SRC / "challenge" / f"HR{e:05d}.mat")["val"]
        rec = wfdb.rdrecord(str(SRC / "v103" / Path(fname[target]).name), physical=False)
        ref = rec.d_signal.T
        age, sex = parse_challenge_header(SRC / "challenge" / f"HR{e:05d}.hea")
        chal_hdr = wfdb.rdheader(str(SRC / "challenge" / f"HR{e:05d}"))
        same_shape = chal.shape == ref.shape
        exact = bool(same_shape and np.array_equal(chal.astype(np.int64), ref.astype(np.int64)))
        sex_expected = {0: "Male", 1: "Female"}.get(int(sex103[target]))
        age_v103 = age103[target]
        age_match, age_rule = age_compatibility(age, age_v103, age101[e])
        identity_rows.append({
            "challenge_hr_id": e, "compared_v103_ecg_id": target,
            "relationship": "dropped_duplicate_vs_kept" if e != target else "same_id",
            "published_split": split.get(e, "unlisted"),
            "challenge_shape": "x".join(map(str, chal.shape)),
            "v103_shape": "x".join(map(str, ref.shape)),
            "digital_values_identical": exact,
            "gain_match": bool(chal_hdr.adc_gain == rec.adc_gain and chal_hdr.baseline == rec.baseline),
            "lead_names_match": bool([s.upper() for s in chal_hdr.sig_name] == [s.upper() for s in rec.sig_name]),
            "challenge_age": age, "v103_age": age_v103, "age_match": bool(age_match),
            "age_comparison_rule": age_rule,
            "literal_lead_names_match": chal_hdr.sig_name == rec.sig_name,
            "unit_names_casefold_match": [s.lower() for s in chal_hdr.units] == [s.lower() for s in rec.units],
            "constant_lead_indices": ";".join(map(str, np.flatnonzero(np.ptp(chal, axis=1) == 0))),
            "challenge_sex": sex, "v103_sex": sex_expected, "sex_match": bool(sex == sex_expected),
        })
    identity = pd.DataFrame(identity_rows)
    identity.to_csv(OUT / "ptbxl_waveform_identity_sample.csv", index=False)
    (OUT / "ptbxl_source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    validate_identity(identity)

    result = {
        "status": "cross_version_and_duplicate_exposure_audited_not_an_evaluation",
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "v101_records": len(v101), "v103_records": len(v103),
        "v103_patients": int(v103.patient_id.nunique()),
        "dropped_records": sorted(dropped_expected),
        "dropped_count": len(dropped_expected),
        "dropped_in_v102_changelog": len(v102_map),
        "dropped_only_in_v103_changelog": sorted(set(v103_map) - set(v102_map)),
        "drop_reason": "identical raw waveform duplicates/triplicates per official changelogs",
        "dropped_same_patient_as_kept_partner": int(sum(
            pid101[d] == pid101[k] for d, k in v103_map.items())),
        "dropped_different_patient_than_kept_partner": sorted(
            d for d, k in v103_map.items() if pid101[d] != pid101[k]),
        "surviving_patient_id_changed_between_versions": patient_changed,
        "surviving_strat_fold_changed_between_versions": fold_changed,
        "published_split_ptbxl_rows": len(upstream),
        "challenge_hr_ids_absent_from_published_split": unlisted_hr,
        "dropped_records_published_split": dict(Counter(split.get(d, "unlisted") for d in v103_map)),
        "kept_partner_split_vs_dropped_split": [
            {"dropped": d, "dropped_split": split.get(d, "unlisted"),
             "kept": k, "kept_split": split.get(k, "unlisted"),
             "dropped_patient_v101": pid101[d], "kept_patient_v103": pid103[k]}
            for d, k in sorted(v103_map.items())],
        "cohort_filters": {
            "F0_own_record_published_test": counts(f0),
            "F1_all_current_v103_patient_records_test": counts(f1),
            "F2_all_v101_history_patient_records_test": counts(f2),
            "F3_no_identical_waveform_partner_in_train_or_valid": counts(f3),
            "removed_F1_to_F2_by_dropped_history": counts(lost_by_history),
            "removed_F2_to_F3_by_duplicate_partner": counts(lost_by_dup),
            "F3_records_with_identical_partner_in_test": int(
                f3.dup_partner_splits.map(lambda value: "test" in value.split(";")).sum()),
        },
        "label_presence_under_mapping_proposals": label_summary,
        "waveform_identity_sample": {
            "n_compared": len(identity),
            "same_id_pairs": int((identity.relationship == "same_id").sum()),
            "dropped_vs_kept_pairs": int((identity.relationship != "same_id").sum()),
            "digital_identical": int(identity.digital_values_identical.sum()),
            "gain_and_lead_match": int((identity.gain_match & identity.lead_names_match).sum()),
            "age_match": int(identity.age_match.sum()),
            "sex_match": int(identity.sex_match.sum()),
            "seed": SAMPLE_SEED,
        },
        "limitations": [
            "Identity verified on a bounded sample, not all 21837 records.",
            "Published-list membership is not a runtime training log.",
            "ECG-FM adapted-checkpoint exposure and any external adaptation remain outside this metadata audit.",
            "Label flags are counts under proposed mappings; no clinician adjudication and no model outputs.",
            "Diagnostic likelihood sweep leaves form/rhythm presence unchanged by design; conduction is empirically invariant here.",
            "Lead names are compared case-insensitively; literal spelling and constant leads are recorded separately.",
        ],
        "new_inference_runs": 0, "new_training_runs": 0,
        "script_sha256": sha(Path(__file__)),
    }
    (OUT / "ptbxl_cross_version_audit.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps({k: result[k] for k in (
        "dropped_count", "challenge_hr_ids_absent_from_published_split",
        "dropped_records_published_split", "cohort_filters", "waveform_identity_sample")},
        indent=2, default=str))


if __name__ == "__main__":
    main()
