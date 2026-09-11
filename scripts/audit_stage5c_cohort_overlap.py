"""Measure how much of the F3 external cohort sat in stage 5c's PTB-XL training split.

Reconnaissance for the 2026-09-11 adjudication, not part of the frozen external
validation protocol. No model inference, no training, no waveform is touched: the
split in `preprocess_ptbxl.py` is a seeded shuffle over patient ids, so it replays
from metadata alone.

`external_validation_protocol_v1.md` §1.2 names stage 5c as a live contamination
risk and leaves it at "absence of a reference is not proof of absence". This script
turns that into a number. The number is conditional on 5c being an ancestor of a07,
which the missing 5d warm-start log cannot settle — but the measurement itself does
not depend on that question.

Run: .venv/Scripts/python scripts/audit_stage5c_cohort_overlap.py
"""

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "work/exposure_sources_20260910"
LINEAGE = ROOT / "results/external_cohort_lineage_20260910"
OUT = LINEAGE / "stage5c_cohort_overlap.json"

# preprocess_ptbxl.py: SEED = 42, patient-level 70/15/15, val and test taken first.
SEED = 42
RECORDED_TRAIN_RECORDS = 12847  # records/03_eval_results.md:131


def map_label(scp_codes, scp_df):
    """preprocess_ptbxl.py map_label: 1 emergency (MI/STTC), 0 pure NORM, -1 excluded."""
    classes = set()
    for code, likelihood in scp_codes.items():
        if likelihood == 0:
            continue
        if code in scp_df.index:
            cls = scp_df.loc[code, "diagnostic_class"]
            if pd.notna(cls) and cls != "":
                classes.add(cls)
    if "MI" in classes or "STTC" in classes:
        return 1
    if "NORM" in classes and len(classes) == 1:
        return 0
    return -1


def patient_split(patient_ids, seed=SEED):
    """preprocess_ptbxl.py patient_split, reproduced exactly."""
    rng = np.random.default_rng(seed)
    patients = np.array(patient_ids)
    rng.shuffle(patients)
    n_val = int(len(patients) * 0.15)
    n_test = int(len(patients) * 0.15)
    return {
        "val": set(patients[:n_val]),
        "test": set(patients[n_val:n_val + n_test]),
        "train": set(patients[n_val + n_test:]),
    }


def main():
    db = pd.read_csv(PRIOR / "ptbxl_database.csv")
    scp = pd.read_csv(PRIOR / "scp_statements.csv", index_col=0)
    db["label"] = [map_label(ast.literal_eval(t), scp) for t in db.scp_codes]

    splits = patient_split(db.patient_id.unique())
    train = db[db.patient_id.isin(splits["train"])]
    train_used = train[train.label >= 0]

    f3 = pd.read_csv(LINEAGE / "ptbxl_f3_eligible_records.csv")
    overlap_records = sorted(set(f3.ecg_id) & set(train_used.ecg_id))
    overlap_patients = sorted(set(f3.patient_id) & set(train_used.patient_id))

    result = {
        "status": "metadata_replay_not_an_evaluation",
        "note": ("Conditional on stage 5c being an ancestor of a07, which the missing "
                 "5d warm-start log does not establish. The replay itself needs no "
                 "waveforms and no inference."),
        "ptbxl_version_replayed": "1.0.3",
        "seed": SEED,
        "patients_total": int(db.patient_id.nunique()),
        "stage5c_train_patients": len(splits["train"]),
        "stage5c_train_records": len(train),
        "stage5c_train_records_label_usable": len(train_used),
        "recorded_train_records_in_03_eval_results": RECORDED_TRAIN_RECORDS,
        "replay_vs_recorded_delta": len(train_used) - RECORDED_TRAIN_RECORDS,
        "f3_records": len(f3),
        "f3_patients": int(f3.patient_id.nunique()),
        "f3_records_in_stage5c_train": len(overlap_records),
        "f3_patients_in_stage5c_train": len(overlap_patients),
        "f3_record_overlap_fraction": round(len(overlap_records) / len(f3), 4),
        "overlapping_ecg_ids": overlap_records,
        "new_inference_runs": 0,
        "new_training_runs": 0,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items()
                      if k != "overlapping_ecg_ids"}, indent=2))


if __name__ == "__main__":
    main()
