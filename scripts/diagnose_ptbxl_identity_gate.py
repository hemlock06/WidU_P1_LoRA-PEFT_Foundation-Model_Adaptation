"""Quantify every audited pair without converting approximate matches to passes."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.audit_dir
    pairs = pd.read_csv(out / "ptbxl_waveform_identity_sample.csv")
    meta = pd.read_csv(ROOT / "work/exposure_sources_20260910/ptbxl_database.csv").set_index("ecg_id")
    src = ROOT / "work/external_lineage_sources_20260910/ptbxl"
    rows = []
    for pair in pairs.itertuples():
        path = src / "challenge" / f"HR{pair.challenge_hr_id:05d}.mat"
        x = loadmat(path)["val"].astype(np.int64)
        target = src / "v103" / Path(meta.loc[pair.compared_v103_ecg_id, "filename_hr"]).name
        y = wfdb.rdrecord(str(target), physical=False).d_signal.T.astype(np.int64)
        for lead in range(12):
            delta = x[lead] - y[lead]
            rows.append({"challenge_id": pair.challenge_hr_id,
                         "v103_id": pair.compared_v103_ecg_id, "lead_index": lead,
                         "different_samples": int(np.count_nonzero(delta)),
                         "max_absolute_digital_difference": int(np.abs(delta).max()),
                         "mean_digital_difference": float(delta.mean()),
                         "std_digital_difference": float(delta.std()),
                         "correlation": float(np.corrcoef(x[lead], y[lead])[0, 1])
                             if np.ptp(x[lead]) and np.ptp(y[lead]) else None})
    pd.DataFrame(rows).to_csv(out / "identity_differences_by_lead.csv", index=False)
    different = sorted({(r["challenge_id"], r["v103_id"]) for r in rows if r["different_samples"]})
    critical = pairs[pairs.challenge_hr_id.isin([142, 11810])]
    summary = {"status": "FAIL_EXACT_IDENTITY_GATE", "compared_pairs": len(pairs),
               "nonidentical_pairs": different, "exact_pairs": len(pairs)-len(different),
               "f3_exclusion_pairs_exact": bool(len(critical) == 2 and critical.digital_values_identical.all()),
               "cohort_modified": False, "external_inference_runs": 0,
               "interpretation": "Changelog deletion relation is not universal byte identity; no gate relaxation.",
               "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out / "identity_gate_diagnosis.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
