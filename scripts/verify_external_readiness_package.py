"""Verify archived bytes and count evidence without running model inference."""
import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/external_readiness_final_20260911"


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def main():
    base = "a92e9a729a29a1e364d81b983390619bf09c1352"
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    names = subprocess.check_output(["git", "diff", "--name-only", base, head], cwd=ROOT).decode().splitlines()
    inventory = []
    for name in names:
        payload = (ROOT / name).read_bytes()
        committed = subprocess.check_output(["git", "show", f"{head}:{name}"], cwd=ROOT)
        # .gitattributes itself may use normal checkout text endings; preserve both hashes.
        same = payload == committed
        if name != ".gitattributes":
            assert same, name
        inventory.append({"path": name, "bytes": len(payload), "sha256": sha(payload),
                          "commit_blob_sha256": sha(committed), "byte_equal": same})
    control = json.loads((OUT / "cpsc_test_control.json").read_text())
    assert control["status"] == "PASS_TEST_RECONSTRUCTION_ONLY"
    assert control["records"] == 936 and not control["failures"] and control["skipped"] == 0
    assert control["signal_mismatches"] == control["label_mismatches"] == 0
    for name, fingerprint in control["code_sha256"].items():
        assert sha((ROOT / "scripts" / name).read_bytes()) == fingerprint
    for name in ("ptbxl_f3_eligible_records.csv", "ptbxl_f3_label_flags.csv"):
        assert (OUT / name).read_bytes() == (ROOT / "results/external_cohort_lineage_20260910" / name).read_bytes()
    cohort = pd.read_csv(OUT / "ptbxl_f3_eligible_records.csv")
    assert (len(cohort), cohort.patient_id.nunique()) == (1682, 1673)
    identity = pd.read_csv(OUT / "ptbxl_waveform_identity_sample.csv")
    assert len(identity) == 78 and identity.digital_values_identical.sum() == 76
    failed = set(zip(identity.loc[~identity.digital_values_identical, "challenge_hr_id"],
                     identity.loc[~identity.digital_values_identical, "compared_v103_ecg_id"]))
    assert failed == {(3832, 15768), (11838, 11839)}
    responses = pd.read_csv(OUT / "review_responses.csv")
    assert len(responses) == responses.finding_id_sha256.nunique() == 52
    suite = ET.parse(OUT / "tests.xml").getroot().find("testsuite")
    assert int(suite.attrib["tests"]) == 64
    assert all(int(suite.attrib[k]) == 0 for k in ("errors", "failures", "skipped"))
    receipt = {"status": "PASS_ARCHIVE_CHECKS_EXTERNAL_IDENTITY_STILL_FAILS",
               "audited_commit": head, "base_commit": base, "file_count": len(inventory),
               "files": inventory, "cpsc_control": "PASS936", "tests_passed": 64,
               "review_responses": 52, "missing_responses": 0,
               "identity_pairs": 78, "identity_exact": 76, "external_inference_runs": 0,
               "thresholds_computed": False, "push_performed": False}
    (OUT / "package_verification.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps({k:v for k,v in receipt.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
