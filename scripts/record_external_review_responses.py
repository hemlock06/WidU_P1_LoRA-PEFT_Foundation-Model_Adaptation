"""Preserve the52 prior adjudications and distinguish responses from resolved gates."""
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/external_cohort_lineage_20260910/review_adjudication_20260911.csv"
SOURCE_SHA = "a38f2333429488fa918f17cf8596fa776feca118f7e6aa02ef39072b0b4ad224"
OUT = ROOT / "results/external_readiness_final_20260911"
AMEND = "records/external_validation_protocol_amendment_20260911.md"

# Indices only address the SHA-pinned source; output uses full title and title hash.
# Prior verdicts/evidence remain verbatim. Responses never imply external performance passed.
RESPONSES = {
 7: ("previously_corrected", "VFDB802 positives/109 ambiguous retained; no new training."),
 8: ("partial_execution", "Test936 reconstruction is byte exact; validation933 still unbuilt."),
 9: ("plan_clarified_pending_execution", "Seed42 primary table/CI; ten-seed mean/min for stability."),
10: ("previously_corrected", "Four original SHA pins retained and independently checked."),
11: ("previously_corrected", "Measured INCART gain240..1063 retained."),
12: ("previously_corrected", "INCART ST10 records/9 patients retained and recalculated."),
13: ("documented", "Enumerated excluded rhythms; SR exemption explicit; no flag-file changes."),
14: ("code_corrected", "VBig included; corrected audit now60 ventricular-ectopy records."),
15: ("documented_limitation", "Historical scaling result is recorded evidence, not newly reproduced causation."),
16: ("gate_failed", "Both F3-specific pairs exact; all38 audit exposes2 other nonexact pairs. Hold retained."),
17: ("documented_limitation", "Patient IDs are recorded IDs; no exhaustive waveform/biological independence claim."),
18: ("documented", "Secondary844 definition explicitly exempts SR; eligibility file unchanged."),
19: ("documented_limitation", "Stage5c common-parent record and1012/1005 replay disclosed; ancestry not proven."),
20: ("documented_limitation", "Published pretraining list is not runtime training evidence; third gap restored."),
21: ("plan_clarified_pending_execution", "Case-only lead canonicalization specified; units/order required; no scaling."),
22: ("documented", "Full-cohort priority column is not secondary mask; explicit one-group exclusion rule."),
23: ("documented", "Ten excluded other-rhythm codes enumerated, SR exempt;189 versus1529 disclosed."),
24: ("documented_limitation", "Validation selected checkpoint and fits thresholds; uncertainty conditional on both."),
25: ("plan_clarified_pending_execution", "Restore all five internal sensitivity floors0.50..0.90 before outcomes."),
26: ("plan_clarified_pending_execution", "Rank primary repeat-mean Macro-F1; seed42 ranking descriptive; tau-b specified."),
27: ("previously_corrected", "ST endpoint93 STD_ and0 STE_ retained; no acute ischemia claim."),
28: ("plan_clarified_pending_execution", "Sensitivity/achieved specificity primary; no expected-failure exemption."),
29: ("documented_limitation", "Replay1012/1682 overlap is conditional;12849 versus12847 discrepancy retained."),
30: ("documented_limitation", "Threshold order statistics and negative counts disclosed; bootstrap excludes calibration uncertainty."),
31: ("documented", "AFIB91 versus1 at diagnostic threshold; STD93 presence-based; label schema fixed."),
32: ("code_corrected", "Age300 missing/privacy branches separated; six synthetic cases pass, not population validation."),
33: ("code_corrected_gate_failed", "Identity fail gate tested and actually fired on2 of78 pairs; evidence saved."),
34: ("code_corrected", "F1/F2/F3 counts and removed IDs guarded; mutant filter tests pass."),
35: ("code_corrected", "Exact36/38 map cardinalities required; empty1.0.2 parse regression rejected."),
36: ("previously_corrected", "Record420 four noise spans retained."),
37: ("previously_corrected", "Disease positives91..111 retained, not all below100."),
38: ("previously_corrected", "592 markers split519 rhythm/73 noise retained."),
39: ("documented", "Rename sinus negative criterion>=99.9%;1603 versus1602 strict; tolerance unchanged."),
40: ("previously_corrected", "Gain240..1063 retained; official rounded description distinguished."),
41: ("documented", "159 omitted annotations broken into147 other beats and12 rhythm markers."),
42: ("code_corrected", "Added7 conduction-code records/4 patients; not temporal restricted-endpoint labels."),
43: ("documented_limitation", "AF positives span2 of32 source groups;32 is not AF-positive sample size."),
44: ("documented_limitation", "Historical INCART arrays absent;7811 annotation replay is not byte reproduction."),
45: ("documented_limitation", "Missing inversion-diagnosis script limits historical causal explanation."),
46: ("previously_corrected", "INCART measured gains retained separately from source prose."),
47: ("retained_rejection", "Official rule text exists in tracked script/protocol; page-content hash remains a separate gap."),
48: ("pending_execution", "Low expected failure rate does not make a>1%gate inert; external fault-injection runner remains unbuilt. Amplitude is descriptive by design."),
49: ("documented", "No closer-to-deployment claim; source readiness distinct from clinical/device performance."),
50: ("documented_limitation", "Unmapped563/1682=33.47% included as mapped-endpoint negatives; not healthy controls."),
51: ("previously_corrected", "Conduction111, others91/93/95; uncertainty remains broad."),
52: ("documented", "Other rhythm189 requires SR exemption; including SR gives1529."),
53: ("documented", "Literal lead spellings differ; case-normalized names and units compared separately."),
54: ("gate_failed", "38 catalogued relations tested;36 exact/2 nonexact. Global identical-duplicate claim withdrawn."),
55: ("code_corrected_not_reached", "Count now tests semicolon token=test, but summary emission prevented by identity failure."),
56: ("documented", "Likelihood sweep structurally leaves form/rhythm flags unchanged; not robustness evidence for their presence rule."),
57: ("code_corrected", "Constant-lead indices retained;12722 V5 reported without inferring source omission cause."),
58: ("retained_rejection_with_guard", "Prior47 headers did not reproduce float crash; malformed future age now records incompatibility."),
}


def main():
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_SHA
    frame = pd.read_csv(SOURCE).fillna("")
    target = frame.index[frame.adjudication == "not_adjudicated"]
    assert len(target) == 52 and set(target) == set(RESPONSES)
    rows = []
    for index in target:
        row = frame.loc[index].to_dict()
        row["finding_id_sha256"] = hashlib.sha256(row["title"].encode()).hexdigest()
        row["response_status"], row["response_reason_and_action"] = RESPONSES[index]
        row["correction_record"] = AMEND
        row["source_csv_sha256"] = SOURCE_SHA
        row["resolution_commit_lookup"] = "git log -1 --format=%H -- results/external_readiness_final_20260911/review_responses.csv"
        rows.append(row)
    result = pd.DataFrame(rows)
    assert result.finding_id_sha256.nunique() == 52
    assert result.evidence_20260911.str.len().gt(0).all()
    result.to_csv(OUT / "review_responses.csv", index=False)
    receipt = {"source_rows": len(frame), "original_unadjudicated": 52,
               "now_adjudicated": 52, "missing_responses": 0,
               "prior_verdict_counts": result.adjudication_20260911.value_counts().to_dict(),
               "response_status_counts": result.response_status.value_counts().to_dict(),
               "external_validation_complete": False,
               "note": "Adjudication completeness is not correction or external-gate completion."}
    (OUT / "review_response_verification.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
