# Verification evidence and explicit hold,2026-09-11

This directory's name identifies the last acquisition attempt, **not a successful external
validation**. Exact-identity gate: FAIL. External inference/training:0. F3 IDs and labels
remain byte-identical to the frozen originals. Thresholds have not been computed.

Read `../../records/paper_decisions_and_implications_20260911.md` for the paper argument and
`../../records/external_validation_protocol_amendment_20260911.md` for decisions and limits.

- `cpsc_test_control.json`:936 test reconstructed signals/labels byte-exact; copied original
  IDs are an input, not an independently reconstructed split. Actual arrays are under
  `work/cpsc_external_reconstruction_verified_20260911/`, excluded from Git.
- `cpsc_test_raw_inventory.json`: ordered936 raw header/waveform hashes and per-record deltas.
- `ptbxl_waveform_identity_sample.csv`:78 comparisons;40 same-ID pairs exact and36/38
  catalogued deletion pairs exact.3832→15768 and11838→11839 differ. Case-normalized lead
  names are distinct from literal spelling; constant lead indices are retained.
- `identity_differences_by_lead.csv`:936 pair/lead rows, including zero-difference rows;
  no time shifts, amplitude corrections, or fitted matching tolerances were used.
- `identity_gate_diagnosis.json`: the two F3-specific exclusions match exactly; no cohort
  edit or automatic gate relaxation follows from that fact.
- `ptbxl_source_manifest.json`: public source URLs, bytes and hashes. Raw waveforms remain
  local under `work/external_lineage_sources_20260910/ptbxl/`, not in these results.
- `review_responses.csv`: all52 original unchecked findings mapped to preserved prior
  verdict/evidence and a current reason/action. A documented limitation is not a fixed gate.
- `review_response_verification.json`:52/52 mapped; zero missing responses. Original52
  adjudications were28 open/9 partial/13 previously fixed/2 rejected; those historical
  statuses remain unchanged in the source ledger.
- `tests.xml`:64 tests passed, no errors/failures/skips. Invocation verified a07 SHA before
  locally allowing the three trusted NumPy types in a scoped torch safe-globals context.
  The first whole-suite attempt reached64 test dots but exited1 during pytest shared-temp
  cleanup (`WinError5`, pytest-current). A new private `--basetemp` gave exit0; no shared
  temp directory was removed and serialization/security policy was not changed globally.

Acquisition failure history is in `../external_readiness_corrections_20260911/ptbxl_sample_failures.json`
(20 timeout attempts) and `../external_readiness_retry_20260911/ptbxl_sample_failures.json`
(2 timeout attempts). Final attempt acquired all files and reached the actual identity
comparison. Report attempts separately; these are not22 distinct missing subjects.
Corrected INCART audit in the corrections directory yields60 ventricular-ectopy records,
7 conduction-code records/4 patients, unchanged7811=540+7271 historical window-count replay.

Reproduction commands from repository root (each new run should use a new output directory):

```powershell
.venv\Scripts\python.exe scripts\reconstruct_cpsc_external_control.py --out-dir work\cpsc_control_NEW
.venv\Scripts\python.exe scripts\audit_ptbxl_cross_version.py --out-dir results\ptbxl_audit_NEW
# Expected to exit nonzero while the two nonexact catalogued pairs are treated as exact.
.venv\Scripts\python.exe scripts\diagnose_ptbxl_identity_gate.py --audit-dir results\ptbxl_audit_NEW
.venv\Scripts\python.exe scripts\audit_incart_annotations.py --out-dir results\incart_audit_NEW
.venv\Scripts\python.exe scripts\record_external_review_responses.py
```

No raw waveforms, checkpoint weights or private conversation transcripts are included.
The claim is a reproducible preparation/audit result and decision record, not external
performance, clinical benefit, real wearable optimality or early prediction.

Byte-preservation correction: Git initially normalized line endings for some evidence files.
Commit0064368 adds explicit `-text` paths and stores the actual hash-addressed local bytes,
including the original F3 CSVs and the preprocessing source. No waveform, label value, ID,
or preprocessing operation changed. `package_verification.json` compares the43 changed files
at that commit against local bytes; this is repository verification, not remote verification.
