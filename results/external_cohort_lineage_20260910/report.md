# External cohort lineage audit: PTB-XL cross-version identity and INCART annotations

Date: 2026-09-10 KST. No model inference, no training, no external outcome was inspected.
This audit decides which records can enter a *conditional* external evaluation and what
each cohort can and cannot label. It does not make any cohort "independent of P1".

Scripts: `scripts/audit_ptbxl_cross_version.py`, `scripts/audit_incart_annotations.py`.
Sources with URL/bytes/SHA256: `ptbxl_source_manifest.json`, `incart_source_manifest.json`.
Downloaded here (all public PhysioNet/GitHub files): PTB-XL 1.0.1 metadata CSV, both PTB-XL
changelogs, both RECORDS lists, 47 Challenge PTB-XL header/.mat pairs and 40 PTB-XL 1.0.3
records500 header/.dat pairs, INCART 75 headers + 75 annotation files + metadata texts,
74 Challenge INCART headers, and the Challenge SNOMED mapping tables. INCART `.dat`
waveforms (833 MB) and full PTB-XL waveforms were **not** downloaded.

## 1. PTB-XL: what changed between 1.0.1 and 1.0.3

- 1.0.1 has 21837 records (ecg_id 1..21837). 1.0.3 has 21799 records / 18869 patients.
- The 38 removed records are exactly the `->drop` entries of the official changelogs:
  36 identical-raw-waveform duplicates in 1.0.2 and 2 further former-triplicate members
  (3801, 11815) in 1.0.3. The metadata difference and the changelog set are equal.
- For 20 of the 38, the dropped record and its kept identical-waveform partner share a
  1.0.1 `patient_id`. For **18** they carry *different* patient ids (list in `audit.json`).
  Therefore `patient_id` alone is not a reliable identity boundary in PTB-XL; identical
  waveforms exist under distinct patient ids in the Challenge release.
- Surviving records: `patient_id` and `strat_fold` are unchanged between versions (0 diffs).
- The official RECORDS files join the last `records100` line to the first `records500`
  line without a newline; the parser reads them by pattern (test covers this).

## 2. Published ECG-FM split membership after the version reconciliation

- The pinned split lists 21836 PTB-XL rows. Challenge HR12722 is the single id absent;
  it is 1.0.3 record 12722 (patient 2818, fold 2). Its Challenge header shows an all-zero
  V5 lead (checksum 0). That is an observation, not a verified cause of the omission.
- Dropped records in the published split: train 29, valid 5, test 4.

| Filter | Records | Patients | Removed by this step |
|---|---:|---:|---|
| F0 own record in published test | 2146 | 2118 | – |
| F1 all *current* 1.0.3 records of the patient in test | 1685 | 1676 | reproduces the provisional value |
| F2 all *1.0.1-history* records of the patient in test | 1684 | 1675 | 2507 (same-patient dropped duplicate 2506 was train) |
| F3 no identical-waveform dropped partner in train/valid | 1682 | 1673 | 13803 (partner 142 train), 15741 (partner 11810 train), both under different patient ids |

No F3 record has an identical-waveform partner inside the published test list.
`ptbxl_f3_eligible_records.csv` holds the 1682 ids; `ptbxl_cross_version_table.csv`
holds all 21837 ids with both patient ids, split, drop status and partner splits.

## 3. HR -> ecg_id waveform identity (bounded sample)

47 pairs compared digitally (int16, 12 x 5000): 40 same-id pairs (including 1, 21837,
the unlisted 12722, kept partners 138/144/2507/11816/3802, 24 random F3 records and 8
random non-F3 records, seed 20260910) and 7 dropped-Challenge-record versus kept-1.0.3
record pairs (137->138, 143->144, 2506->2507, 11814/11815->11816, 3800/3801->3802).
All 47 were identical in digital values, gain (1000/mV), baseline, lead names, age and
sex. This confirms the numeric HR->ecg_id assumption and the duplicate claim on the
sample; it is not a check of all 21837 records.

## 4. Label availability in F3 under the proposed SCP mapping

Presence counts (diagnostic-statement likelihood >= 50; form/rhythm codes by presence):
AFIB 91; STD_/STE_ 93; 1AVB/CLBBB/CRBBB 111; PAC/PVC 95; NORM 834. Other conduction codes
(ILBBB/IRBBB/IVCD/LAFB/LPFB/WPW/2AVB/3AVB) 268 and other rhythm codes 189 are present and
require explicit exclusion rules. Under the historical CPSC priority (ST > AF > conduction >
ectopy > normal) the exclusive counts are 93 / 71 / 99 / 59 / 797 with 563 unmapped.
The NORM count is sensitive to the likelihood threshold (839 / 834 / 633 at >=1 / >=50 / =100).
Positives per disease run 91 to 111, so intervals will be wide for every group.

## 5. What F3 does and does not establish

F3 satisfies three independence levels against the **published** ECG-FM pretraining list:
record, patient (including 1.0.1 history) and identical waveform. It does **not** establish
(a) runtime pretraining logs, (b) independence from P1's adaptation ancestry: the 5d
warm-start checkpoint has no training log in any commit, and the binary 5c model was trained
on a PTB-XL 1.0.3 patient-level split (`preprocess_ptbxl.py`, seed 42). Neither the code nor
the records reference 5c as an ancestor of 5d, but absence of a reference is not proof.
Any PTB-XL evaluation therefore remains **conditional external validation**.

## 6. INCART: original versus Challenge and annotation content

- Original: 75 records from 32 Holter sources (patients 1..32 in header comments and
  `files-patients-diagnoses.txt`), 12 leads, 257 Hz, 462600 samples (1800 s) each. Measured
  ADC gains span **240..1063** across the 75 headers; the official README states "250 to 1100",
  which is the source's own rounding and not a measurement of these files. All 150
  header/annotation files match the official SHA256SUMS.
- Challenge subset: 74 records; `I0036` (original I36, patient 16) is absent. Age and sex
  agree for all 74 pairs. Challenge headers carry record-level SNOMED codes only.
- Beat labels: N 150410, V 20013, A 1944, F 219, R 3174 (all in I16/I17 = patient 8, plus 8
  in I71), L 0. Rhythm markers exist in only 6 records: `(AFIB` I49/I50, `(PREX` I68/I69/I70/I71,
  `(WPWAF` I71.
- Replaying `preprocess_incart.py` from annotations alone reproduces the historical set
  exactly: 3 AF records x 180 windows = 540 positives, 69 normal-candidate records -> 7271
  windows, 3 excluded (PREX), total 7811. So the earlier binary INCART evaluations
  (③ 0.710, 5f 0.284) used positives from **two patients** (22 and 30) and labelled all 180
  windows of I71 as AF although its annotated AF covers 148.9 s (11 full + 5 partial windows):
  at least 164 of the 540 positive windows were not AF by annotation. Negatives came from 29
  patients; the N-ratio >= 0.9 rule admits up to 10% ectopic beats per "normal" window.
- Four-group availability: AF is temporal but from 2 patients; ectopy is beat-level (V in 70
  records, A in 33); conduction has only RBBB beats in patient 8 (plus 8 beats in I71) and no
  1AVB/LBBB; ST change exists only as record-level free text, in 10 records across **9**
  distinct patients (I03, I06, I12, I15, I24, I40, I50, I59, I60, I62; only patient 26 owns
  two). Two independent routes agree — the free-text descriptions and the Challenge ST SNOMED
  codes select the identical 10 records. This count is recomputed from `incart_record_table.csv`
  and is not produced by the audit script.
  INCART cannot support a four-group evaluation. It can support an ectopy window endpoint and
  a small AF endpoint with patient-grouped uncertainty (32 groups).
- Exposure: INCART is excluded from ECG-FM pretraining (paper and published list) and was
  never in P1 training. It was used in P1 inference-only evaluations and in the 5f inversion
  diagnosis, so it is not "unseen in development".

## 7. Adversarial review: what was adjudicated and what was not

An adversarial review of this report ran on 2026-09-10 across seven independent dimensions,
each re-deriving the numbers from the primary sources rather than reading them back. It raised
**62 findings**, each of which was then to be independently cross-checked three times.

**Only 10 of the 62 were cross-checked.** The checking stage failed partway through and lost
most of its checks, and the tally treated a finding with zero completed checks as rejected. So
**52 findings — including 4 marked critical and 25 marked major — were filed as rejected when
they had never been checked at all.** This is the failure mode this project records elsewhere,
reproduced inside the harness built to catch it: an absent verification is not a passed one,
and a harness that dies reports silence rather than safety. The counts given here are the
corrected reading, not the tally's output.

Of the 10 that were cross-checked, 7 were upheld and 3 were rejected on primary-source
grounds. The complete list with per-finding adjudication status is
`review_findings_20260910.csv`.

The findings acted on below were each **re-derived by hand from the primary artefacts before
being accepted**, not taken on the reviewer's word:

| Corrected | Was | Now | Where |
|---|---|---|---|
| INCART ST-change patients | 8 | **9** | §6, this report |
| INCART ADC gain range | 250..1100 (quoted from README) | **240..1063** (measured) | §6, this report |
| Record 420 noise spans in the hand-trace | three | **four** | §8, this report |
| PTB-XL per-disease positives | "below 100 each" | **91 to 111** | §4, this report |
| VFDB positive windows under the protocol's own exclusion | 1039 / 118 | **802 / 109** | emergency protocol §5 |
| ST endpoint composition | "STD_/STE_" | **STD_ only; STE_ contributes 0 of 93** | validation protocol §4 |
| Secondary cohort rule | SR carve-out unstated | **stated** | validation protocol §4 |
| Threshold source | presented as available | **waveforms absent; reconstruction + positive control required** | validation protocol §7 |
| Abort-rule hashes | named but not recorded | **four SHA-256 values recorded** | validation protocol §10 |
| Analysis 1 decision statistic | "stably" undefined | **mean ≥ 0.50 and min ≥ 0.50 over ten seeds** | validation protocol §9 |
| INCART amplitude precedent | read as an amplitude warning | **corrected to what §⑪ concluded** | validation protocol §6 |

**Script defects found, fixed and re-run on 2026-09-11:**

- `audit_incart_annotations.py` reported `challenge_pvc_records: 0`. That was a regex artifact:
  the Challenge headers use `VEB`/`VPVC`, not `PVC`. The field is now
  `challenge_ventricular_ectopy_records: 59`. A zero that means "the pattern never matches"
  must not be shipped looking like an absence.
- The same script computed no ST record or patient count, so the §6 figures had no derivation.
  It now emits both routes and their agreement: `st_change_records_by_description` and
  `st_change_records_by_challenge_code` select the identical 10 records
  (`st_change_routes_agree: true`) over `st_change_patient_count: 9`.
- `audit_vfdb_episodes.py` counted positive windows without the noise filter it applied to
  negatives, counted 22 `UNLABELED_START` pseudo-episodes inside `rhythm_episodes_total`, and
  did not disclose its end-of-record noise fallback. All three are fixed; the emergency-rhythm
  protocol §7 carries the reconciled figures.

The Smart App Control block described in §8 cleared overnight, which is what made the fixes
verifiable rather than merely written. Both audits were re-run and their previously verified
figures are unchanged: the INCART replay still reproduces 7811 windows as 540 and 7271, and the
PTB-XL cohort filters still give 2146 / 1685 / 1684 / 1682.

**The 52 unchecked findings are unresolved, not dismissed.** They are listed in
`review_findings_20260910.csv` under `adjudication: not_adjudicated`, with their severity and
the artefact each concerns, and must be re-examined before this work is relied on.

## 8. Files

`ptbxl_cross_version_audit.json`, `ptbxl_cross_version_table.csv`, `ptbxl_f3_eligible_records.csv`,
`ptbxl_f3_label_flags.csv`, `ptbxl_waveform_identity_sample.csv`, `ptbxl_source_manifest.json`,
`review_findings_20260910.csv`,
`incart_annotation_audit.json`, `incart_record_table.csv`, `incart_rhythm_intervals.csv`,
`incart_window_annotation_summary.csv`, `incart_source_manifest.json`.
Both scripts run on the repository `.venv` (Python 3.10.21, numpy 1.26.4, pandas 2.3.3,
wfdb 4.3.1, scipy 1.15.3) without GPU.

Test status, stated exactly:

- `tests/test_external_cohort_lineage.py` — 7 tests, **executed and passed** together with
  `test_pretraining_exposure.py` and `test_stage2_lineage.py` (14 passed).
- `tests/test_vfdb_episodes.py` — 4 tests, **written but never executed**. Windows Smart App
  Control began blocking this repository's Python 3.10 interpreter at 2026-09-10 16:06:42,
  after the audits had already produced their outputs but before these tests could run. They
  are committed unexecuted and must be run before the emergency-rhythm work proceeds.
  The episode logic they pin was instead verified by hand-tracing records 605 (leading noise),
  424 (noise inside an episode, repeated identical rhythm) and 420 (four noise spans, three
  identical rhythm markers) against the official annotation rule, and by reconciling every
  raw marker count with its merged episode count (VT 93 markers to 90 episodes, explained
  entirely by records 420 and 426). Hand-tracing is evidence; it is not a substitute for the
  regression tests, which stay unverified until executed.

