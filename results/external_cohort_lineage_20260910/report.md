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
Positive counts per disease are below 100 each; intervals will be wide.

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
  `files-patients-diagnoses.txt`), 12 leads, 257 Hz, 462600 samples (1800 s) each, gains
  250..1100 ADC/mV. All 150 header/annotation files match the official SHA256SUMS.
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
  1AVB/LBBB; ST change exists only as record-level free text (10 records / 8 patients).
  INCART cannot support a four-group evaluation. It can support an ectopy window endpoint and
  a small AF endpoint with patient-grouped uncertainty (32 groups).
- Exposure: INCART is excluded from ECG-FM pretraining (paper and published list) and was
  never in P1 training. It was used in P1 inference-only evaluations and in the 5f inversion
  diagnosis, so it is not "unseen in development".

## 7. Files

`ptbxl_cross_version_audit.json`, `ptbxl_cross_version_table.csv`, `ptbxl_f3_eligible_records.csv`,
`ptbxl_f3_label_flags.csv`, `ptbxl_waveform_identity_sample.csv`, `ptbxl_source_manifest.json`,
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
  424 (noise inside an episode, repeated identical rhythm) and 420 (three noise spans, three
  identical rhythm markers) against the official annotation rule, and by reconciling every
  raw marker count with its merged episode count (VT 93 markers to 90 episodes, explained
  entirely by records 420 and 426). Hand-tracing is evidence; it is not a substitute for the
  regression tests, which stay unverified until executed.

