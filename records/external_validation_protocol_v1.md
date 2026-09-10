# External validation protocol v1 — frozen before any external outcome

Date: 2026-09-10 KST. **No external model output has been produced or inspected.** This
document fixes the cohort, labels, preprocessing, thresholds, metrics and analysis before
any PTB-XL prediction exists, so that the result cannot be selected after the fact.

Prerequisite audits, all completed and hash-preserved:
`results/pretraining_exposure_20260910/` (published ECG-FM split membership),
`results/stage2_lineage/` (original split recovery, warm-start lineage unresolved),
`results/external_cohort_lineage_20260910/` (PTB-XL cross-version identity, INCART scope).

## 1. What this study can and cannot establish

It can establish: how the frozen P1 model and the eight already-selected input
configurations behave on a public 12-lead cohort that is disjoint from the *published*
ECG-FM pretraining split at record, patient and identical-waveform level.

It cannot establish independence of P1's own adaptation ancestry. Two gaps remain open and
are stated in every report produced under this protocol:

1. The 5d warm-start checkpoint (`outputs/lora_multitask/lora_multitask_best.pt`) has never
   been located, and no commit contains its training log. What data initialised the chain
   that produced a07 is unverified.
2. A separate binary model (stage 5c, `train_lora_mixed.py`) was trained on a PTB-XL
   patient-level split (`preprocess_ptbxl.py`, seed 42, 70/15/15). Neither the code nor the
   records name 5c as an ancestor of 5d, but absence of a reference is not proof of absence.

Therefore every result under this protocol is labelled **conditional external validation**,
never "independent validation", "unseen cohort" or "generalisation to new patients".

## 2. Frozen items — changing any of these invalidates the protocol

| Item | Value |
|---|---|
| Model | a07 epoch 18, alpha 0.7, SHA-256 `287148bfd01ac67b5192268c6c69cc4cc230c004bdfdf332285f38a3b43b08dd` |
| Weights | Frozen. No retraining, no fine-tuning, no calibration layer. |
| Configurations | The eight rows of `results/paper_package_20260910/candidate_table.csv`, unchanged: II; I+aVR; I+III+aVR+aVF; I+III+aVR+aVL+aVF+V5; II+aVR+aVL+aVF+V2; II+aVR+V1+V2; V1+V2; 12-lead |
| Reference | 12-lead, same model, same run |
| Inference | float32, batch 32, TF32 off, deterministic cuDNN, seed 42 reset per configuration, legacy stochastic feature mask unchanged, unused leads zero-filled — identical to `scripts/evaluate_electrode_coverage.py` |
| Repeats | Feature-mask seeds 30000–30009, as in the internal repeat study |

## 3. Cohort

Source: PTB-XL 1.0.3 (21799 records / 18869 patients), `records500` 12-lead 500 Hz.

Eligibility is the F3 set from `ptbxl_f3_eligible_records.csv`: **1682 records / 1673
patients**, defined as records satisfying all four conditions.

1. The record's own Challenge id is in the published ECG-FM **test** split.
2. Every 1.0.3 record of that patient is in the published test split.
3. Every 1.0.1 record of that patient — including the 38 later removed as duplicates — is in
   the published test split.
4. The record has no identical-waveform duplicate partner that sits in the published train or
   valid split. Two records fail only this condition (13803, 15741); both have partners under
   a *different* patient id, so patient-level filtering alone would have missed them.

Records are never re-selected after outcomes. If a record fails to load or preprocess it is
excluded and counted in the failure table; the cohort is not topped up.

## 4. Label mapping

Source: `scp_statements.csv` from PTB-XL 1.0.3 (SHA `ad05b0b1…3bac8f`). A diagnostic
statement counts as present only at likelihood ≥ 50; form and rhythm statements count by
presence, since PTB-XL does not assign them likelihoods in the same way.

| Group | SCP codes | Category in the official dictionary |
|---|---|---|
| AF | AFIB | rhythm |
| ST change (surrogate) | STD_, STE_ | form, "non-specific ST depression/elevation" |
| Conduction (restricted) | 1AVB, CLBBB, CRBBB | diagnostic, class CD |
| Ectopy | PAC, PVC | form |
| Normal | NORM | diagnostic, class NORM |

Rules fixed here:

- STD_/STE_ is a **non-specific ST-change surrogate**, matching the historical CPSC
  STD/STE task. It is not acute ischemia and not myocardial infarction. MI and the full STTC
  superclass must not be substituted; doing so would change the endpoint.
- The conduction endpoint is deliberately restricted to the three CPSC codes. Other
  conduction statements (ILBBB, IRBBB, IVCD, LAFB, LPFB, WPW, 2AVB, 3AVB) are *not* positives.
- A likelihood of 0 on a listed statement does not erase its presence; it is recorded and
  handled by the rule above, not silently dropped.
- Records with no mapped code are not silently discarded; they appear in the cohort tables
  with an explicit "unmapped" marker.

## 5. Two evaluation views, primary declared in advance

**Primary — per-disease one-vs-rest on the full F3 cohort (1682 records / 1673 patients).**
This matches how Sens@95Sp is defined in the frozen internal protocol, which is already
one-vs-rest, and it preserves the positives.

| Group | Positive records | Positive patients |
|---|---:|---:|
| AF | 91 | 90 |
| ST change surrogate | 93 | 93 |
| Conduction restricted | 111 | 111 |
| Ectopy | 95 | 95 |
| Normal | 834 | 833 |

63 records carry two or more of the four disease groups and are positive for each.

The negative set here contains diagnoses absent from the CPSC five-class taxonomy. That is
closer to deployment than the internal test, and it means specificity is **not** comparable
with the internal number. This is stated with every reported value.

**Secondary — exclusive five-class view** for argmax accuracy and Macro-F1 comparability with
the internal test. Records must carry no other-conduction and no other-rhythm code and map to
exactly one group: **844 records / 843 patients** — NSR 696, AF 38, ST change 36, conduction
54, ectopy 20.

This view is reported but is **underpowered by construction**: 20 ectopy positives cannot
resolve the internal ectopy weakness. Its intervals will be wide and must not be read as
agreement or disagreement with the internal result on their own.

## 6. Preprocessing parity

PTB-XL `records500` is already 500 Hz and exactly 5000 samples per record, and `wfdb`
returns `p_signal` in mV with gain 1000/mV — the same code path the frozen CPSC preprocessing
uses (`wfdb.rdrecord(...).p_signal`, first 5000 samples, right zero-pad, transpose to
(12, 5000), float32, no normalisation). No resampling, truncation or unit conversion is
required or permitted.

Lead order in both sources is I, II, III, aVR, aVL, aVF, V1–V6 and is asserted per record
rather than assumed.

**Preflight, run and reported before any prediction:** per-lead amplitude distribution of the
PTB-XL cohort against the CPSC training split. Unit parity is not amplitude parity — INCART
is the precedent, where both sources were in mV yet differed about sixfold in standard
deviation, and a global scale correction did not repair the resulting failure
(`records/03_eval_results.md` §⑪). The measured distributions are reported as context. **No
scale correction is applied**, because doing so would alter the frozen input contract.

## 7. Thresholds

Sens@95Sp operating points are taken from the original CPSC multiclass **validation** split
(933 records, `work/stage2_original_assets/data/processed/cpsc2018_mc/val/`), computed once
per configuration and per disease, written to a threshold file, hashed, and committed
**before** any PTB-XL inference runs. The internal test split is never used to set them.

This is the first time P1 reports a genuinely pre-fixed operating point. The internal
Sens@95Sp values are empirical points chosen inside the test set and are therefore not
comparable to these; both numbers appear side by side with that distinction stated.

The validation split is P1's own model-selection data, so these thresholds are fitted to the
CPSC domain. Threshold transfer failure is an expected and reportable outcome, not a defect
to be corrected by re-fitting on PTB-XL.

## 8. Metrics

Reported per configuration and per disease: AUROC; sensitivity at the pre-fixed threshold;
specificity at the same threshold; F1; and, for the secondary view, the argmax confusion
matrix, accuracy and Macro-F1 over five classes.

Uncertainty: **patient-clustered** bootstrap, 2000 resamples, seed 31415, resampling
patients rather than records, paired against the 12-lead reference with the same resample
indices. PTB-XL supplies patient ids, so this protocol corrects the internal study's
record-level resampling, which could not account for patient dependence. Replicates missing a
class are skipped and counted; no resample is redrawn to obtain a favourable interval.

Intervals are exploratory and unadjusted for multiple comparisons. Eight configurations ×
four diseases is 32 comparisons; no superiority, equivalence or non-inferiority claim follows
from any interval that excludes zero.

## 9. Prespecified analyses, and what would count as a finding

1. Does any low-contact configuration reach all four disease sensitivities ≥ 0.50 at the
   pre-fixed threshold, stably across the ten feature-mask seeds? The internal study did not:
   V1+V2 passed at seed 42 but averaged 0.4707 on ectopy over ten seeds.
2. Does the ranking of the eight configurations by Macro-F1 agree with the internal ranking?
   Rank agreement is reported as a Kendall tau with its interval, not as pass/fail.
3. Does the frozen model degrade in the direction seen for INCART (systematic class
   inversion) or only in magnitude? Confusion structure is reported, not just aggregate scores.

Exploratory analyses are permitted and are labelled exploratory in the same table.

## 10. Abort and integrity rules

- If preprocessing fails on more than 1% of the cohort, the run stops and the cause is
  reported before any metric is computed.
- If the a07 checkpoint hash, the candidate table hash, or the cohort file hash does not match
  the values recorded here, the run stops.
- Raw per-record probabilities, record ids, both label views, failure and skip counts, the
  threshold file hash, environment and wall-clock timing are preserved for every run, as in
  the internal study.
- The internal frozen artefacts are not modified. New outputs go to a new results directory.

## 11. Execution status

This protocol is written and frozen; **it has not been executed.** Execution requires the
repository Python 3.10 environment, which Windows Smart App Control began blocking on this
machine at 2026-09-10 16:06:42 (see the session closeout). No cohort waveforms have been
downloaded, no inference has been run, and no external outcome has been inspected.
