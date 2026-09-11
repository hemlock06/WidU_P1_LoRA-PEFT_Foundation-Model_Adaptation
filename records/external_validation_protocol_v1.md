# External validation protocol v1 — frozen before any external outcome

> 2026-09-11 update: read [the pre-outcome amendment](external_validation_protocol_amendment_20260911.md)
> with this historical v1. It corrects identity scope, analysis under-specification and execution
> status. Test reconstruction passed; the expanded identity audit failed on2 of78 pairs.
> External inference remains held. The model, candidate list and F3 CSVs are unchanged.

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

"Patient-level" here means PTB-XL's own `patient_id`, and that field is **not** a reliable
identity boundary: 18 of the 38 removed duplicates carry a different `patient_id` from the
identical-waveform record they duplicate. Condition 4 exists to cover exactly that gap, and it
is what removes 13803 and 15741. Condition 2 and 3 alone would have kept both.

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
- **`SR` (sinus rhythm) is exempt from the "other rhythm" exclusion below.** Nearly every
  PTB-XL record carries it, so treating it as a disqualifying rhythm code would collapse the
  secondary cohort. The exemption is what makes that cohort 844 records rather than roughly a
  twelfth of that; it is applied in `audit_ptbxl_cross_version.py` and is stated here because
  the secondary cohort is not reproducible from the exclusion rule without it.

⚠️ **The ST-change endpoint is, in this cohort, entirely ST depression.** All 93 F3 positives
carry `STD_`; `STE_` contributes **zero**. The endpoint is therefore narrower than its name and
than the historical CPSC STD/STE task it is matched to, and any result must be reported as an
ST-depression result rather than as ST change in general.

## 5. Two evaluation views, primary declared in advance

**Primary — per-disease one-vs-rest on the full F3 cohort (1682 records / 1673 patients).**

This choice is inherited, not made today. The internal protocol frozen on 2026-09-09
(`records/electrode_coverage_protocol_v1.md`, "Metrics and selection, fixed before new
outcomes") already declares "Primary four disease groups: AF, ischemia, conduction, ectopy",
runs its Pareto selection over "all four Sens@95Sp values (maximize)", and states that "No
mean score may hide a weak disease class". A per-disease Sens@95Sp is a per-class ROC, which
is one-vs-rest by construction — the term itself is used in
`records/paper_synthesis_20260910.md` §6, not in the frozen protocol, and is quoted here from
that document rather than attributed to the protocol.

So per-disease evaluation is the frozen study's own primary framing, and carrying it across is
continuity, not a view selected after seeing that it yields more positives. The exclusive
five-class view below is the one that departs from the internal framing, which is why it is
secondary despite being the closer match to the model's output contract.

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
PTB-XL cohort against the CPSC training split, reported as context only.

The INCART precedent is cited here for what it actually concluded, which is the opposite of an
amplitude warning. `records/03_eval_results.md` §⑪ found INCART about sixfold larger in
standard deviation than the CPSC training data, **tested the scale hypothesis and rejected it**
(global rescaling moved AUROC 0.281 to 0.274), and attributed the failure to systematic
misclassification at the representation level. So an amplitude gap is not by itself evidence of
transfer risk, and a matching amplitude distribution is not evidence of safety. The preflight
exists so the distributions are on record, not so a decision can be read off them. **No scale
correction is applied**, because doing so would alter the frozen input contract and because the
one time it was tried it did not help.

## 7. Thresholds

Sens@95Sp operating points are taken from the original CPSC multiclass **validation** split
(933 records, `work/stage2_original_assets/data/processed/cpsc2018_mc/val/`), computed once
per configuration and per disease, written to a threshold file, hashed, and committed
**before** any PTB-XL inference runs. The internal test split is never used to set them.

⚠️ **That directory holds only `record_ids.npy`, `labels.npy` and `labels_bin.npy` — there is
no `signals.npy`.** The 2026-09-09 recovery restored identifiers and labels, not waveforms, so
the validation waveforms must be **regenerated** before any threshold can be computed. All 933
raw `.mat` files are present in `data/raw/cpsc2018/`, and the frozen preprocessing is
deterministic and independent of the split seed, so regeneration is possible — but it is a
reconstruction and is labelled as one wherever the thresholds are reported.

**Mandatory positive control before the reconstruction is trusted:** regenerate the *test*
split from raw with the same code path and compare it byte-for-byte against the surviving
`data/processed/cpsc2018_mc/test/signals.npy`. If that comparison is not exact, the
regeneration does not reproduce the original arrays and the threshold step stops. Passing it
establishes the pipeline, not the validation array itself; the validation reconstruction is
still reported as reconstructed.

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
   pre-fixed threshold, across the ten feature-mask seeds? **Decision statistic, fixed here:**
   a configuration passes only if, for each of the four diseases, the *mean* sensitivity over
   seeds 30000–30009 is ≥ 0.50 **and** the *minimum* over those ten seeds is ≥ 0.50. Reporting
   uses the mean, the minimum and the per-seed spread. The internal study fails this test:
   V1+V2 cleared all four floors at seed 42 but its ten-seed ectopy mean was 0.470732 with a
   minimum of 0.426829. Single-seed passes are never reported as passes.
2. Does the ranking of the eight configurations by Macro-F1 agree with the internal ranking?
   Rank agreement is reported as a Kendall tau with its interval, not as pass/fail.
3. Does the frozen model degrade in the direction seen for INCART (systematic class
   inversion) or only in magnitude? Confusion structure is reported, not just aggregate scores.

Exploratory analyses are permitted and are labelled exploratory in the same table.

## 10. Abort and integrity rules

- If preprocessing fails on more than 1% of the cohort, the run stops and the cause is
  reported before any metric is computed.
- If any of these SHA-256 values does not match, the run stops. They are recorded here so the
  rule is executable rather than nominal.

| Artefact | SHA-256 |
|---|---|
| a07 checkpoint | `287148bfd01ac67b5192268c6c69cc4cc230c004bdfdf332285f38a3b43b08dd` |
| `results/paper_package_20260910/candidate_table.csv` | `83241b75482dcef01bf49851d5aa6fd4255ca86495d45223fb89e69818805456` |
| `results/external_cohort_lineage_20260910/ptbxl_f3_eligible_records.csv` | `b23fbe6661efa02ff9eda9b1d17631d4b7fb6e7e96855a46eacba4c1ecbd9c8d` |
| `results/external_cohort_lineage_20260910/ptbxl_f3_label_flags.csv` | `3f33afd43a3b8d600ab1b232e4553c311888fa333e2708fca421227bafab24c5` |
- Raw per-record probabilities, record ids, both label views, failure and skip counts, the
  threshold file hash, environment and wall-clock timing are preserved for every run, as in
  the internal study.
- The internal frozen artefacts are not modified. New outputs go to a new results directory.

## 11. Execution status

This protocol is written and frozen; **it has not been executed.** Execution requires the
repository Python 3.10 environment, which Windows Smart App Control began blocking on this
machine at 2026-09-10 16:06:42 (see the session closeout). No cohort waveforms have been
downloaded, no inference has been run, and no external outcome has been inspected.
