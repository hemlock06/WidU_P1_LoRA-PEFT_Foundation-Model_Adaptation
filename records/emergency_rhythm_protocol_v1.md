# Emergency rhythm (VT/VF) expansion protocol v1 — design only, training not authorised

Date: 2026-09-10 KST. Status: **specification.** No emergency-rhythm model has been trained
or evaluated. Nothing here is a performance result. Training begins only after hemlo approves
a concrete plan; this document exists so that approval can be given against a fixed design
rather than against an intention.

Evidence base: `results/vfdb_readiness_20260910/` (acquisition and format audit) and
`results/vfdb_episodes_20260910/` (episode construction under the official annotation rule).

## 1. Why this is a separate study, not an extra output head

The frozen P1 model has a binary emergency head and a five-class head covering NSR, AF,
ST change, conduction and ectopy. VT, ventricular flutter, ventricular fibrillation and
asystole are **not** among its targets. Enlarging the multiclass head's output dimension is a
one-line change and confers no ability to detect these rhythms; a new head or adapter must be
trained against annotated data and evaluated on its own terms.

The electrode shortlist from the internal study does not transfer. Those configurations were
ranked on the four CPSC groups using standard 12-lead inputs; VFDB provides two channels both
named `ECG` with no documented electrode positions. Carrying the ranking across would assert a
lead correspondence that the data does not support.

AF and VF are different arrhythmias. The existing binary "emergency" label (AF or ST change)
is not a validated clinical emergency definition and is not a proxy for cardiac arrest.

## 2. Data actually in hand

MIT-BIH Malignant Ventricular Ectopy Database v1.0.0, 22 records, all 67 files hashed against
the official source, all 22 WFDB checksums recomputed and matching.

| Property | Observed |
|---|---|
| Channels | 2, both named `ECG`; no lead or electrode geometry documented |
| Sampling | 250 Hz |
| Length | 525000 samples = 2100 s per record; 46200 s total (12 h 50 m) |
| Annotations | 592 markers, of which 73 are `(NOISE` and 519 change rhythm; **no beat labels** |
| Subjects | Not identified. Records cannot be grouped by patient. |

The official overview describes half-hour records; the headers say 35 minutes. Header-derived
lengths are used.

## 3. Episode construction rule, fixed

The official database page states: "The rhythm change annotations are placed at the beginning
of the episode of the indicated rhythm. The previous rhythm continues during episodes marked
by (NOISE; the noise ends at the time of the next annotation."

Implemented in `scripts/audit_vfdb_episodes.py`:

- A rhythm marker sets the rhythm state from its sample until the **next rhythm marker**.
- A `(NOISE` marker opens a noise span ending at the next marker of any kind and **does not
  change the rhythm state**. A naive previous-annotation-to-next-annotation parser labels the
  post-noise interval as NOISE and is wrong.
- Samples before the first rhythm marker are `UNLABELED_START`, never assumed normal. Record
  605 opens with `(NOISE`, and one record's first marker arrives 13999 samples in.
- **Where the official rule is silent, the implementation extrapolates and this is not a
  reading of the rule.** Six of 22 records (419, 420, 422, 423, 424, 605) end with a `(NOISE`
  marker, so there is no "next annotation" for the noise to end at. The code extends that span
  to end-of-record. It accounts for 1771.5 s of the 6064.9 s total noise, or 29.2%; the
  bounded, rule-derived part is 4293.4 s. Reading it the other way (noise ends at the marker)
  moves the noise-free full-malignant window count from 732 to 829 and the clean-negative
  count from 1603 to 1630. Both readings are recorded; neither is established by the source.

Of the 592 annotations, **73 are `(NOISE` markers**, which by the rule above do not change
rhythm; 519 are rhythm markers. Applying the rule yields 488 spans, of which 22 are
`UNLABELED_START` pre-annotation regions totalling 1706.1 s. **Real annotated rhythm episodes
are therefore 466**, not 488.

| Target set | Episodes | Records | Total s | of which noise | Median | ≥10 s | ≥30 s | of those ≥30 s: >50% noise |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| VT | 90 | 19 | 6203.1 | 963.1 (15.5%) | 4.0 s | 21 | 15 | 4 |
| VF + VFIB + VFL | 105 | 8 | 4269.7 | 1255.6 (29.4%) | 2.9 s | 18 | 16 | 3 |
| VT + VFL + VF + VFIB | 195 | 22 | 10472.8 | 2218.7 (21.2%) | 3.4 s | 39 | 31 | 7 |
| Asystole | 12 | 6 | 795.3 | 3.2 (0.4%) | 11.4 s | 3 | 3 | 0 |

**The noise column is load-bearing and must not be dropped when this table is quoted.**
"15 VT episodes ≥30 s" reads as sustained analysable VT, but four of them are more than half
annotated noise (record 605: 452.2 s at 93.7% noise; 420: 669.8 s at 57.9%; 425: 96.8 s at
70.3%; 426: 102.3 s at 65.8%), and for the malignant set the worst is 419 VFL at 719.2 s and
92.6% noise. Usable sustained-episode counts are smaller than the ≥30 s column suggests.

Marker counts are not adjudicated clinical events, and raw marker totals (VT 93, VFL 98,
VF 9, VFIB 4 in the readiness audit) differ from episode counts because consecutive identical
states merge — the merge fires 53 times on real data, 52 of them the rhythm-re-annotated-
after-noise case. Neither figure is a patient count.

Negatives: 1603 ten-second windows across 15 records are pure sinus with no noise overlap
(1778 before the noise filter). That is the entire clean-negative supply, from an
all-abnormal cohort.

**Test status.** `tests/test_vfdb_episodes.py` pins the noise-does-not-terminate-rhythm
contrast, the unlabeled prefix and the late-first-marker case. It does **not** yet exercise
the merge step, `(NOISE` as the final marker, or two consecutive `(NOISE` markers, and the
suite has never been executed (see §7). The merge step is what actually realises "the previous
rhythm continues" when the same rhythm is re-annotated after noise, so a regression there
would change every count in the table above while the current tests stayed green.

## 4. What the data cannot support

- **No population-representative normal.** Every record was selected because the subject had
  sustained VT, flutter or fibrillation. False-alert rates per person-day cannot be estimated
  here at all, and 12 h 50 m of abnormal-cohort recording is not a seven-day observation.
- **No subject grouping.** With 22 records and no subject ids, a record-level split may still
  place two records from one person on both sides. Any split is reported as record-level, and
  the possibility of subject leakage is stated rather than dismissed.
- **No pulse information.** VT here is an ECG rhythm label. Pulseless VT and pulseless
  electrical activity cannot be distinguished from the ECG, so no cardiac-arrest detection
  claim can be made under any result.
- **No lead identity.** Channel-to-standard-lead mapping is unknown.

## 5. Proposed design, subject to approval

**Split.** Record-level grouped cross-validation over the 22 records, all folds fixed before
any training. Given 8 records carrying flutter/fibrillation and 19 carrying VT, folds are
stratified so each holdout contains at least one record of the primary target. Leave-one-
record-out is the fallback if fold counts prove degenerate. No record ever appears in both
train and holdout of the same fold.

**Target.** Primary endpoint is the combined malignant ventricular set (VT, VFL, VF, VFIB),
which is the only target present in all 22 records. VT alone and VF/VFL alone are secondary.
Asystole is reported descriptively only: 12 episodes in 6 records cannot support an endpoint.

**Input.** 10 s windows resampled 250 → 500 Hz to match the frozen input contract, with the
two channels placed into a declared pair of the twelve slots and the rest zero-filled. The
placement is an explicit assumption, not a measurement; at least two placements are evaluated
and reported side by side so the assumption's effect is visible. Windows overlapping noise
spans or `UNLABELED_START` are excluded and counted.

**Label.** A window is positive when the target set covers at least half of it; windows with
target coverage strictly between 0 and 0.5 are excluded from training and reported separately
at evaluation.

Under the combined target, and **after applying the exclusion stated in the Input paragraph
above**, this is **802 positive windows and 109 ambiguous**, against 1603 clean negatives — a
ratio of 1:2.00. Before the exclusion the same rule gives 1039 and 118, and that larger figure
must not be used to size the study: 237 of those 1039 (22.8%) overlap noise or the unlabeled
prefix, and 206 of them are entirely annotated noise. The counts are re-derivable from
`results/vfdb_episodes_20260910/window_labels_10s.csv` by filtering
`malignant_ventricular_VT_VFL_VF_VFIB_fraction >= 0.5` together with `noise_fraction == 0`
and `unlabeled_fraction == 0`.

⚠️ The shipped `results/vfdb_episodes_20260910/audit.json` reports the **un-excluded**
positive counts (`windows_fraction_ge_0.5`) while its negative count already applies the noise
filter, so the two classes in that artifact are not built under the same signal-quality rule.
That asymmetry is a defect in the audit script, recorded in §7; the corrected figures above,
not the artifact's, govern this protocol.

**Models compared.** (a) frozen ECG-FM backbone with a new rhythm head; (b) the frozen P1
backbone with a new adapter; (c) a small independent detector trained from scratch. Which is
better is unknown; the comparison is the experiment. The existing P1 output contract and all
frozen artefacts are untouched — this is a separate model version.

**Metrics.** Episode-level sensitivity, detection delay from episode onset, and false alerts
per hour of non-target recording, with the alert persistence and merge rules taken from
`records/wearable_acquisition_spec_v1.md` §4 so the two studies remain comparable. Window-level
scores are reported as diagnostics only.

**Exposure check before training.** VFDB is absent from the published ECG-FM pretraining split
and from the Challenge 2021 training set, but this must be re-verified against the pinned
split list rather than assumed, and the unresolved P1 warm-start lineage applies here too.

## 6. Sequence

1. Approval of this design by hemlo. **Training does not start before this step.**
2. Freeze folds, window rules and metrics; commit the fold file with its hash.
3. Exposure re-verification.
4. Baseline (c) first, since it is independent of the unresolved P1 ancestry.
5. (a) and (b), compared on the same folds.
6. Report including negative results, and an explicit statement of every limit in §4.

An additional bounded feasibility study on a second source with subject identifiers would be
required before any claim generalises beyond these 22 records; VFDB alone cannot support one.

## 7. Known defects in the shipped artifacts, and what has not been re-run

An adversarial review on 2026-09-10 found the following in `scripts/audit_vfdb_episodes.py`
and `results/vfdb_episodes_20260910/`. Every figure below was re-derived independently from
`window_labels_10s.csv` and `episodes.csv` before being accepted.

| Defect | Effect | Status |
|---|---|---|
| Positive windows counted without a noise filter while negatives apply one | 965 "full malignant" windows include 206 that are entirely annotated noise; only 732 are clean | **Fixed and re-run.** `clean_windows_*` fields now apply the same rule to both classes |
| `rhythm_episodes_total: 488` counts 22 `UNLABELED_START` pseudo-episodes | Real annotated episodes are 466; summing `seconds_by_rhythm` adds 1706.1 s of unannotated recording | **Fixed and re-run.** `annotated_rhythm_episodes: 466` added alongside |
| The `rule` string did not mention the end-of-record noise fallback | 29.2% of reported noise seconds are extrapolation, presented as rule-derived | **Fixed and re-run.** `rule_extrapolation` and `trailing_noise_extrapolation` added |
| `tests/test_vfdb_episodes.py` never exercised the merge step or either `(NOISE` edge case | A regression splitting the 53 merges would pass the suite | **Fixed.** Five tests added; 9 pass |

Resolved on 2026-09-11. The Smart App Control block that stopped the interpreter on 2026-09-10
at 16:06:42 cleared overnight, so the script was corrected, re-run, and the artifact now agrees
with this document on every figure above: 73 noise markers of 592, 466 annotated episodes,
1771.5 s of trailing extrapolation (29.21% of reported noise), 802 clean positive windows and
109 clean ambiguous against 1603 clean negatives. Full suite: 53 passed, 2 pre-existing
`test_embedding_contract.py` errors from torch 2.7's default `weights_only` loader, unchanged
from before this work.

Still open: the 52 review findings that were never adjudicated (see the lineage report §7).
