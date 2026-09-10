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
| Annotations | 592 rhythm-change markers; **no beat labels** |
| Subjects | Not identified. Records cannot be grouped by patient. |

The official overview describes half-hour records; the headers say 35 minutes. Header-derived
lengths are used.

## 3. Episode construction rule, fixed

The official database page states: "The rhythm change annotations are placed at the beginning
of the episode of the indicated rhythm. The previous rhythm continues during episodes marked
by (NOISE; the noise ends at the time of the next annotation."

Implemented in `scripts/audit_vfdb_episodes.py` and pinned by `tests/test_vfdb_episodes.py`:

- A rhythm marker sets the rhythm state from its sample until the **next rhythm marker**.
- A `(NOISE` marker opens a noise span ending at the next marker of any kind and **does not
  change the rhythm state**. A naive previous-annotation-to-next-annotation parser labels the
  post-noise interval as NOISE and is wrong; one regression test encodes exactly this contrast.
- Samples before the first rhythm marker are `UNLABELED_START`, never assumed normal. Record
  605 opens with `(NOISE`, and one record's first marker arrives 13999 samples in.

Under this rule the 592 markers yield the following, with 73 noise spans:

| Target set | Episodes | Records | Total seconds | Median | ≥10 s | ≥30 s |
|---|---:|---:|---:|---:|---:|---:|
| VT | 90 | 19 | 6203 | 4.0 s | 21 | 15 |
| VF + VFIB + VFL | 105 | 8 | 4270 | 2.9 s | 18 | 16 |
| VT + VFL + VF + VFIB | 195 | 22 | 10473 | 3.4 s | 39 | 31 |
| Asystole | 12 | 6 | 795 | 11.4 s | 8 | 3 |

Marker counts are not adjudicated clinical events, and raw marker totals (VT 93, VFL 98,
VF 9, VFIB 4 in the readiness audit) differ from episode counts because consecutive identical
states merge. Neither figure is a patient count.

Negatives: 1603 ten-second windows across 15 records are pure sinus with no noise overlap.
That is the entire clean-negative supply, from an all-abnormal cohort.

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
at evaluation. Under the combined target this is 1039 positive windows, 118 ambiguous.

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
