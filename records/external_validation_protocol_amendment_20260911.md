# External protocol clarification and stop record — 2026-09-11

Status: pre-outcome clarification; **external inference is held at the identity gate**.
No PTB-XL model probabilities or performance statistics were produced or inspected.
Read with `external_validation_protocol_v1.md`; dated corrections here take precedence
over its historical execution status and broad identity claims. Frozen a07 weights,
eight candidates, F3 IDs and label flags remain unchanged. This is not public preregistration.

## 1. Decisions, reasons, alternatives and consequences

| Decision | Primary evidence and reason | Alternative not adopted | Consequence |
|---|---|---|---|
| Retain the published F3 record list | Recomputed F3 remains1682 records/1673 recorded patient IDs; both frozen CSV hashes match | Remove records after seeing model scores, or silently create F4 | Conditional comparison stays tied to its original cohort; no independent-patient claim |
| Quantify stage5c risk but do not assert proven contamination | `stage5c_cohort_overlap.json`: current-version replay12849 usable training records versus historical recorded12847; F3 overlap1012 records/1005 patients | Treat replay as a historical training log or assume shared parent means direct ancestry | About60.17% overlap is conditional on replay fidelity and5c→5d inheritance; actual a07 exposure is unresolved |
| Test all38 catalogued deletion relations | Prior47 comparisons included only7 deletion pairs; the two F3-specific exclusions had not been directly checked | Extrapolate exact identity from the earlier sample | New78-pair audit finds76 exact and2 nonexact; exact-identity gate fails |
| Preserve the failing identity gate | `identity_gate_diagnosis.json` and936 lead-level difference rows | Set a favorable correlation tolerance after observing differences | No external inference/threshold calibration proceeds in this execution; a documented relation-specific gate design remains necessary |
| Reconstruct test before trusting validation inputs | Surviving936 test arrays and rawCPSC headers/waveforms are available | Treat regenerated mc_ml1033 test as the original936 test | Reconstructed signals and labels match original file bytes; this verifies the tested preprocessing path, not patient independence |
| Preserve ID provenance | Original936 IDs determine order and are copied with hashes | Claim that copied IDs independently reproduce the original split | ID order is retained; validation933 remains a reconstruction, not a recovered original waveform file |
| State primary operating-point performance explicitly | Per-disease coverage was the internal objective; aggregate F1 can hide a weak class | Select a metric after PTB outcomes | Sensitivity and achieved specificity at the CPSC-validation threshold are primary; AUROC and F1 are supporting metrics |
| Make stochastic summaries explicit | Legacy feature masking remains random in eval | Remove stochastic masking or select the best seed | Seed42 provides the primary point table and patient bootstrap; ten seeds30000–30009 provide stability summaries |
| Report model-selection and threshold uncertainty | Validation933 selected a07 and is reused to fit operating points | Describe thresholds as known population95%-specificity points | External intervals are conditional on fitted thresholds/model/candidates; they do not propagate calibration or candidate-selection uncertainty |
| Keep input physical units | WFDB physical signals are mV; PTB uses AVR/AVL/AVF while CPSC uses aVR/aVL/aVF | Normalize amplitude based on external outcomes | Canonicalize names by case only; assert order, units, sample rate and shape; no waveform scaling |

## 2. Boundaries of the evidence

Three gaps remain separate: (a) the published ECG-FM list is not a runtime training log,
(b) the5d warm-start asset/log is absent from the searched locations/history, and
(c) stage5c used a PTB-XL split but its inheritance into5d/a07 is unverified.
The5c binary model and5d/5f have records of a common earlier parent; that does not prove
that one branch inherited the other's updates. Current-version metadata replay yields
two more usable records than the historical12,847. Version changes and preprocessing
exclusions are possible explanations, not established causes.

The new waveform audit directly confirms142→13803 and11810→15741, the two F3 exclusion
relations. But3832→15768 and11838→11839 differ digitally, despite equal gain/baseline and
case-normalized lead names. The old global wording “38 identical-waveform duplicates” is
withdrawn. Use “38 catalogued drop/keep relations, of which36 matched exactly in this audit.”
This relation list is not an exhaustive near-duplicate search over all records. Patient IDs
are database identifiers, not verified biological identity. The unchanged F3 set does not
establish independence of a07 from these subjects.

## 3. Label and estimand clarification (no label-file changes)

- `other_rhythm` excludes AFLT, SVARR, SVTAC, PSVT, SARRH, STACH, SBRAD, PACE, BIGU and
  TRIGU. SR is exempt. Counts are189 with the exemption versus1529 including SR;1342
  records carry SR. This is an endpoint design choice, not evidence that all excluded
  rhythms are clinically unimportant.
- Primary disease flags are nonexclusive. Secondary844/843 requires exactly one mapped
  group including normal, no other conduction flag and no excluded rhythm flag. Its
  counts areNSR696/AF38/ST36/conduction54/ectopy20. The full-cohort
  `exclusive_priority_label` column instead applies historical priority to all1682 records;
  it is descriptive and **must not be used as the secondary-cohort inclusion mask**.
- F3 contains563/1682 (33.47%) records with no mapped group. They remain negatives for
  individual mapped endpoints, with the unmapped flag retained; absence of a mapped code
  does not mean absence of cardiovascular disease.63 records have multiple disease flags.
- AFIB presence contributes91 positives; applying diagnostic likelihood≥50 incorrectly
  would keep only1. STD_ presence contributes93; STE_ contributes0. Those are rhythm/form
  codes. The diagnostic-likelihood sweep cannot test sensitivity of their presence rule.
  AF/ST/ectopy/other-rhythm are structurally unchanged; conduction is empirically unchanged
  in these data. Normal counts839/834/633 at1/50/100 demonstrate a different behavior.
- Label ischemia in historical P1 tables is a STD/STE surrogate. Here it is **ST depression**,
  not acute myocardial ischemia or infarction. No emergency severity inference follows
  from treating the four disease groups as four classes.

## 4. Analysis-plan clarification before any outcomes

These details resolve under-specification in v1; they are **plan amendments**, not corrections
to already measured performance. They must be committed before thresholds or external inference.

Threshold calculation, once the identity hold is resolved: use seed42 validation probabilities
per candidate; one-vs-rest ROC with `drop_intermediate=False`, predicted positive when
`score >= threshold`. Among operating points with empirical specificity≥0.95, maximize
sensitivity; on a sensitivity tie choose the largest threshold. An infinite threshold is
serialized explicitly as a named sentinel and yields no positive predictions. Require both
classes and finite input probabilities. Do not interpolate a threshold or refit on PTB-XL.
Store TP/FN/FP/TN and validation class counts alongside all32 thresholds, source hashes,
model/environment/seeds and elapsed time. Validation is933 records with class counts
136/176/163/379/79; per-disease negative counts757/770/554/854. Sampling uncertainty of
these empirical order statistics is not covered by the external-only bootstrap.

Use fixed thresholds for all repeat seeds. Primary intervals use seed42 predictions,
2000 patient-cluster resamples, seed31415, paired candidate/reference draws, with missing-class
replicates skipped and counted without redraw. Report95% percentile intervals. This measures
record sampling conditional on database patient IDs; it does not validate those IDs.
Secondary Macro-F1 always averages all five classes; missing required classes in a bootstrap
draw make that draw invalid, rather than changing the average's class set.

Restore the complete internal sensitivity-floor grid0.50/0.60/0.70/0.80/0.90. At each floor,
all four diseases must have both mean and minimum sensitivity over the ten seeds at or above
that floor. Report failed floors and weakest diseases, not only a successful rung.

Rank agreement uses secondary Macro-F1 averaged across seeds30000–30009 versus the internal
`repeat_mean_macro_f1` column. Seed42 ranking is separately descriptive. Kendall tau-b handles
ties. For its exploratory interval, patient resampling is shared across all candidates and
seeds, internal ranks are fixed, external ranks are recalculated from seed-mean F1; invalid
draws or undefined tau are skipped and counted. No equivalence claim follows from a wide
interval or an interval containing zero.

## 5. Corrections to other evidence used in the argument

INCART record-level ventricular ectopy is60, after adding VBig (I32) to the code matching.
Record-level conduction codes BBB/RBBB/NSIVCB/MoI occur in7 records/4 patients; that does not
create temporal labels for the restricted three-class conduction endpoint. AF positives in
the historical replay come from2 of32 source groups. Its7811 windows=540+7271 are an
annotation-based reconstruction; the original processed arrays are absent and have not been
byte-compared. Remaining159 annotations outside the six-item historical list comprise
147 other beats (j92,n32,S16,Q6,B1) and12 rhythm markers. Do not call the list exhaustive.

The INCART scaling experiment's0.281→0.274 AUROC is a recorded historical observation in
`03_eval_results.md`; the cited `_diag_incart.py` is absent from the searched worktree and
Git object history. It is not a freshly reproduced demonstration of a representation cause.
The no-scaling decision rests on the frozen input contract, with amplitude comparison as
descriptive context. An amplitude match or a source-check pass is not deployment validation.

VFDB1603 negative windows have≥99.9% sinus occupancy and no annotated noise. Strict100%
occupancy yields1602; record610 at800s has0.9996. The tolerance is retained and named,
not silently changed. No model has been trained on these episode labels. The9 VFDB tests
were run and passed; previous “written but never executed” wording is obsolete.

## 6. Present execution state and next gate

Test reconstruction: PASS,936 records, zero signal/label differences, zero failures/skips.
Waveform comparison: FAIL exact-identity gate,76/78 exact. Acquisition attempts preserved:
20 timeout failures, then2 timeout failures, then zero missing files; these are attempt
counts, not22 distinct missing records. No external inference and no new training.
Validation reconstruction, calibration, threshold commit and external metrics remain undone.

Next: review the two nonexact drop/keep relations and define any replacement gate explicitly,
before viewing external scores. A conservative catalogued-relation exclusion may be defensible
without claiming exact identity, but is not silently substituted here. Retain the passing
critical pairs, failed pairs, provenance and the unchanged F3 hashes. All proposed later
analysis remains conditional on this prerequisite. No new P1 push is authorised by this record.
