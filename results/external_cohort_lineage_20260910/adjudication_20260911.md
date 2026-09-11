# Adjudication of the 52 never-checked review findings

Date: 2026-09-11 KST. No model inference, no training, no external outcome was inspected.
Per-finding verdicts and evidence: `review_adjudication_20260911.csv`
(columns `adjudication_20260911`, `evidence_20260911`).

The 2026-09-10 adversarial review raised 62 findings. Ten were independently cross-checked
(7 upheld, 3 rejected); the other 52 were filed as rejected without ever being checked.
This pass re-derived each of those 52 from a primary artefact in this repository.

## Result

| Verdict | critical | major | minor | total |
|---|---:|---:|---:|---:|
| `upheld_open` — defect confirmed and still present | 0 | 16 | 12 | **28** |
| `upheld_partial` — substance confirmed, part fixed or wording inaccurate | 1 | 5 | 3 | **9** |
| `upheld_fixed` — confirmed, already corrected in the current artefacts | 3 | 4 | 6 | **13** |
| `rejected` — not supported by the primary source | 0 | 0 | 2 | **2** |

Fifty of the 52 had substance. The tally that discarded them was wrong in both directions it
could be wrong: it dropped 4 findings marked critical, three of which had already been acted on
without being credited, and it dropped 37 that are still live.

## The one that changes what the protocol can claim

**The 5c contamination risk is measurable, and it is 60%.**
`external_validation_protocol_v1.md` §1.2 names the risk that a PTB-XL-trained sibling model
(stage 5c) sits somewhere in the frozen model's ancestry, and leaves it at "absence of a
reference is not proof of absence." That risk does not have to stay unquantified.
`preprocess_ptbxl.py:134-150` splits patients with `np.random.default_rng(42).shuffle` and a
15/15/70 cut over all 18869 patients — from metadata alone, no waveforms, no inference.
Replaying it here:

| Quantity | Replay on 1.0.3 metadata | Recorded in `03_eval_results.md` |
|---|---:|---:|
| 5c train patients | 13209 | — |
| 5c train records | 15223 | — |
| of those, label-usable (NORM or MI/STTC) | 12849 | 12,847 |

The 0.016% gap is consistent with 5c having run on an earlier PTB-XL release. Against the F3
cohort:

> **1012 of the 1682 F3 records (60.2%), and 1005 of the 1673 F3 patients, were in the 5c
> PTB-XL training set.**

This is conditional on 5c being an ancestor of a07, which is exactly the thing the missing 5d
warm-start log cannot settle. But the measurement costs one metadata replay, and the protocol
neither performs it nor pre-registers performing it. Two further affirmative facts are also
absent from §1.2: 5c's training set was 12,847 PTB-XL records at `ptbxl_ratio=0.3`
(`03_eval_results.md:131`), and 5c warm-started from `lora_multisnr_best.pt`
(`03_eval_results.md:262`) — the same line 5d and 5f descend from, which makes 5c a sibling off
a shared parent rather than an unrelated model.

## Still open, grouped by what they threaten

**Cohort identity.** The identical-waveform filter is the only thing separating F3 from F2, and
it acts on exactly two records — 13803 and 15741. Neither they nor their partners (142, 11810)
appear in the 47-pair waveform sample; their identity is taken from the changelog. Only 5 of
the 36 claimed 1.0.2 duplicates are waveform-verified at all, and neither changelog states the
criterion by which duplicates were identified.

**Audit script guards.** No assertion guards any waveform-identity result — a run in which
every pair mismatched exits 0 and writes a normal-looking artefact. F2 and F3 have no
regression pin (the single cohort assertion is on F1). The 1.0.2 subset gate passes on an empty
parse while publishing `dropped_in_v102_changelog: 36`. `F3_records_with_identical_partner_in_test`
measures "has any partner", not "in test". The `diag_likelihood_min` sweep is invariant for four
of its seven keys and the key names do not say so. The `age > 89` branch conflates 204
genuinely-over-89 records with 89 that have no age at all, and has zero coverage in the sample.

**Label construction.** One clause — "form and rhythm statements count by presence" — carries
100% of the ST endpoint (93 of 93) and 98.9% of AF (90 of 91); a likelihood ≥ 50 rule would
leave 0 and 1. The other-rhythm exclusion set is enumerated nowhere in the protocol although it
swings the secondary cohort 12.41-fold (844 against 68). The frozen `ptbxl_f3_label_flags.csv`
ships a second five-class definition whose ectopy N is 59 against the protocol's 20 — 2.95x, not
the 2x the reviewer claimed. 563 of 1682 records (33.5%) carry no mapped code at all, which is
the unquantified reason specificity is incomparable.

**Analysis pre-specification.** §9.1 keeps only the 0.50 rung of the internal protocol's
mandatory 0.50/0.60/0.70/0.80/0.90 grid and does not disclose the narrowing. §9.2 compares
against "the internal ranking" while the same frozen file carries two rankings that agree at
Kendall tau 0.5714. No primary metric is declared, and §7 pre-labels the failure of the metric
§9.1 depends on as "expected". The pre-fixed threshold is the 95th-percentile order statistic
of 854 negatives for the smallest class and enters the bootstrap as a constant. Both abort
conditions are inert by construction. The protocol declares its gap list exhaustive at two while
`report.md` §5 lists three.

**Reproducibility of cited evidence.** `scripts/_diag_incart.py`, cited twice in
`03_eval_results.md` §11, exists in neither the working tree nor any git object. Every number in
the INCART inversion diagnosis rests on it — and `external_validation_protocol_v1.md` §6 rests
its "no scale correction is applied" decision on that diagnosis.

**Measurement wording.** Lead names are asserted as `aVR/aVL/aVF` for both sources; records500
uses `AVR/AVL/AVF`, so the literal per-record assertion §6 promises would abort every run
(0 of 40 pairs match literally; units differ too, `mv` against `mV`). "Pure sinus" negatives use
a 0.999 tolerance — 1603 windows, or 1602 if read strictly; the extra one is record 610 at
800.0 s. The INCART beat list omits exactly 159 annotations including the database's only
unspecified bundle-branch-block beat. Ventricular ectopy is 60 records, not the corrected 59
(record I32 carries `VBig` with 57 V beats and no `VEB`/`VPVC`). Record-level conduction codes
cover 7 records across 4 patients, which the report does not admit while admitting the
equivalent evidence for ST change. `report.md` §4 still prints "other rhythm codes 189" without
saying SR is excluded; including SR gives 1529.

## The two rejected

- *"The official rule ... is quoted but not preserved anywhere in the repository"* — false as
  worded. The sentence is in `audit_vfdb_episodes.py:3-5` and `emergency_rhythm_protocol_v1.md:45-47`.
  The narrower true point, which the finding does not make: the database page carrying the rule
  is the one input to the episode logic that `vfdb_readiness_20260910/source_manifest.json` does
  not hash.
- *"`float()` on the raw Challenge header age string will crash the run"* — not reproducible.
  All 47 downloaded headers carry numeric ages, and the Challenge convention for unknown is
  `NaN`, which parses without raising. Scope: those 47 headers; the rest were not downloaded, so
  this is unchecked rather than excluded.

## Two stale statements found while checking, outside the 52

Both concern the VFDB test suite, which now exists at 9 tests and passes
(`.venv` 3.10.21, `pytest tests/test_vfdb_episodes.py tests/test_external_cohort_lineage.py`
→ 16 passed):

- `report.md` §8 says "`tests/test_vfdb_episodes.py` — 4 tests, **written but never executed**."
- `emergency_rhythm_protocol_v1.md` §3 says the suite "does **not** yet exercise the merge step,
  `(NOISE` as the final marker, or two consecutive `(NOISE` markers, and the suite has never
  been executed."

All three named cases now have dedicated tests
(`test_same_rhythm_reannounced_after_noise_merges_into_one_episode`,
`test_trailing_noise_marker_extends_to_end_of_record`,
`test_consecutive_noise_markers_each_end_at_the_next_marker`). The 2026-09-10 review upheld a
finding that the claimed coverage *overstated* what the tests exercised; the same two sentences
now *understate* it. Left uncorrected here because correcting protocol text is outside this
pass's scope.

## Method and scope

Every verdict names what was run or read. Verdicts are keyed to findings by exact title prefix
with a uniqueness assertion, so a later edit to `review_findings_20260910.csv` cannot silently
re-point one. Nothing in the frozen protocol, the audit scripts or their outputs was modified.
No external cohort outcome was inspected; the protocol remains frozen and unexecuted.
