# Wearable acquisition and seven-day evaluation specification v1

Date: 2026-09-10. Status: specification only. No wearable recording exists in this
repository, and nothing here is a measured result. This document defines what must be
collected and how it will be scored so that a future study can be compared with the
completed internal electrode exploration without re-deriving the rules after seeing data.

## 1. Why the current evidence cannot answer the seven-day question

The internal test is 936 separate 10-second recordings (2 h 36 min nominal input including
padding); the full local CPSC set is 6877 short recordings totalling about 30 h 28 min.
Neither is one person observed continuously. A classification false positive on a 10-second
record is not an alert, so multiplying a record-level false-positive rate by 60480 (the number
of 10-second windows in 7 days) is not a valid alert-per-day estimate and must not be reported.

## 2. Data to be collected per participant

| Item | Requirement |
|---|---|
| Person identifier | Pseudonymous `person_id`; one key per participant; all files keyed by it |
| Duration | Target 7 x 24 h continuous wear per participant; record actual wear time |
| Wearable raw signal | Unfiltered ADC samples from every measured contact pair, sample rate, ADC gain/baseline, units, dropped samples/gaps with timestamps |
| Electrode geometry | Actual contact locations for each measured pair; which standard lead (if any) each pair approximates; reference electrode and DRL/ground presence and location; electrode material and skin preparation |
| Simultaneous reference ECG | Clinical 12-lead Holter or equivalent, worn concurrently for at least a defined reference window per participant (target: full duration; minimum: two 24 h periods including sleep) |
| Time synchronisation | Shared clock or event marker at start/end; measured residual offset and drift; sync error budget stated before analysis |
| Contact and motion | Per-channel impedance/contact-quality signal if available; tri-axial accelerometer; device on/off events |
| Annotations | Board-certified reader(s) mark episode start/end for AF, ST change, conduction abnormality, ectopy, and separately VT/VF/asystole; confidence; unreadable segments marked explicitly |
| Consent and approval | Institutional review, informed consent, and reader authorisation are prerequisites and are not claimed by this document |

## 3. Reference-annotation rules fixed before analysis

- Episode = contiguous reader-marked interval on the reference ECG. Minimum episode duration
  per class is fixed in advance (proposal: AF 30 s, VT 3 beats, VF/VFL any, ST change 60 s,
  conduction and ectopy per-beat labels aggregated to 10-second windows).
- Unreadable reference segments are excluded from both numerator and denominator and their
  total duration is reported.
- Reader disagreement is resolved by a documented adjudication rule, not by the model.

## 4. Alert rules fixed before analysis

- Window: 10 s, non-overlapping, 500 Hz, 12 x 5000 input with unused leads zero-filled
  exactly as in the frozen electrode runner.
- Persistence: an alert requires the per-class score to exceed its fixed threshold in `k`
  of the last `n` windows (proposal k=2, n=3); thresholds come from the CPSC validation split
  (see external validation protocol v1), not from the wearable data.
- Merge: alerts of the same class separated by less than 60 s form one alert.
- Refractory: after an alert, the same class cannot re-alert for 5 min.
- Suppression: windows flagged as poor contact or device-off produce no alert and are
  counted in unusable time.

## 5. Metrics

| Metric | Definition |
|---|---|
| Episode sensitivity | Episodes with at least one overlapping alert / all episodes (per class, per participant, pooled with patient-clustered intervals) |
| Missed episodes | Count and duration distribution |
| Detection delay | Time from episode onset to first alert; report median and 90th percentile; state that window length, update period, inference time and persistence all contribute |
| False alerts per participant-day | Alerts with no overlapping episode / analysable wear days |
| Analysable wear time | Fraction of elapsed time with device on and contact quality acceptable; report signal-loss and motion-flag fractions |
| Reference coverage | Fraction of wear time with valid reference ECG |

Per-class results are reported separately; VT/VF/asystole cannot be scored by the current
four-group model and require the separate emergency-rhythm study.

## 6. Sample size

The number of participants and required episode counts are derived from the target
confidence-interval width for episode sensitivity of the rarest class, expected prevalence,
and expected dropout. No number is fixed here because no prevalence estimate for the target
population exists in this repository; deriving one from the internal test set is not valid.

## 7. Traceability

Every analysis run records: device firmware, electrode lot, model checkpoint SHA-256
(`287148bf…08dd` for the frozen a07 model), threshold file hash, alert-rule parameters,
reader identities (pseudonymous), sync error measurements, and the exact list of excluded
segments. Results are reported with the acquisition limits above; hardware optimality,
early prediction and clinical suitability are not claims this study design can support.
