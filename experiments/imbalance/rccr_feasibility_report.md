# RC-CR feasibility result: stopped after first design fold

## Decision

The replacement method did **not** pass the pre-specified launch gate on the
smaller `russellmitchell` design fold.  The `fox` fold and any confirmatory run
were therefore not started.  Lowering the +0.01 threshold, selecting a more
favourable test operating point, or tuning on this outcome would invalidate
the stop rule.

## Identifiability result

For the `russellmitchell` fold at 1:1000 revealed-label ratio:

- 73 training rules were observed;
- only 2 rules had full-training incident rates between 5% and 95%, at least
  50 positives and 50 negatives, and at least one revealed label of each class;
- these two rules supplied 5 revealed incident groups and only 8 of the 99
  revealed positives (8.08%);
- using revealed data alone, the learner could form pairs for 93 positives,
  41 incident groups, and 16 rules, but most of these rules were nearly pure
  under the diagnostic full-training truth.

Thus the data technically passed the minimum 2-rule/2-group gate, but the
identifiable contextual signal was narrow.

## Frozen q=100 result

The validation-selected lowest-workload point was `eps=0.5,
threshold=0.01`.  DeepCASE already achieved test recall 0.998814 there, leaving
only 0.001186 maximum possible absolute improvement.  All seven validation
operating points had less than 0.01 recall headroom, so the practical-effect
criterion could not be demonstrated within the registered grid.

| Endpoint | Method | Workload | Recall | Delta vs DeepCASE | Precision | Reject |
|---|---|---:|---:|---:|---:|---:|
| Lowest validation workload | DeepCASE | 0.264096 | 0.998814 | — | 0.910293 | 0.036997 |
| Lowest validation workload | Rule prior | 0.264096 | 0.997446 | -0.001368 | 0.909046 | 0.036997 |
| Lowest validation workload | Context BCE | 0.264096 | 0.970535 | -0.028280 | 0.884519 | 0.036997 |
| Lowest validation workload | Rule + context BCE | 0.264096 | 0.997446 | -0.001368 | 0.909046 | 0.036997 |
| Lowest validation workload | Within-rule pairwise RC-CR | 0.264096 | 0.997446 | -0.001368 | 0.909046 | 0.036997 |
| DeepCASE default | DeepCASE | 0.275975 | 0.968619 | — | 0.844777 | 0.085368 |
| DeepCASE default | Rule prior | 0.275975 | 0.967707 | -0.000912 | 0.843981 | 0.085368 |
| DeepCASE default | Context BCE | 0.275975 | 0.968801 | +0.000182 | 0.844936 | 0.085368 |
| DeepCASE default | Rule + context BCE | 0.275975 | 0.967433 | -0.001186 | 0.843743 | 0.085368 |
| DeepCASE default | Within-rule pairwise RC-CR | 0.275975 | 0.968254 | -0.000365 | 0.844459 | 0.085368 |

Incident-group recall was 1.0 for every method and endpoint, so that metric
also had no discriminative headroom on this fold.  On the 486-row ambiguous
rule test stratum, the low-workload pairwise recall was 0.688889 versus
DeepCASE 0.822222; at the default endpoint both were 1.0.

## Integrity checks

- The q=100 fit, revealed, validation, and test caches are bound to checkpoint
  SHA-256 `33df33f48045d617f32597e9f8a03846318896b74c3a919db460eb5a395337b8`.
- Seventy DeepCASE metric values reconstructed by the new runner match the
  previously frozen q=100 source result exactly; maximum absolute difference
  is 0.
- All RC-CR predictions preserve the original Reject count and exact Incident
  review slots by construction.
- The complete local research test suite passes: 22/22 tests.

## Interpretation

This result rejects the current linear RC-CR candidate as a route to a +0.01
AIT-ADS improvement.  It does not establish that all contextual ranking is
impossible.  It shows that the present combination of extremely sparse
revealed positives, nearly rule-determined labels, and ceiling-level DeepCASE
recall cannot support the intended superiority claim.

The scientifically valid next prerequisite is a new environment in which the
same detector rules produce substantial incident and non-incident populations
and where the baseline has meaningful recall headroom at a realistic review
capacity.  A new architecture should not be selected on the already inspected
AIT-ADS outcomes.

Exact machine-readable results are stored under
`results/rccr_feasibility_v1/design/russellmitchell/ratio_1000/seed_1729.json`;
hash-bound q=100 caches are stored in the corresponding `cache` directory.
