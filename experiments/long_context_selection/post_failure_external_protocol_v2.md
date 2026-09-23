# External HDFS confirmation protocol v2

Version 1 remains a failed development gate because a purely relative safety
limit is degenerate when baseline validation has zero false escalations. This
v2 protocol was finalized after the AIT development analysis but before any
HDFS file was downloaded or inspected. It does not modify the failed primary
AIT confirmation.

## Frozen method and data roles

`anchor5_assoc5` always returns ten positions: the newest five candidate slots
plus five older candidates chosen by absolute target-conditioned association
evidence fitted from development labels only. Temporal order is restored after
selection. The original DeepCASE `last10` is the baseline. Both use the same
event-prediction model, training budget, interpreter, policy labels, random
seeds, and seven-point operating grid.

We use the exact three HDFS files linked by the DeepCASE authors to the DeepLog
repository. A line is a sequence and no context crosses lines. HDFS has no
timestamps, so it tests position-budgeted selection but not the AIT one-day
timeout. `hdfs_train` is normal training data. Exact line SHA-256 after removing
only CR/LF determines the split of both official test files: modulo-ten buckets
0--3 development, 4--5 validation, and 6--9 final test. Hashing keeps exact
duplicate official-test sequences in one partition, including identical event
patterns with different labels. Such contradictory patterns are retained and
reported because they reveal an information limit, not a parsing error. Exact
patterns also found in official training are reported and excluded in a
secondary sensitivity analysis; the primary analysis follows the authors'
official train/test roles and scores every final sequence.

The event vocabulary, association table, neural ContextBuilder and Interpreter
policy are fitted from official training plus development only. Validation
labels may select an eligible operating point. Final labels are inaccessible
to all fitting and selection code and are used only once by the final analyzer.

## Safety-aware operating point

The baseline is `(eps=.1, threshold=.2)`. For validation false-escalation rate
`b` per 1,000 normal events, the selector eligibility limit is:

- `b + 1` when `b < 1/1000`;
- `1.05 * b` otherwise.

This hybrid non-inferiority margin avoids division by zero while remaining an
absolute one-per-thousand bound in the low-rate regime. Among eligible points,
selection minimizes absolute distance to baseline workload on the unlabeled
final stream, then prefers lower workload and frozen grid order. No eligible
point is a seed-level failure.

## Endpoints and inference

Primary recall is automatic anomalous-event recall; Reject is unresolved and
never credited. Review workload is automatic Anomaly plus Reject. False
escalation is a normal event automatically labeled Anomaly.

For each final sequence, predictions are averaged over three neural seeds; the
seeds are technical repeats, not independent samples. A paired bootstrap
resamples anomalous sequences for recall gain and normal sequences for false
escalation non-inferiority. Exactly 20,000 resamples use seed 260906.

External success requires all of the following:

1. the 95% paired sequence-bootstrap lower bound for strict recall gain exceeds
   zero;
2. the 95% upper bound of selector false escalation minus the hybrid safety
   limit is at most zero;
3. every seed has an eligible selector point;
4. both methods use exactly ten positions and every final sequence/event is
   scored.

Otherwise the HDFS external claim fails. No post-hoc threshold, split, context
allocation, or alternative metric may reverse this decision.
