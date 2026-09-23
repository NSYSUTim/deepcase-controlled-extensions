# RC-CR design-only feasibility protocol

## Status and scope

This protocol replaces the unsuccessful training-only focal auxiliary head with
a reject-preserving, rule-conditioned contextual ranker.  AIT-ADS has already
been inspected in the preceding study, so every result here is exploratory
design evidence.  No AIT-ADS result from this protocol may be represented as a
fresh confirmatory test.

## Fixed pipeline

1. Load the previously frozen event-only DeepCASE checkpoint.
2. Compute the exact attention-query representation with `q=100`.
3. Reconstruct the original aligned-v2 DBSCAN/KDTree Interpreter.
4. Preserve every original negative Reject code.
5. Preserve the exact number of accepted alerts marked Incident at the paired
   DeepCASE operating point.
6. Rank only accepted alerts using train-only revealed labels.

The primary candidate is `within_rule_pairwise`:

\[
S_t=b_{r(t)}+w^Tz_t^{(100)},\qquad
L=\log(1+\exp[-(S_+-S_-)]).
\]

Every positive/negative pair has the same current rule, so the rule prior
cancels inside the ranking loss.  Incident groups are sampled uniformly.
`context_bce`, `rule_context_bce`, and `rule_prior` are design baselines, not
parts of the proposed objective.  Focal loss is absent.

## Leakage controls

- Event vocabulary, rule priors, ranker weights, and DBSCAN clusters use only
  training scenarios.
- The validation scenario selects weight decay within each objective.
- The low-workload operating point is selected only from DeepCASE validation
  workload, independently of RC-CR outcomes.
- Test labels are used only for the final design-fold metrics.
- Ties use a fixed pseudorandom key independent of outcome labels.
- All methods share the same checkpoint, q=100 vectors, Reject mask, and exact
  review count.

## Stop rules

Before model fitting, at least two genuinely mixed training rules and two
incident groups must have revealed same-rule positive/negative pairs.  Passing
this gate establishes only learnability, not superiority.

The first, smaller design fold is `russellmitchell`, seed 1729.  The larger
`fox` fold is run only if the primary pairwise model exceeds both DeepCASE and
the rule-only baseline and achieves an absolute alert-level recall gain of at
least 0.01 at identical workload.  The threshold is not lowered after seeing
an outcome.  Failure stops the AIT-ADS feasibility experiment.

Even if both design folds passed, a superiority claim would still require a
separately registered, untouched external-environment evaluation.
