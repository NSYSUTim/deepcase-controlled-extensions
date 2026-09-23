# Frozen q=100 rescue protocol

Frozen on 2026-09-03 before producing any q=100 result for the six
confirmatory scenarios.

## Status and scope

The completed q=0 experiment remains the primary experiment and is not
overwritten.  This q=100 study is a transparently post-primary, secondary
rescue experiment motivated by a fidelity issue: the upstream DeepCASE
`Interpreter.attention_query` default is 100 optimization iterations, whereas
the first primary experiment used zero iterations for computational reasons.
Query iterations optimize attention for the observed target event; they are
not analyst queries and do not alter detector rules, alert order, labels,
prevalence, or Reject semantics.

The six q=100 confirmatory outcomes have not been computed at the time of this
freeze.  Existing q=100 results for `russellmitchell` are exploratory design
results only.

## Fixed data and estimand

- Data: the existing canonical AIT-ADS strict-past contexts and event-level
  incident labels.
- Training label ratio: 1 incident label to 1000 non-incident labels.
- Design scenarios: `russellmitchell` and `fox`.
- Confirmatory scenarios: `harrison`, `santos`, `shaw`, `wardbeck`, `wheeler`,
  and `wilson`.
- Confirmatory seeds: 1729, 2718, and 31415.
- Rules, duplicate alerts, chronological stream, natural validation/test
  prevalence, Interpreter clustering, and all three Reject conditions remain
  unchanged.
- Workload is the test alert-review rate: every Incident or Reject prediction
  requires review.  It is a proxy for analyst capacity, not elapsed analyst
  minutes.
- Primary q=100 estimand: scenario-equal mean difference in event-level
  incident-alert recall between the selected incident-aligned method and
  DeepCASE at no greater alert-review workload.  This is not unique-incident
  correlation recall.

## Bounded design selection

Only these neural candidates are eligible:

1. `deepcase`: next-event objective only;
2. `proposed`: group-balanced supervised sampling plus binary incident focal
   auxiliary loss;
3. `aux_group_balanced_bce`: the same group-balanced sampling and incident
   head, replacing focal loss with BCE.

All three use q=100 and identical full-compute model settings.  Candidate
selection uses only the two design scenarios, seed 1729, at the closest tested
workload not exceeding DeepCASE's default `(eps=0.10, threshold=0.20)` review
rate.  The incident-aligned candidate with the larger mean design-scenario
incident recall is selected; a tie within 1e-12 is broken by lower mean
workload, then higher relaxed F1, then the simpler BCE loss.  If neither
incident-aligned candidate has positive mean recall difference from DeepCASE,
the rescue stops without inspecting q=100 confirmatory outcomes.

No additional loss, sampler, threshold grid, model architecture, seed, or
label definition may be introduced after this freeze.

## Confirmatory execution and decision

The selected candidate and DeepCASE are evaluated on all six confirmatory
scenarios and all three fixed seeds with q=100.  Random seeds are averaged
within scenario; the six scenarios are the independent units.  Operating
points are selected using review counts only, never test incident outcomes.

Evidence for a q=100 improvement requires all of the following:

1. all six scenario pairs are feasible at matched workload;
2. scenario-equal mean delta incident recall is positive;
3. the 95% scenario bootstrap interval has a lower bound above zero;
4. the one-sided exact sign-flip p-value remains below 0.05 after Holm
   correction for the completed q=0 primary and this q=100 secondary test;
5. mean workload is no greater than DeepCASE's workload;
6. the improvement is practically material: at least 0.01 absolute recall,
   or at least 10% relative workload reduction at non-inferior recall (a
   non-inferiority margin of 0.005 absolute recall);
7. the training-defined ambiguous-rule diagnostic is reported and is not
   omitted if unfavorable.

Passing these criteria supports only a dataset-scoped secondary result for
AIT-ADS under q=100.  It does not establish real-SOC generalization.  Failure
of any criterion stops the superiority claim; the result remains a transparent
robustness/null finding.

## Design selection result (recorded after the freeze)

The two full-compute design folds completed with q=100 and seed 1729.  At the
matched DeepCASE-default workload, the mean incident-alert recalls were:

- DeepCASE: 0.9840665280;
- group-balanced focal (`proposed`): 0.9843479004;
- group-balanced BCE (`aux_group_balanced_bce`): 0.9836343008.

The corresponding mean recall differences from DeepCASE were +0.0002813724
for focal and -0.0004322272 for BCE.  Under the frozen selection rule, focal is
therefore the sole q=100 confirmatory candidate.  The positive difference
passes only the direction gate; it is far below the final 0.01 practical-effect
criterion and is not confirmatory evidence.

## Confirmatory result and stop decision (recorded after evaluation)

All 36 expected q=100 result records (six scenarios, three seeds, two methods)
completed and passed the fail-closed record audit.  All 36 result-record
checkpoint hashes matched the loaded checkpoint files.  Matched-workload
selection was feasible for only 17 of 18 planned scenario-seed pairs:
`wheeler` seed 2718 had a DeepCASE budget of 0.71027053, while the lowest
tested proposed workload was 0.71527247.

Across the 17 available pairs, after averaging seeds within scenario and then
weighting the six represented scenarios equally, the descriptive results were:

- DeepCASE recall 0.98987898 and workload 0.46807583;
- proposed recall 0.98152843 and workload 0.46376346;
- delta recall -0.00835054;
- delta workload -0.00431238 (a 0.92% relative reduction).

Only `wheeler` had a positive scenario recall difference (+0.00073510); the
other five scenario differences were negative.  The training-defined
ambiguous-rule subset had descriptive delta recall -0.01841755 and delta
workload -0.03179099.

The planned seed panel is incomplete, so confirmatory CI and p-values are not
computed; doing inference after dropping the workload-infeasible pair would
condition on a method-dependent outcome.  The rescue fails the feasibility,
positive-effect, inferential, and practical-effect gates.  It passes only the
available-pair mean-workload constraint, and the required ambiguous-rule result
has been reported despite its unfavorable direction.  The superiority claim
is stopped.  These data must not be used for another round of loss, sampler,
or threshold selection.
