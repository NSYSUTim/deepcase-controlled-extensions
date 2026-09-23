# Information-aware context selection: frozen research protocol

## Status and scope

This protocol is frozen before any natural-context displacement or selector
outcome is computed for this study.  AIT-ADS has been used by earlier work in
this repository, so this is an internal protocol freeze rather than an
external preregistration.  Existing results are not used to choose the method
or its operating point.

The study asks whether the fixed last-10 retrieval rule, rather than merely
the downstream attention mechanism, discards useful strictly-past context in
AIT-ADS.  It does not assume that benign alerts displace attack alerts.  That
mechanism must first be observed.

## Data and labels

- Source: the existing canonical AIT-ADS table built from the authors'
  alert-level CSV files.
- Primary label: `event_incident`, traced by the dataset authors to labelled
  source log lines or flows.
- Primary incident unit: `event_incident_group`.
- `window_incident` is not substituted for the primary label.
- Context is same-machine, timestamp-strictly-earlier, at most one day old.
  Rows tied at one-second resolution cannot observe one another.
- Alert multiplicity is preserved.
- Candidate-pool caps: 20, 50, and 100.  The primary selector budget is 10.

## Independent units and roles

- Exploratory design scenarios: `fox`, `russellmitchell`.
- Confirmatory scenario units: `harrison`, `santos`, `shaw`, `wardbeck`,
  `wheeler`, `wilson`.
- Final inference is paired by complete held-out scenario, never by alert row
  or random seed.
- Scenario-disjoint evaluation is mandatory.  Test labels may score a frozen
  method but may not fit event frequencies, association scores, selector
  parameters, event vocabulary, model parameters, clusters, or policies.

## Questions and gates

### Gate A: retrieval-opportunity audit (descriptive, not sufficient)

For incident targets with at least one labelled incident predecessor in the
last-100/one-day same-machine candidate pool, report:

1. any-incident displacement@10: no incident predecessor appears in last-10;
2. same-group displacement@10: no predecessor from the target's incident
   group appears in last-10;
3. last-10 event-type redundancy and temporal span;
4. predecessor coverage as the cap grows from 10 to 20, 50, and 100.

No selector is declared useful from these labels.  They are an oracle audit
only.  A displacement estimate below 0.10 is evidence against the proposed
mechanism, but is not alone a stop rule because non-incident precursors may
still predict the target event.

### Gate B: model upper bound (selector launch decision)

On both design scenarios compare paired versions of the original event-only
DeepCASE pipeline using:

- `last10`;
- `last100` (capacity upper bound, not budget matched);
- `oracle10` (label-using diagnostic, never a deployable method).

The selector is launched only if at least one upper bound improves the
pre-specified strict automatic incident recall by at least 0.03 over last10 on
each design scenario, or by at least 0.05 on one design scenario without a
negative change on the other.  The comparison uses the same event-training
budget, cluster-policy label budget, Interpreter grid, and workload-matching
rule.  If neither upper bound passes, the natural AIT-ADS selector hypothesis
is recorded as failed and learned-selector development stops.

Before any Gate-B model outcome was generated, the following implementation
details were fixed.  `oracle10` ranks the last-100 candidates by whether the
*candidate event itself* has an incident label, keeps the ten most-recent
incident candidates, and fills unused positions with the most-recent remaining
candidates.  It never reads the current target's label or incident group.  It
is still non-deployable because test-stream predecessor labels would not be
available online.  All methods use every training-scenario outcome when
assigning cluster policies; this deliberately generous common upper-bound
setting avoids mistaking label scarcity for a retrieval failure.  The event
vocabulary, neural model and clusters remain training-scenario-only.

For each model seed, last10 at `(eps=0.10, threshold=0.20)` fixes the unlabeled
test-stream workload target.  Each competing method selects the frozen-grid
point with minimum absolute workload difference; ties prefer lower workload
and then the earlier registered grid point.  Outcomes are not used in this
choice.  Gate-B scenario effects are the mean paired gain over the three fixed
model seeds.  Thresholds and gate inequalities are unchanged by this
clarification.

#### Feasibility amendment before oracle/selector outcomes

The engineering smoke run showed that a length-100 recurrent DeepCASE cache
is not a proportionate or budget-matched upper bound: even a hidden-size-16,
ten-step smoke checkpoint required roughly forty minutes for one complete CPU
cache and approached system memory limits.  No complete Last100 model result
existed when this amendment was made.  Length-100 DeepCASE is therefore
retained only as an incomplete feasibility/cost diagnostic and is removed from
the Gate-B decision.

This does not weaken the fixed-budget estimand.  The relevant upper bound is
whether a better subset `S` exists inside the same last-100 candidate pool
subject to `|S| = 10`, not whether a model receiving ten times as many inputs
can outperform Last10.  All subsequent DeepCASE inputs therefore have exactly
ten positions.  Gate B uses two visibly non-deployable diagnostics:

- `predecessor_oracle10`, the previously fixed candidate-label oracle;
- `class_oracle10`, which learns target-event/candidate-event/recency-band
  evidence only on training scenarios, then uses the held-out target class to
  choose the ten class-favouring candidates. Its held-out labels are used only
  to estimate this constrained ceiling and can never support a deployable
  superiority claim.

A separate additive teacher, fitted only on training scenarios, compares
Last10 features with all last-100 candidate features using held-out AUROC,
average precision, log loss, and Brier score. This tests whether the long pool
contains accessible incremental information without pretending that a
different classifier is the final DeepCASE method. If an oracle passes the
unchanged strict-recall Gate-B threshold, `association_abs10` uses the same
training-only evidence table but ranks absolute evidence strength and never
reads held-out outcomes. Random and unique-recent controls remain required in
Gate C. This amendment changes computation, not the final success criterion.

### Gate C: selector feasibility

If Gate B passes, compare budget-matched, label-free selectors on the design
scenarios:

- random10 (20 retrieval seeds);
- unique-recent10;
- rarity10;
- association10;
- fixed relevance-diversity10;
- teacher-guided10 if the fixed selector shows a positive signal.

A selector advances to confirmation only if it gains at least 0.02 strict
automatic incident recall over last10 on each design scenario, or at least
0.03 on one with no negative change on the other, at matched workload.  It
must not use incident labels at retrieval time.

### Final success criterion

The frozen selector succeeds only if, over all six confirmatory held-out
scenarios:

1. the paired mean gain in strict automatic incident recall at matched review
   workload is positive and its scenario-clustered 95% bootstrap interval
   excludes zero;
2. at least four of six scenario effects are positive;
3. the false-escalation rate per 1,000 non-incidents does not increase by more
   than 5% relative;
4. the context supplied to DeepCASE remains exactly 10 positions;
5. natural-stream results, not an injected-noise subset, support the claim.

Otherwise the confirmatory claim fails.  Noise stress tests and efficiency
results remain secondary and cannot reverse that decision.

## Primary operational estimand

DeepCASE policies have three outcomes: Incident, Non-Incident, and Reject.
The primary safety endpoint is strict automatic incident recall:

`correct automatic Incident decisions / all true incident alerts`.

Reject therefore does not count as a successful automatic detection.  Review
workload is `(automatic Incident + Reject) / all alerts`; only automatic
Non-Incident dismissals reduce review.  Competing methods are compared at the
review workload produced by last10 with the original DeepCASE defaults on the
same unlabeled test stream.  A method may choose only among a frozen grid and
may not use test outcomes to select an operating point.

Secondary endpoints are strict macro-F1 (Reject is a missed true class),
incident-group automatic recall, accepted-decision error, three Reject rates,
candidate preprocessing time, and model inference time.

## Non-negotiable validity controls

- No random row split and no context crossing scenario boundaries.
- No same-timestamp ordering signal.
- No test-fitted rarity, transition association, normalization, or selector.
- No attack/window label in a deployable selector feature.
- No dropping unknown test events or rejected samples from primary metrics.
- Last10 and selected10 receive the same number of positions and the same
  downstream architecture and training steps.
- Last100 is reported explicitly as a larger-compute upper bound.
- Oracle10 is visibly labelled non-deployable and cannot support the final
  superiority claim.
- Random retrieval uncertainty is averaged, not best-seed selected.
- Hyperparameters are frozen using design scenarios before confirmatory
  outcomes are generated.
- Statistical inference resamples scenarios/groups, not individual alerts.
- Null or adverse outcomes are retained and reported.

## Interpretation limits

AIT-ADS is a synthetic enterprise testbed with eight related attack
executions.  Passing this protocol supports a reproducible result on AIT-ADS;
it does not by itself establish effectiveness in a live SOC.  Event-level
incident prevalence is high and must be reported.  A public external dataset
with compatible alert lineage would be required for a broader claim.
