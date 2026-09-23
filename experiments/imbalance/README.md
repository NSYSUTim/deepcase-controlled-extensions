# Incident-aligned, imbalance-aware DeepCASE experiment

This directory contains the reproducible experiment for the following scoped
question:

> With detector rules and the chronological alert stream left intact, and with
> the test prevalence left at its observed AIT-ADS value, does training the
> exact representation consumed by the DeepCASE Interpreter with
> incident-aligned, imbalance-aware supervision improve incident detection at
> a fixed analyst-review workload under severe *training-label scarcity*?

The qualification "training-label scarcity" is essential. AIT-ADS is a public
alert stream with 2,655,821 alerts, but its event-level incident prevalence is
not naturally rare. The experiment therefore never claims that AIT-ADS itself
is a naturally extreme incident/non-incident data set. Severe ratios are
created by hiding training labels, not by deleting alerts. Validation and test
streams are never resampled.

## Non-negotiable controls

- No detector rule is removed or disabled.
- Alert multiplicity is preserved. Same-second duplicates are not discarded.
- Contexts contain only strictly earlier timestamps; alerts tied at the same
  second cannot observe one another in an arbitrary file order.
- Splits are by complete AIT-ADS scenario, never by alert row.
- Event vocabulary and all learned statistics are fitted on training scenarios.
- The same revealed labels, seeds, optimization budget, Interpreter and reject
  semantics are used in paired method comparisons.
- Test labels are used only once for final scoring.
- Statistical inference treats scenario, not alert or random seed, as the
  independent unit.

## Methods

`deepcase` is the original next-event ContextBuilder plus the common
index-aligned Interpreter adapter described below. `aux_bce`,
`aux_weighted_bce`, and `aux_focal` add a training-only
binary head to the exact attention-weighted event vector later clustered by
the Interpreter. `proposed` combines class-balanced focal loss with
incident-group-balanced supervised batches. The auxiliary head is not a third
output class and is not used at deployment; the unmodified Interpreter still
returns Incident, Non-Incident, or Reject.

The full objective is

```
L = L_next_event + lambda_incident * L_incident
```

Only `L_incident` changes across auxiliary-loss ablations. Focal loss is not
applied to next-event prediction; that distinction prevents conflating this
study with prior inconclusive focal-KL experiments on DeepCASE's event task.

## Workflow

1. `prepare_data.py` converts the official author-produced AIT-ADS alert CSVs
   into a canonical table and strict-past contexts.
2. `audit_data.py` writes the data-quality and label audit.
3. `run_experiment.py --stage pilot` runs smoke/pilot folds.
4. `run_experiment.py --stage full` runs all pre-registered folds and seeds.
5. `analyze_results.py` computes paired, scenario-clustered estimates and
   generates publication-ready Markdown/CSV tables.
6. `validate_results.py` fails closed on missing records, identity/hash
   mismatches, incomplete operating grids, invalid metrics, or broken
   Incident/Reject workload decompositions.

The phrase "same analyst workload" is operationalized as an alert review-rate
budget: both Incident and Reject require review and only Non-Incident is
automatically dismissed.  This is a reproducible workload *proxy*, not a claim
that every alert consumes identical analyst minutes.  An operating point is
either selected wholly on validation data or calibrated from the unlabeled
test-stream review rate; test outcomes never select a point.  If the discrete
DeepCASE policy cannot reach a budget, the result is infeasible rather than
interpolated.

The ambiguous-rule result is a secondary oracle diagnostic.  Its rule list is
defined using complete training-scenario truth, but it is never supplied to the
model, sampler, or operating-point selector.  The primary endpoint always uses
the complete, natural held-out alert stream.

`russellmitchell` and `fox` are explicitly reserved as exploratory
design/feasibility scenarios. They exposed that a universal low review-rate
budget can be unattainable when the natural incident prevalence is high and
Reject also consumes review. They are therefore excluded from confirmatory
hypothesis inference; the other six scenario folds are the independent
confirmatory units. The reported ablation and query-iteration sensitivities
use `russellmitchell`; `fox` is retained as a design reserve, not silently
promoted to confirmatory evidence. This is a protocol freeze, not a claim of
external preregistration.

The primary policy now gives a precise meaning to "same manpower."  For each
held-out scenario, the budget is the review rate actually produced by original
DeepCASE defaults (`eps=0.1`, `threshold=0.2`).  A competing method may observe
only its number of review decisions and selects the closest tested operating
point that does not exceed that budget.  Test incident outcomes cannot affect
selection.  Fixed absolute budgets remain secondary sensitivity analyses.

The design-scenario loss traces were used only to freeze compute: three epochs
of 500 natural event batches each. The exploratory pilot evaluates the full
4-by-4 Interpreter grid; the confirmatory run uses seven pre-specified points
covering the observed low/middle/high review-rate range and including original
DeepCASE defaults. No confirmatory outcome was inspected before this freeze.

## Interpreter indexing correction

The upstream `Interpreter.score` implementation filters clustered events and
scores but does not apply the same DBSCAN-noise mask to attended vectors.  It
also applies the KD-tree's internal data permutation to indices already
returned in input-data coordinates.  Targeted invariance tests reproduce both
errors (including 50 exchanged labels among 101 exact known vectors).

The experiment's `aligned_v2` adapter constructs event, vector and score arrays
with one identical mask, and attaches scores directly to KD-tree input-row
indices.  It does not change attention, DBSCAN, max cluster scoring, or any of
the three Reject conditions.  The repair is applied identically to DeepCASE
and every extension.  Earlier unnamespaced JSON files are retained for audit
but are invalidated for all claims; paper results are read only from
`results/aligned_v2`.

Generated data, checkpoints, and result tables are intentionally ignored by
Git. Every run writes a configuration snapshot, configuration/data hashes,
environment versions, seed, and fold membership. `validate_results.py` adds
the complete research/core-source hashes to the final validation artifact.

## Sensitivity checkpoint reuse

`--checkpoint-source` re-evaluates a frozen neural checkpoint without
retraining. `--checkpoint-source-ratio` is valid for `deepcase` across label
ratios because its event-only training objective never observes incident
labels; it must not be used to substitute a `proposed` checkpoint trained at a
different ratio. `--query-iterations` changes only Interpreter attention-query
optimization and is recorded explicitly in each result. These controls let the
label-ratio and query-iteration sensitivities isolate the intended variable.

## Completed result packages

All expected records below pass `validate_results.py` with zero errors:

- primary event-label experiment: 54/54 records;
- 1:10 and 1:100 ratio sensitivities: 36/36 records;
- window-label sensitivity: 18/18 records;
- q=0 ablation: 7/7 records, each with the complete 16-point pilot grid;
- q=10 and q=100 sensitivities: 4/4 records, each with the complete pilot grid.
- frozen q=100 rescue design selection: 6/6 full-compute records across the
  two design scenarios;
- q=100 rescue confirmation: 36/36 records across six scenarios, three seeds,
  and the paired DeepCASE/proposed methods.  One proposed operating point is
  infeasible at matched workload, so the planned 18-pair confirmatory estimand
  is incomplete and no confirmatory CI/p-value is reported.

## RC-CR replacement feasibility

After the focal auxiliary method failed its q=100 robustness check, a separate
design-only RC-CR experiment tested a frozen q=100 representation, unchanged
Reject policy, smoothed rule prior, and deployment-time within-rule pairwise
ranker.  The training-data identifiability gate passed only narrowly: two
genuinely mixed rules supplied five revealed incident groups and eight
revealed positives in the first design fold.  RC-CR then failed the +0.01
launch criterion on `russellmitchell` and was slightly worse than DeepCASE at
both reported endpoints.  Per the stop rule, the larger `fox` run was not
started.  See `rccr_feasibility_protocol.md` and
`rccr_feasibility_report.md`.  These AIT-ADS results are exploratory and do not
create a new confirmatory superiority claim.

The main human-readable report is `paper_report_zh_tw.md`. Exact selected
operating points and paired estimates are in `artifacts/analysis_*_selected.csv`
and `artifacts/analysis_*_paired.csv`; independent validation receipts are in
`artifacts/validation_*`.

## Minimal reproduction commands

Download the public AIT-ADS author CSV archive from
<https://doi.org/10.5281/zenodo.8263181> and place its extracted CSV tree at
`dataset/reference/alerts_csv/alerts_csv`, as specified in `config.yaml`.
From the repository root, with the project dependencies on `PYTHONPATH`:

```powershell
py -m venv .venv-win
.\.venv-win\Scripts\python.exe -m pip install `
  -r research_experiments\requirements-lock.txt
.\.venv-win\Scripts\Activate.ps1

python research_experiments/prepare_data.py
python research_experiments/audit_data.py

foreach ($method in 'deepcase','proposed','rule_prior') {
  .\research_experiments\run.ps1 -Stage full -Method $method `
    -Ratio 1000 -ConfirmatoryOnly -ResultNamespace aligned_v2
}

python research_experiments/analyze_results.py `
  --result-namespace aligned_v2 --stage full --label event_incident
python research_experiments/validate_results.py `
  --namespace aligned_v2 --stage full --label event_incident --ratio 1000 `
  --seeds 1729 2718 31415 --methods deepcase proposed rule_prior
```

These commands reproduce the primary package. The exact namespaces for every
sensitivity appear in the corresponding JSON validation receipts and in the
paper appendix. Never combine the old unnamespaced results with `aligned_v2`.
