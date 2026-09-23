# Information-aware context selection for DeepCASE

This directory is an independent, fail-closed experiment.  It does not modify
the translated DeepCASE sources or the earlier `research_experiments` study.

The frozen logic is in `protocol.md`.  The workflow is deliberately gated:

1. build strictly-past last-100 candidate row indices;
2. audit whether last-10 actually displaces labelled incident predecessors;
3. test `last100` and a non-deployable `oracle10` as model upper bounds;
4. develop label-free selectors only if the upper-bound gate passes;
5. freeze the selector and evaluate six complete held-out scenarios;
6. report success or failure without selecting favourable test subsets.

Generated arrays, checkpoints, results, and reports are stored under
`artifacts/` and `results/`.

Commands use the repository's local dependency directory on `PYTHONPATH`.
The staged model runner is `run_gate_b.py`; `analyze_gate_b.py` applies the
frozen workload match only after all paired runs are complete.  Every run is
checkpointed and can be resumed without selecting favorable seeds.

## Final status

The complete fail-closed conclusion is in `report_zh_tw.md` and the
machine-readable decision is `results/overall_decision.json`.

- The frozen AIT confirmatory safe-superiority claim failed. Strict incident
  recall increased, but false escalations increased far beyond the registered
  5% limit.
- `anchor5_assoc5` is post-failure method development, not independent AIT
  confirmation.
- The frozen HDFS formula produced a numerical pass, but
  `results/hdfs_external/validity_audit.json` shows that its safety bound is
  non-binding and that the benchmark cannot establish an operational SOC
  safety claim.

## Rebuild the final audit artifacts

From the repository root in PowerShell:

```powershell
$env:PYTHONPATH=((Resolve-Path '.pydeps').Path + ';' + (Resolve-Path '.').Path)
$env:PYTHONDONTWRITEBYTECODE='1'
& '.\.venv-win\Scripts\python.exe' 'context_selection_experiment\audit_hdfs_validity.py'
& '.\.venv-win\Scripts\python.exe' 'context_selection_experiment\finalize_study.py'
& '.\.venv-win\Scripts\python.exe' 'context_selection_experiment\make_figures.py'
& '.\.venv-win\Scripts\python.exe' -m pytest 'context_selection_experiment\tests' -q
```

`finalize_study.py` asserts that the frozen AIT failure, post-failure status,
HDFS numerical pass, and HDFS operational-invalidity audit have not been
silently reinterpreted.
