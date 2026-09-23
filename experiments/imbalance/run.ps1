param(
    [ValidateSet('smoke', 'pilot', 'full')]
    [string]$Stage = 'smoke',
    [string]$Method = '',
    [string]$TestScenario = '',
    [ValidateSet('', 'event_incident', 'window_incident')]
    [string]$Label = '',
    [int]$Ratio = 0,
    [int]$Seed = 0,
    [switch]$EvaluateOnly,
    [switch]$ConfirmatoryOnly,
    [string]$ResultNamespace = 'aligned_v2',
    [string]$CheckpointSource = '',
    [int]$CheckpointSourceRatio = 0,
    [int]$QueryIterations = -1,
    [string]$PythonExe = '',
    [switch]$Force
)

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSScriptRoot
if (-not $PythonExe) {
    $Candidates = @(
        (Join-Path $RepoRoot '.venv-win\Scripts\python.exe'),
        (Join-Path $RepoRoot '.venv\Scripts\python.exe')
    )
    $LocalPython = $Candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    if ($LocalPython) {
        $PythonExe = (Resolve-Path -LiteralPath $LocalPython).Path
    } else {
        $PythonExe = 'python'
    }
}
$PythonPaths = @($RepoRoot, $PSScriptRoot)
$LocalDeps = Join-Path $RepoRoot '.pydeps'
if (Test-Path -LiteralPath $LocalDeps) {
    $PythonPaths = @((Resolve-Path -LiteralPath $LocalDeps).Path) + $PythonPaths
}
$env:PYTHONPATH = $PythonPaths -join ';'
$arguments = @((Join-Path $PSScriptRoot 'run_experiment.py'), '--stage', $Stage)
$arguments += @('--result-namespace', $ResultNamespace)
if ($Method) { $arguments += @('--method', $Method) }
if ($TestScenario) { $arguments += @('--test-scenario', $TestScenario) }
if ($Label) { $arguments += @('--label', $Label) }
if ($Ratio -gt 0) { $arguments += @('--ratio', "$Ratio") }
if ($Seed -gt 0) { $arguments += @('--seed', "$Seed") }
if ($EvaluateOnly) { $arguments += '--evaluate-only' }
if ($ConfirmatoryOnly) { $arguments += '--confirmatory-only' }
if ($CheckpointSource) { $arguments += @('--checkpoint-source', $CheckpointSource) }
if ($CheckpointSourceRatio -gt 0) { $arguments += @('--checkpoint-source-ratio', "$CheckpointSourceRatio") }
if ($QueryIterations -ge 0) { $arguments += @('--query-iterations', "$QueryIterations") }
if ($Force) { $arguments += '--force' }
& $PythonExe @arguments
