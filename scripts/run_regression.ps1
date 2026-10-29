# Revision 272: explicit fast startup checks and full pre-commit regression gate.
# Run from any working directory:
#   & .\scripts\run_regression.ps1
#   & .\scripts\run_regression.ps1 -Full
param(
    [switch]$Full,
    [string]$PythonExecutable = 'python'
)

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$PreviousPythonPath = $env:PYTHONPATH
try {
    $env:PYTHONPATH = Join-Path $Root 'src'
    Push-Location $Root
    try {
        if ($Full) {
            Write-Host 'Running FULL unittest regression suite...'
            & $PythonExecutable -m unittest discover -s tests -p 'test_*.py' -v
            if ($LASTEXITCODE -ne 0) {
                throw 'Full unittest regression tests failed.'
            }
        }
        else {
            # Named, representative smoke suites. Do not rely on implicit discovery.
            # The full suite must be run separately before commit/release.
            $FastPatterns = @(
                'test_classifier_source_boundary_212.py',
                'test_csv_column_schema_185.py',
                'test_csv_dataset_binding_188.py',
                'test_csv_gui_training_readiness_260.py',
                'test_csv_full_classifier_run_259.py',
                'test_csv_saved_prediction_gui_265.py',
                'test_csv_prediction_export_267.py',
                'test_csv_dataset_cascade_269c.py',
                'test_local_startup_gate_270.py'
            )
            Write-Host 'Running FAST startup regression gate...'
            foreach ($Pattern in $FastPatterns) {
                if (-not (Test-Path -LiteralPath (Join-Path 'tests' $Pattern) -PathType Leaf)) {
                    throw "Required fast regression module is missing: $Pattern"
                }
                & $PythonExecutable -m unittest discover -s tests -p $Pattern -q
                if ($LASTEXITCODE -ne 0) {
                    throw "Fast regression tests failed: $Pattern"
                }
            }
        }

        # These pytest-style tests are not executed by unittest discovery.
        Write-Host 'Running pytest configuration and metadata regression tests...'
        & $PythonExecutable -m pytest tests/test_config.py tests/test_meta.py -q
        if ($LASTEXITCODE -ne 0) {
            throw 'Pytest regression tests failed.'
        }
        Write-Host 'Regression gate PASSED.'
    }
    finally {
        Pop-Location
    }
}
finally {
    if ($null -eq $PreviousPythonPath) {
        Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
    }
    else {
        $env:PYTHONPATH = $PreviousPythonPath
    }
}
