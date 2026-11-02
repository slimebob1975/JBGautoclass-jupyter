
# Backend-only refactoring gate for JBG AutoClassification 0.19.1+.
# Place this file under scripts/ in the repository.

param(
    [string]$PythonExecutable = 'python',
    [switch]$FullRegression,
    [string]$LogDirectory = ''
)

$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path

if (-not (Test-Path -LiteralPath (Join-Path $Root 'src'))) {
    throw 'Install script in repository/scripts/'
}

if (-not $LogDirectory) {
    $LogDirectory = Join-Path $Root 'test-results'
}

New-Item -ItemType Directory -Path $LogDirectory -Force | Out-Null

$Stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$Log = Join-Path $LogDirectory "backend_refactor_$Stamp.log"

$PreviousPythonPath = $env:PYTHONPATH
$PreviousPythonHashSeed = $env:PYTHONHASHSEED

$Failed = @()
$Passed = @()


function Invoke-TestGroup {
    param(
        [string]$Name,
        [string[]]$TestArguments
    )

    "`n=== $Name ===" | Tee-Object -FilePath $Log -Append

    # Prevent PowerShell 5.1 from treating Python's normal
    # stderr output as a terminating PowerShell error.
    $PreviousErrorActionPreference = $ErrorActionPreference

    try {
        $ErrorActionPreference = 'Continue'

        & $PythonExecutable @TestArguments 2>&1 |
            ForEach-Object { $_.ToString() } |
            Tee-Object -FilePath $Log -Append

        $Code = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $PreviousErrorActionPreference
    }

    if ($Code -ne 0) {
        $script:Failed += "$Name (exit $Code)"
        "FAIL: $Name (exit $Code)" |
            Tee-Object -FilePath $Log -Append
    }
    else {
        $script:Passed += $Name
        "PASS: $Name" |
            Tee-Object -FilePath $Log -Append
    }
}

try {
    $env:PYTHONPATH = Join-Path $Root 'src'
    $env:PYTHONHASHSEED = '0'

    Push-Location $Root

    try {
        $UnittestGroups = [ordered]@{
            'CSV registration and ingestion' = @(
                'test_csv_dataset_ingest_230.py',
                'test_csv_data_source_182.py',
                'test_csv_column_schema_185.py'
            )
            'Source isolation and data layers' = @(
                'test_classifier_source_boundary_212.py',
                'test_csv_shared_datalayer_274.py',
                'test_csv_combined_datalayer_273b.py'
            )
            'Real training and TaskRunner' = @(
                'test_csv_full_classifier_run_259.py',
                'test_csv_combined_taskrunner_273d.py',
                'test_csv_taskrunner_execution_220.py'
            )
            'Prediction and result integration' = @(
                'test_csv_prediction_integration_216.py',
                'test_csv_prediction_contract_209.py',
                'test_csv_prediction_export_267.py'
            )
            'Persistence and roundtrip' = @(
                'test_csv_fitted_model_roundtrip_255.py'
            )
            'Core TaskRunner regressions' = @(
                'test_taskrunner_source_gate_213.py'
            )
        }

        foreach ($Group in $UnittestGroups.Keys) {
            foreach ($Pattern in $UnittestGroups[$Group]) {
                $TestPath = Join-Path (Join-Path $Root 'tests') $Pattern

                if (-not (Test-Path -LiteralPath $TestPath -PathType Leaf)) {
                    $Failed += "MISSING: $Pattern"
                    "FAIL: missing $TestPath" |
                        Tee-Object -FilePath $Log -Append
                    continue
                }

                Invoke-TestGroup `
                    -Name "$Group / $Pattern" `
                    -TestArguments @(
                        '-m', 'unittest', 'discover',
                        '-s', 'tests',
                        '-p', $Pattern,
                        '-v'
                    )
            }
        }

       Invoke-TestGroup `
            -Name 'Configuration, algorithms, DataLayer, features and results' `
            -TestArguments @(
                '-m', 'pytest', '-q',
                'tests/test_config.py',
                'tests/test_meta.py',
                'tests/test_datalayer.py',
                'tests/test_feature_importance.py',
                'tests/test_feature_importance_config.py',
                'tests/test_algorithm_grid_search_params.py',
                'tests/test_model_persistence.py',
                'tests/test_taskrunner_regression_suite.py'
            )

        if ($FullRegression) {
            Invoke-TestGroup `
                -Name 'Full unittest regression' `
                -TestArguments @(
                    '-m', 'unittest', 'discover',
                    '-s', 'tests',
                    '-p', 'test_*.py',
                    '-v'
                )
        }

        "`n=== SUMMARY ===" |
            Tee-Object -FilePath $Log -Append

        "Passed groups: $($Passed.Count); Failed groups: $($Failed.Count)" |
            Tee-Object -FilePath $Log -Append

        foreach ($Failure in $Failed) {
            "FAIL: $Failure" |
                Tee-Object -FilePath $Log -Append
        }

        "Log: $Log" |
            Tee-Object -FilePath $Log -Append

        if ($Failed.Count -gt 0) {
            throw 'Backend refactor gate FAILED. See log above.'
        }

        Write-Host 'Backend refactor gate PASSED.'
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

    if ($null -eq $PreviousPythonHashSeed) {
        Remove-Item Env:PYTHONHASHSEED -ErrorAction SilentlyContinue
    }
    else {
        $env:PYTHONHASHSEED = $PreviousPythonHashSeed
    }
}
