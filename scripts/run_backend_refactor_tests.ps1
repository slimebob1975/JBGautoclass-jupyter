
# Backend-only refactoring gate for JBG AutoClassification 0.19.1+.
# Place this file under scripts/ in the repository.
#
# Revision 284b: adds the application-layer group (all tests/test_revision_NNN*.py
# with NNN >= 280) and prints a per-group summary table at the end, which is also
# written to a separate *_summary.txt file next to the full log.

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
$SummaryFile = Join-Path $LogDirectory "backend_refactor_$($Stamp)_summary.txt"

$PreviousPythonPath = $env:PYTHONPATH
$PreviousPythonHashSeed = $env:PYTHONHASHSEED
$PreviousPythonIoEncoding = $env:PYTHONIOENCODING
$PreviousOutputEncoding = $null

$Failed = @()
$Passed = @()
$Results = New-Object System.Collections.Generic.List[object]
$GateClock = [System.Diagnostics.Stopwatch]::StartNew()

# Maximum failing-test lines (or trailing output lines) shown per failed group.
$FailureTailLines = 15


function Get-TestCounts {
    # Parse the final unittest or pytest report. Unknown values stay $null,
    # so the summary never shows invented numbers.
    param([string[]]$Lines)

    $Counts = [ordered]@{ Tests = $null; Passed = $null; Failed = $null; Skipped = $null }
    $Text = $Lines -join "`n"

    $Ran = [regex]::Matches($Text, '(?m)^Ran (\d+) tests? in ')
    if ($Ran.Count -gt 0) {
        # unittest: 'Ran N tests', then 'OK', 'OK (skipped=2)' or
        # 'FAILED (failures=1, errors=2, skipped=1)'.
        $Total = [int]$Ran[$Ran.Count - 1].Groups[1].Value
        $Failures = 0; $Errors = 0; $Skips = 0
        $Status = [regex]::Matches($Text, '(?m)^(OK|FAILED)(?: \((.*)\))?\s*$')
        if ($Status.Count -gt 0) {
            $Details = $Status[$Status.Count - 1].Groups[2].Value
            foreach ($Part in ($Details -split ',\s*')) {
                if ($Part -match '^(failures|errors|skipped|expected failures|unexpected successes)=(\d+)$') {
                    switch ($Matches[1]) {
                        'failures' { $Failures = [int]$Matches[2] }
                        'errors'   { $Errors = [int]$Matches[2] }
                        'skipped'  { $Skips = [int]$Matches[2] }
                    }
                }
            }
        }
        $Counts.Tests = $Total
        $Counts.Failed = $Failures + $Errors
        $Counts.Skipped = $Skips
        $Counts.Passed = $Total - $Failures - $Errors - $Skips
        return [pscustomobject]$Counts
    }

    # pytest: last line such as '1 failed, 123 passed, 1 skipped, 29 errors in 7.04s'.
    $Final = [regex]::Matches($Text, '(?m)^=*\s*((?:\d+ \w+(?:, )?)+) in [\d.]+s')
    if ($Final.Count -gt 0) {
        $Summary = $Final[$Final.Count - 1].Groups[1].Value
        $Found = @{}
        foreach ($Item in [regex]::Matches($Summary, '(\d+) (\w+)')) {
            $Found[$Item.Groups[2].Value] = [int]$Item.Groups[1].Value
        }
        $Ok = 0; $Bad = 0; $Skips = 0
        foreach ($Key in $Found.Keys) {
            switch -Regex ($Key) {
                '^(passed|xpassed)$'  { $Ok += $Found[$Key] }
                '^(failed|errors?)$'  { $Bad += $Found[$Key] }
                '^(skipped|xfailed)$' { $Skips += $Found[$Key] }
            }
        }
        $Counts.Tests = $Ok + $Bad + $Skips
        $Counts.Passed = $Ok
        $Counts.Failed = $Bad
        $Counts.Skipped = $Skips
    }
    return [pscustomobject]$Counts
}


function Get-FailureExcerpt {
    # Prefer the failing test names (pytest 'FAILED'/'ERROR', unittest
    # 'FAIL:'/'ERROR:'); otherwise fall back to the last output lines.
    param([string[]]$Lines)

    if (-not $Lines -or $Lines.Count -eq 0) { return @() }
    $Named = @($Lines | Where-Object { $_ -match '^(FAILED|ERROR) |^(FAIL|ERROR): ' })
    if ($Named.Count -eq 0) {
        $Start = [math]::Max(0, $Lines.Count - $FailureTailLines)
        return $Lines[$Start..($Lines.Count - 1)]
    }
    $Excerpt = New-Object System.Collections.Generic.List[string]
    foreach ($Line in ($Named | Select-Object -First $FailureTailLines)) { $Excerpt.Add($Line) }
    if ($Named.Count -gt $FailureTailLines) {
        $Excerpt.Add("... and $($Named.Count - $FailureTailLines) more; see the full log.")
    }
    $Last = $Lines[$Lines.Count - 1]
    if ($Excerpt[$Excerpt.Count - 1] -ne $Last) { $Excerpt.Add($Last) }
    return $Excerpt.ToArray()
}


function Add-Result {
    param(
        [string]$Name,
        [string]$Status,
        [object]$Counts,
        [double]$Seconds,
        [string[]]$Tail
    )

    $script:Results.Add([pscustomobject]@{
        Group    = $Name
        Status   = $Status
        Tests    = $Counts.Tests
        Passed   = $Counts.Passed
        Failed   = $Counts.Failed
        Skipped  = $Counts.Skipped
        Seconds  = [math]::Round($Seconds, 1)
        Tail     = $Tail
    })
}


function Invoke-TestGroup {
    param(
        [string]$Name,
        [string[]]$TestArguments
    )

    "`n=== $Name ===" | Tee-Object -FilePath $Log -Append

    $Lines = New-Object System.Collections.Generic.List[string]
    $Clock = [System.Diagnostics.Stopwatch]::StartNew()

    # Prevent PowerShell 5.1 from treating Python's normal
    # stderr output as a terminating PowerShell error.
    $PreviousErrorActionPreference = $ErrorActionPreference

    try {
        $ErrorActionPreference = 'Continue'

        & $PythonExecutable @TestArguments 2>&1 |
            ForEach-Object { $Line = $_.ToString(); $Lines.Add($Line); $Line } |
            Tee-Object -FilePath $Log -Append

        $Code = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $PreviousErrorActionPreference
    }

    $Clock.Stop()
    $Counts = Get-TestCounts -Lines $Lines.ToArray()

    if ($Code -ne 0) {
        $script:Failed += "$Name (exit $Code)"
        "FAIL: $Name (exit $Code)" |
            Tee-Object -FilePath $Log -Append
        $Tail = Get-FailureExcerpt -Lines $Lines.ToArray()
        Add-Result -Name $Name -Status 'FAIL' -Counts $Counts -Seconds $Clock.Elapsed.TotalSeconds -Tail $Tail
    }
    else {
        $script:Passed += $Name
        "PASS: $Name" |
            Tee-Object -FilePath $Log -Append
        Add-Result -Name $Name -Status 'PASS' -Counts $Counts -Seconds $Clock.Elapsed.TotalSeconds -Tail @()
    }
}


function Write-GateSummary {
    # Compact end-of-run overview: console, full log and a separate file.
    $Width = 200
    $Report = New-Object System.Collections.Generic.List[string]
    $Report.Add("=== GROUP RESULTS ===")
    $Report.Add("Gate started $Stamp; total duration $([math]::Round($GateClock.Elapsed.TotalMinutes, 1)) min")
    $Table = $Results |
        Select-Object Status, Tests, Passed, Failed, Skipped, Seconds, Group |
        Format-Table -AutoSize |
        Out-String -Width $Width
    foreach ($Row in ($Table -split "`r?`n")) {
        if ($Row.Trim()) { $Report.Add($Row.TrimEnd()) }
    }

    $Known = @($Results | Where-Object { $null -ne $_.Tests })
    if ($Known.Count -gt 0) {
        $Sum = @{ Tests = 0; Passed = 0; Failed = 0; Skipped = 0 }
        foreach ($Result in $Known) {
            foreach ($Key in @('Tests', 'Passed', 'Failed', 'Skipped')) { $Sum[$Key] += $Result.$Key }
        }
        $Report.Add("Tests in parsed groups: $($Sum.Tests) run, $($Sum.Passed) passed, " +
                    "$($Sum.Failed) failed or errors, $($Sum.Skipped) skipped")
    }
    $Unparsed = @($Results | Where-Object { $null -eq $_.Tests -and $_.Status -ne 'MISSING' })
    foreach ($Result in $Unparsed) {
        $Report.Add("Note: no test report found for '$($Result.Group)'; see the full log.")
    }

    foreach ($Result in @($Results | Where-Object { $_.Status -ne 'PASS' })) {
        $Report.Add("")
        $Report.Add("--- $($Result.Status): $($Result.Group) (last lines) ---")
        foreach ($Line in $Result.Tail) { $Report.Add($Line) }
    }

    $Report | Tee-Object -FilePath $Log -Append
    $Report | Set-Content -LiteralPath $SummaryFile -Encoding UTF8
}


try {
    $env:PYTHONPATH = Join-Path $Root 'src'
    $env:PYTHONHASHSEED = '0'

    # 285c: Python writes UTF-8 and PowerShell reads UTF-8, so Swedish
    # characters in messages (e.g. Windows errors) are not garbled in the log.
    $env:PYTHONIOENCODING = 'utf-8'
    try {
        $PreviousOutputEncoding = [Console]::OutputEncoding
        [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
    }
    catch {
        $PreviousOutputEncoding = $null
    }

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
                    Add-Result -Name "$Group / $Pattern" -Status 'MISSING' `
                        -Counts (Get-TestCounts -Lines @()) -Seconds 0 -Tail @("Missing file: $TestPath")
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

        # Application layer: every revision test from 280 onwards, found by
        # name so that new revision tests are included automatically.
        $ApplicationTests = @(
            Get-ChildItem -LiteralPath (Join-Path $Root 'tests') -Filter 'test_revision_*.py' -File |
                Where-Object { $_.Name -match '^test_revision_(\d+)' -and [int]$Matches[1] -ge 280 } |
                Sort-Object Name |
                ForEach-Object { 'tests/' + $_.Name }
        )

        if ($ApplicationTests.Count -eq 0) {
            $Failed += 'MISSING: application-layer revision tests'
            'FAIL: no tests/test_revision_*.py files from revision 280 onwards' |
                Tee-Object -FilePath $Log -Append
            Add-Result -Name 'Application layer (revisions 280+)' -Status 'MISSING' `
                -Counts (Get-TestCounts -Lines @()) -Seconds 0 -Tail @('No application-layer test files found.')
        }
        else {
            Invoke-TestGroup `
                -Name "Application layer (revisions 280+, $($ApplicationTests.Count) files)" `
                -TestArguments (@('-m', 'pytest', '-q') + $ApplicationTests)
        }

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

        "`n" | Tee-Object -FilePath $Log -Append
        Write-GateSummary

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

        "Summary: $SummaryFile" |
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

    if ($null -eq $PreviousPythonIoEncoding) {
        Remove-Item Env:PYTHONIOENCODING -ErrorAction SilentlyContinue
    }
    else {
        $env:PYTHONIOENCODING = $PreviousPythonIoEncoding
    }

    if ($null -ne $PreviousOutputEncoding) {
        try { [Console]::OutputEncoding = $PreviousOutputEncoding } catch { }
    }
}
