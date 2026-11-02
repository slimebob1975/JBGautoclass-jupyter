# Apply one patch and run both patch-specific tests and the backend refactoring gate.
# Place this script and run_backend_refactor_tests.ps1 in the repository's scripts/ directory.
param(
    [Parameter(Mandatory=$true)][string]$Patch,
    [string[]]$NewTests = @(),
    [string]$PythonExecutable = 'python',
    [switch]$FullRegression
)
$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$PatchFile = (Resolve-Path -LiteralPath $Patch).Path
Push-Location $Root
$OldPythonPath = $env:PYTHONPATH
try {
    & git apply --check -- $PatchFile
    if ($LASTEXITCODE -ne 0) { throw "Patch fails to apply cleanly: $PatchFile" }
    & git apply -- $PatchFile
    if ($LASTEXITCODE -ne 0) { throw "Could not apply patch: $PatchFile" }
    & git diff --check
    if ($LASTEXITCODE -ne 0) { throw 'Whitespace errors after patch.' }

    if ($NewTests.Count -gt 0) {
        $env:PYTHONPATH = Join-Path $Root 'src'
        foreach ($Test in $NewTests) {
            if (-not (Test-Path -LiteralPath $Test)) { throw "New test path missing: $Test" }
        }
        & $PythonExecutable -m pytest -q @NewTests
        if ($LASTEXITCODE -ne 0) { throw 'Patch-specific tests FAILED; patch remains applied for diagnosis.' }
    }
    & (Join-Path $PSScriptRoot 'run_backend_refactor_tests.ps1') -PythonExecutable $PythonExecutable -FullRegression:$FullRegression
    if (-not $?) { throw 'Backend refactoring gate FAILED; patch remains applied for diagnosis.' }
    Write-Host 'PATCH APPLIED; new tests and backend regression gate PASSED. Review diff before commit.'
}
finally {
    if ($null -eq $OldPythonPath) { Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue }
    else { $env:PYTHONPATH = $OldPythonPath }
    Pop-Location
}
