# Revision 282b3: local FastAPI launcher, modeled on start_local_service.ps1.
# Creates a dedicated venv on first run. Subsequent starts work offline.
# Example: & .\scripts\start_fastapi_local.ps1 -Port 8000
[CmdletBinding()]
param(
    [string]$BasePython = 'C:\Users\Grro\AppData\Local\Programs\Python\Python313\python.exe',
    [string]$TempDir = 'C:\temp',
    [string]$VenvName = 'jbg_fastapi_venv',
    [ValidateRange(1, 65535)][int]$Port = 8000,
    [switch]$InstallDependencies
)

$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Src = Join-Path $Root 'src'
$Requirements = Join-Path $Root 'requirements-fastapi.txt'
$LogDir = Join-Path $Root 'logs'
$VenvDir = Join-Path $TempDir $VenvName
$Py = Join-Path $VenvDir 'Scripts\python.exe'
$Uvicorn = Join-Path $VenvDir 'Scripts\uvicorn.exe'
$InstalledMarker = Join-Path $VenvDir '.jbg_fastapi_requirements_282b4_installed'
$Stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$Log = Join-Path $LogDir "jbg-fastapi_$Stamp.log"
$LoggingConfig = Join-Path $LogDir "jbg-fastapi_logging_$Stamp.json"

if (-not (Test-Path -LiteralPath $Requirements)) {
    throw "Missing requirements file: $Requirements"
}
New-Item -ItemType Directory -Force -Path $TempDir, $LogDir | Out-Null

if (-not (Test-Path -LiteralPath $Py)) {
    Write-Host "Creating JBG FastAPI venv: $VenvDir" -ForegroundColor Yellow
    & $BasePython -m venv $VenvDir
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $Py)) {
        throw 'Virtual environment creation failed'
    }
}

# Install on first use or when explicitly requested. No network needed later.
if ($InstallDependencies -or -not (Test-Path -LiteralPath $InstalledMarker)) {
    Write-Host 'Installing FastAPI dependencies in dedicated venv...' -ForegroundColor Cyan
    & $Py -m pip install -r $Requirements
    if ($LASTEXITCODE -ne 0) { throw 'FastAPI dependency installation failed' }
    # The marker is written only after a successful installation.
    Set-Content -LiteralPath $InstalledMarker -Value 'installed' -Encoding ASCII
}

# The module lives in src; set PYTHONPATH before import validation.
$PreviousPythonPath = $env:PYTHONPATH
$PreviousUtf8 = $env:PYTHONUTF8
$PreviousIoEncoding = $env:PYTHONIOENCODING
try {
    $env:PYTHONPATH = if ($PreviousPythonPath) { "$Src$([IO.Path]::PathSeparator)$PreviousPythonPath" } else { $Src }
    $env:PYTHONUTF8 = '1'
    $env:PYTHONIOENCODING = 'utf-8'
    & $Py -c 'import fastapi, uvicorn, JBGFastAPILocal'
    if ($LASTEXITCODE -ne 0) { throw 'FastAPI app imports failed' }

    # Uvicorn normally logs INFO on stderr. Configure console + file handlers
    # to avoid Windows PowerShell treating normal INFO output as an exception.
    $Config = @{
        version = 1
        disable_existing_loggers = $false
        formatters = @{ default = @{ format = '%(levelname)s: %(message)s' } }
        handlers = @{
            console = @{ class = 'logging.StreamHandler'; formatter = 'default'; stream = 'ext://sys.stdout' }
            logfile = @{ class = 'logging.FileHandler'; formatter = 'default'; filename = $Log; encoding = 'utf-8' }
        }
        loggers = @{
            'uvicorn' = @{ handlers = @('console', 'logfile'); level = 'INFO'; propagate = $false }
            'uvicorn.error' = @{ level = 'INFO' }
            'uvicorn.access' = @{ level = 'INFO' }
        }
    }
    [IO.File]::WriteAllText($LoggingConfig, ($Config | ConvertTo-Json -Depth 8), (New-Object System.Text.UTF8Encoding($false)))

    Write-Host "Python: $Py" -ForegroundColor DarkGray
    Write-Host "JBG local API: http://127.0.0.1:$Port/docs" -ForegroundColor Green
    Write-Host 'Read-only API; no training jobs are available until write APIs are implemented.'
    Write-Host "Log: $Log"
    Write-Host 'Stop with Ctrl+C.'

    Push-Location $Root
    try {
        & $Uvicorn 'JBGFastAPILocal:app' --app-dir $Src --host '127.0.0.1' --port $Port --log-config $LoggingConfig
        if ($LASTEXITCODE -ne 0) { throw "Uvicorn exited with code $LASTEXITCODE" }
    }
    finally { Pop-Location }
}
finally {
    $env:PYTHONPATH = $PreviousPythonPath
    $env:PYTHONUTF8 = $PreviousUtf8
    $env:PYTHONIOENCODING = $PreviousIoEncoding
    Remove-Item -LiteralPath $LoggingConfig -ErrorAction SilentlyContinue
}
