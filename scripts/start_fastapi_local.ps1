# Revision 285d: local FastAPI server with durable runs, in the main JBG environment.
# Revision 282b3 created a dedicated venv; runs now need the full JBG engine, so the
# server and its worker processes use the same Python as Voila and the test gate.
# Example: & .\scripts\start_fastapi_local.ps1 -Port 8000
[CmdletBinding()]
param(
    [string]$PythonExecutable = 'python',
    [ValidateRange(1, 65535)][int]$Port = 8000,
    [switch]$InstallDependencies
)

$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Src = Join-Path $Root 'src'
$Requirements = Join-Path $Root 'requirements-fastapi.txt'
$LogDir = Join-Path $Root 'logs'
$Stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$Log = Join-Path $LogDir "jbg-fastapi_$Stamp.log"
$LoggingConfig = Join-Path $LogDir "jbg-fastapi_logging_$Stamp.json"

if (-not (Test-Path -LiteralPath $Requirements)) {
    throw "Missing requirements file: $Requirements"
}
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

# Install the API packages only when explicitly requested or when one is missing.
& $PythonExecutable -c 'import fastapi, uvicorn, fastapi_offline, sqlalchemy, python_multipart, jinja2, babel' 2>$null
if ($InstallDependencies -or $LASTEXITCODE -ne 0) {
    Write-Host 'Installing FastAPI dependencies into the JBG environment...' -ForegroundColor Cyan
    & $PythonExecutable -m pip install -r $Requirements
    if ($LASTEXITCODE -ne 0) { throw 'FastAPI dependency installation failed' }
}

# The modules live in src; set PYTHONPATH before import validation.
$PreviousPythonPath = $env:PYTHONPATH
$PreviousUtf8 = $env:PYTHONUTF8
$PreviousIoEncoding = $env:PYTHONIOENCODING
$PreviousHashSeed = $env:PYTHONHASHSEED
try {
    $env:PYTHONPATH = if ($PreviousPythonPath) { "$Src$([IO.Path]::PathSeparator)$PreviousPythonPath" } else { $Src }
    $env:PYTHONUTF8 = '1'
    $env:PYTHONIOENCODING = 'utf-8'
    $env:PYTHONHASHSEED = '0'
    & $PythonExecutable -c 'import fastapi, uvicorn, sqlalchemy, JBGFastAPIServer'
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

    Write-Host "Python: $PythonExecutable" -ForegroundColor DarkGray
    Write-Host "JBG web interface: http://127.0.0.1:$Port/" -ForegroundColor Green
    Write-Host "API documentation: http://127.0.0.1:$Port/docs"
    Write-Host 'Runs execute in separate worker processes (logs\runs).'
    Write-Host 'Stopping this server does not stop running jobs; cancel them through the API.'
    Write-Host "Log: $Log"
    Write-Host 'Stop with Ctrl+C.'

    Push-Location $Root
    try {
        & $PythonExecutable -m uvicorn 'JBGFastAPIServer:create_server_app' --factory --app-dir $Src --host '127.0.0.1' --port $Port --log-config $LoggingConfig
        if ($LASTEXITCODE -ne 0) { throw "Uvicorn exited with code $LASTEXITCODE" }
    }
    finally { Pop-Location }
}
finally {
    $env:PYTHONPATH = $PreviousPythonPath
    $env:PYTHONUTF8 = $PreviousUtf8
    $env:PYTHONIOENCODING = $PreviousIoEncoding
    $env:PYTHONHASHSEED = $PreviousHashSeed
    Remove-Item -LiteralPath $LoggingConfig -ErrorAction SilentlyContinue
}
