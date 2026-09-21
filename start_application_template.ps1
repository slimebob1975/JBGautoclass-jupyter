# ===================== CONFIG (edit these) ==================================
# Base folders
$TempDir   = 'C:\temp'
$DevRoot   = 'C:\...\'

# Base Python interpreter -- an anaconda based variant is recommended!
$BasePython = 'C:\...\python.exe'

# Virtual environment
$VenvName  = 'ai_venv'
$VenvDir   = Join-Path $TempDir $VenvName
$VenvRequirements = '.\requirements.txt' 

# Executables inside the venv (once created)
$Py        = Join-Path $VenvDir 'Scripts\python.exe'
$Pip       = Join-Path $VenvDir 'Scripts\pip.exe'
$Voila     = Join-Path $VenvDir 'Scripts\voila.exe'
$Activate  = Join-Path $VenvDir 'Scripts\Activate.ps1'
$Deactivate= Join-Path $VenvDir 'Scripts\deactivate'

# Notebook + kernel
$Notebook        = 'JBG_SML_GUI.ipynb'
$KernelName      = $VenvName
$KernelDisplay   = "Python ($VenvName)"
$VoilaPort       = 8866
#
# ===================== END CONFIG (do not edit) =============================
#
# Helper: run in a directory (like a temporary cd with pushd/popd)
function Invoke-InDir {
    param(
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)][scriptblock]$ScriptBlock
    )
    Push-Location $Path
    try   { & $ScriptBlock }
    finally { Pop-Location }
}


# Write Voila/Jupyter/native-process output to a separate server log while
# preserving the same output in the terminal.
function Write-ServerLogLine {
    param(
        [Parameter(Mandatory)][string]$Line,
        [Parameter(Mandatory)][string]$LogPath
    )

    $Level = 'INFO'
    if ($Line -match '(?i)\b(critical|fatal|error|traceback)\b' -or $Line -match '^\s*[EF]\s' -or $Line -match ':\s*[EF]\s') {
        $Level = 'ERROR'
    }
    elseif ($Line -match '(?i)\bwarn(ing)?\b' -or $Line -match '^\s*W\s' -or $Line -match ':\s*W\s') {
        $Level = 'WARNING'
    }
    elseif ($Line -match '(?i)\bdebug\b') {
        $Level = 'DEBUG'
    }

    $Timestamp = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss.fffK')
    Add-Content -LiteralPath $LogPath -Value "[$Timestamp] [$Level] $Line" -Encoding UTF8
}

# Get current working directory
$CurrentDir = $PWD.Path

# --- Set up virtual environment ---
New-Item -ItemType Directory -Force -Path $TempDir | Out-Null

Set-Location $TempDir
if (-not (Test-Path $Py)) {
    Write-Host "Creating venv: $VenvDir" -ForegroundColor Yellow
    & $BasePython -m venv $VenvDir
}

# Activate the venv for this session
. $Activate

# --- Upgrade pip ---
& $Py -m pip install --upgrade pip

# --- Machine learning notebook setup ---
Invoke-InDir -Path $DevRoot -ScriptBlock {
    git.exe pull
    & $Pip install -r $VenvRequirements
}

Invoke-InDir -Path $DevRoot -ScriptBlock {
    & $Pip install --upgrade ipykernel jupyterlab voila
    & $Py  -m ipykernel install --user --name=$KernelName --display-name $KernelDisplay
}

# --- Launch Voila ---
# This log starts outside the notebook/kernel process and therefore also captures
# Voila, Jupyter/IPKernel and native-library output that the in-application Python
# logger cannot intercept (for example TensorFlow startup messages).
$LogDir = Join-Path $DevRoot 'src\JBGclassification\output\logs'
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$ServerTimestamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$ServerLog = Join-Path $LogDir ("jbg-server_{0}_pid{1}.log" -f $ServerTimestamp, $PID)
$env:JBG_SERVER_LOG = $ServerLog

# Keep native Python/Voila output and Windows PowerShell on the same encoding.
# This is especially important on Windows PowerShell 5.1, whose console may
# otherwise use a legacy OEM code page for native-process output.
$PreviousConsoleOutputEncoding = [Console]::OutputEncoding
$PreviousPythonUtf8 = $env:PYTHONUTF8
$PreviousPythonIoEncoding = $env:PYTHONIOENCODING
$Utf8Encoding = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $Utf8Encoding
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'

Write-ServerLogLine -Line "Server log started. Working directory: $DevRoot" -LogPath $ServerLog
Write-ServerLogLine -Line "Voila executable: $Voila; notebook: $Notebook; port: $VoilaPort" -LogPath $ServerLog
Write-Host "Server log: $ServerLog"

$VoilaExitCode = 0
try {
    Invoke-InDir -Path $DevRoot -ScriptBlock {
        & $Voila .\$Notebook --port $VoilaPort 2>&1 | ForEach-Object {
            $Line = $_.ToString()
            Write-ServerLogLine -Line $Line -LogPath $ServerLog
            Write-Host $Line
        }
        $script:VoilaExitCode = $LASTEXITCODE
    }
    Write-ServerLogLine -Line "Voila exited with code $VoilaExitCode" -LogPath $ServerLog
}
finally {
    [Console]::OutputEncoding = $PreviousConsoleOutputEncoding

    if ($null -eq $PreviousPythonUtf8) {
        Remove-Item Env:PYTHONUTF8 -ErrorAction SilentlyContinue
    }
    else {
        $env:PYTHONUTF8 = $PreviousPythonUtf8
    }

    if ($null -eq $PreviousPythonIoEncoding) {
        Remove-Item Env:PYTHONIOENCODING -ErrorAction SilentlyContinue
    }
    else {
        $env:PYTHONIOENCODING = $PreviousPythonIoEncoding
    }

    Remove-Item Env:JBG_SERVER_LOG -ErrorAction SilentlyContinue
}

# --- Deactivate and return to project folder ---
Set-Location $TempDir
& $Deactivate
Set-Location $CurrentDir
