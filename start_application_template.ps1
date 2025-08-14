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

# --- Launch Voilà ---
Invoke-InDir -Path $DevRoot -ScriptBlock {
    & $Voila .\$Notebook --port $VoilaPort
}

# --- Deactivate and return to project folder ---
Set-Location $TempDir
& $Deactivate
Set-Location $CurrentDir
