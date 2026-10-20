<#
.SYNOPSIS
  Review, and optionally stop, abandoned JBG Voila kernels from one STOPPED server session.
.DESCRIPTION
  Dry-run by default. -Apply requires explicit -KernelIds and a server session that
  is no longer running. Only exact kernel IDs rejected by shutdown HTTP 403 in the
  given log, and processes using that session's unique voila_* connection folder,
  can be touched. Refuses -Apply while the configured Voila TCP port is listening.
  Never schedules automatic kills on browser reload.
.EXAMPLE
  .\scripts\cleanup_voila_kernels.ps1 -ServerLog .\logs\jbg-server_example.log
.EXAMPLE
  .\scripts\cleanup_voila_kernels.ps1 -ServerLog .\logs\jbg-server_example.log -KernelIds @('UUID') -Apply
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$ServerLog,
    [string[]]$KernelIds,
    [switch]$Apply
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$logFile = (Resolve-Path -LiteralPath $ServerLog).Path
$logText = [IO.File]::ReadAllText($logFile)
$uuid = '[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}'
$folderMatch = [regex]::Match($logText, '(?im)Storing connection files in (.+?voila_[a-zA-Z0-9_-]+)\.')
if (-not $folderMatch.Success) {
    throw 'Cannot identify the unique Voila connection directory in this server log. No action taken.'
}
$sessionDir = $folderMatch.Groups[1].Value.Trim().TrimEnd('\')
if (-not $sessionDir.StartsWith([IO.Path]::GetTempPath(), [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Connection directory is not under this machine temporary directory. No action taken.'
}
$startMatches = [regex]::Matches($logText, "(?im)Kernel started:\s*($uuid)")
$rejectedMatches = [regex]::Matches($logText, "(?im)403 POST /voila/api/shutdown/($uuid)")
$started = @{}
$rejected = @{}
foreach ($match in $startMatches) { $started[$match.Groups[1].Value.ToLowerInvariant()] = $true }
foreach ($match in $rejectedMatches) { $rejected[$match.Groups[1].Value.ToLowerInvariant()] = $true }

# No log-only inference can establish whether a kernel is idle. This tool is
# deliberately operator-driven and refuses to act until its server is stopped.
$serverId = [regex]::Match([IO.Path]::GetFileName($logFile), '_pid([0-9]+)\.log$')
if (-not $serverId.Success) { throw 'Server log has no launch PID; refusing cleanup.' }
$serverPid = [int]$serverId.Groups[1].Value
# Fail closed unless the exact server endpoint in this log is no longer accepting
# connections. The launcher PID is not sufficient: it may exit while Voila stays up.
$portMatch = [regex]::Match($logText, '(?im)Voila executable:.*?; port:\s*(\d+)\s*$')
if (-not $portMatch.Success) { throw 'Cannot identify the Voila port from this log. No action taken.' }
$voilaPort = [int]$portMatch.Groups[1].Value
if ($voilaPort -lt 1 -or $voilaPort -gt 65535) { throw 'Invalid Voila port; no action taken.' }
function Assert-VoilaPortClosed {
    param([int]$Port)
    if (-not (Get-Command Get-NetTCPConnection -ErrorAction SilentlyContinue)) {
        throw 'Get-NetTCPConnection unavailable; cannot verify stopped server. No action taken.'
    }
    try {
        $listeners = @(Get-NetTCPConnection -State Listen -ErrorAction Stop |
            Where-Object { $_.LocalPort -eq $Port })
    } catch {
        throw "Cannot check TCP listeners on port $Port. No action taken: $_"
    }
    if ($listeners.Count -gt 0) {
        throw "TCP port $Port is still listening. Stop Voila before cleanup; no action taken."
    }
}
$serverProcess = Get-CimInstance Win32_Process -Filter "ProcessId = $serverPid" -ErrorAction Stop
if ($null -ne $serverProcess) {
    throw "Server launcher PID $serverPid is still running. Close the server first; no action taken."
}
if ($Apply) { Assert-VoilaPortClosed -Port $voilaPort }
$requested = @{}
foreach ($id in @($KernelIds)) {
    if ($id -notmatch "^$uuid$") { throw "Invalid kernel ID: $id" }
    $key = $id.ToLowerInvariant()
    if (-not $rejected.ContainsKey($key) -or -not $started.ContainsKey($key)) {
        throw "Kernel $id is not a logged started+rejected kernel in this session. No action taken."
    }
    $requested[$key] = $true
}
if ($Apply -and $requested.Count -eq 0) {
    throw 'For safety, -Apply requires explicit -KernelIds from the dry-run output.'
}

$processes = @(Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" -ErrorAction Stop)
$candidates = @()
foreach ($process in $processes) {
    $cmd = [string]$process.CommandLine
    if ($cmd -notmatch '(?i)(?:^|\s)-m\s+ipykernel_launcher\b') { continue }
    if ($cmd.IndexOf($sessionDir + '\kernel-', [StringComparison]::OrdinalIgnoreCase) -lt 0) { continue }
    $match = [regex]::Match($cmd, "(?i)kernel-($uuid)\.json")
    if (-not $match.Success) { continue }
    $id = $match.Groups[1].Value.ToLowerInvariant()
    if (-not $rejected.ContainsKey($id) -or -not $started.ContainsKey($id)) { continue }
    $candidates += [pscustomobject]@{ KernelId = $id; PID = [int]$process.ProcessId; ParentPID = [int]$process.ParentProcessId }
}

Write-Host "Voila session directory: $sessionDir"
Write-Host "Shutdown-rejected kernel IDs: $($rejected.Count); matching live kernel processes: $($candidates.Count)"
if ($candidates.Count -gt 0) {
    $candidates | Sort-Object KernelId, PID | Format-Table -AutoSize
}
if (-not $Apply) {
    Write-Host 'DRY RUN only. Stop the Voila server, verify no jobs are running, then specify selected IDs with -KernelIds ... -Apply.'
    return
}

# Recheck launcher immediately before altering any process.
if ($null -ne (Get-CimInstance Win32_Process -Filter "ProcessId = $serverPid" -ErrorAction Stop)) {
    throw 'Server launcher PID is active; refusing cleanup.'
}
Assert-VoilaPortClosed -Port $voilaPort
foreach ($id in @($requested.Keys | Sort-Object)) {
    $selectedProcesses = @($candidates | Where-Object { $_.KernelId -eq $id })
    if ($selectedProcesses.Count -eq 0) { Write-Host "No matching live process for $id"; continue }
    # Guard against a server restart during cleanup, not only at invocation.
    Assert-VoilaPortClosed -Port $voilaPort
    # Process IDs may be reused. Revalidate each process command line just before stop.
    foreach ($candidate in @($selectedProcesses | Sort-Object PID -Descending)) {
        $fresh = Get-CimInstance Win32_Process -Filter "ProcessId = $($candidate.PID)"
        if ($null -eq $fresh) { continue }
        if ([string]$fresh.CommandLine -notmatch "(?i)kernel-$([regex]::Escape($id))\.json" -or
            ([string]$fresh.CommandLine).IndexOf($sessionDir + '\kernel-', [StringComparison]::OrdinalIgnoreCase) -lt 0 -or
            [string]$fresh.CommandLine -notmatch '(?i)(?:^|\s)-m\s+ipykernel_launcher\b') {
            throw "PID $($candidate.PID) no longer matches the expected kernel; stopping without further action."
        }
        Stop-Process -Id $candidate.PID -ErrorAction Stop
        Write-Host "Stopped orphan candidate: kernel=$id PID=$($candidate.PID)"
    }
}
