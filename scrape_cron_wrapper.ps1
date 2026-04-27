# scrape_cron_wrapper.ps1
#
# Hourly cron job for top-N leaderboard scrape on Windows.
# Invoked by the scheduled task that install_scrape_task.ps1 creates.
#
# Responsibilities:
#   1. Check the self-expiry timestamp in .scrape_cron_expiry — if past
#      now, uninstall the scheduled task and stop.
#   2. Acquire .scrape_cron.lock (file-based mutex) to prevent overlapping
#      runs. If another wrapper is already running, exit cleanly.
#   3. Run scrape_top5.py + parse_replays.py with PYTHONIOENCODING=utf-8
#      so the Chinese Windows GBK codec doesn't blow up on emoji.
#   4. Append everything to .scrape_cron.log (rotated by size, see
#      _Rotate-Log).

$ErrorActionPreference = "Continue"
# Ensure stdout from Python (with arrows + emoji) is captured as UTF-8.
# Without this, the default console codepage (GBK on Chinese Windows)
# mangles any non-ASCII char written to the log.
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python   = "E:\ANACONDA_NEW\python.exe"
$LogFile  = Join-Path $RepoRoot ".scrape_cron.log"
$LockFile = Join-Path $RepoRoot ".scrape_cron.lock"
$ExpiryFile = Join-Path $RepoRoot ".scrape_cron_expiry"
$TaskName = "OrbitWarsScrape"

function Write-Log {
    param([string]$Msg)
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "$ts  $Msg" | Out-File -FilePath $LogFile -Append -Encoding utf8
}

function Rotate-Log {
    if (Test-Path $LogFile) {
        $size = (Get-Item $LogFile).Length
        if ($size -gt 5MB) {
            Move-Item $LogFile "$LogFile.1" -Force
        }
    }
}

# 1. Expiry check — uninstall if past expiry
if (Test-Path $ExpiryFile) {
    $expiryTs = [int64](Get-Content $ExpiryFile -Raw).Trim()
    $nowTs = [int64](Get-Date -UFormat %s)
    if ($nowTs -gt $expiryTs) {
        Write-Log "expired (now=$nowTs > expiry=$expiryTs); uninstalling task"
        & schtasks /Delete /TN $TaskName /F 2>&1 | Out-File $LogFile -Append -Encoding utf8
        Remove-Item $ExpiryFile -Force -ErrorAction SilentlyContinue
        Remove-Item $LockFile -Force -ErrorAction SilentlyContinue
        exit 0
    }
}

# 2. Lock — bail if another wrapper is running
if (Test-Path $LockFile) {
    $lockAge = (Get-Date) - (Get-Item $LockFile).LastWriteTime
    if ($lockAge.TotalMinutes -lt 90) {
        Write-Log "another scrape running (lock age $($lockAge.TotalMinutes.ToString('F0')) min); skip"
        exit 0
    } else {
        # Stale lock (older than 90min) — previous run probably crashed
        Write-Log "stale lock ($($lockAge.TotalMinutes.ToString('F0')) min); reclaiming"
        Remove-Item $LockFile -Force
    }
}

Set-Content -Path $LockFile -Value $PID

try {
    Rotate-Log
    Write-Log "=== run start (pid=$PID) ==="
    Set-Location $RepoRoot

    # Snapshot dir derived from current time (matches scrape_top5.py default)
    $env:PYTHONIOENCODING = "utf-8"
    $env:PYTHONUTF8 = "1"

    Write-Log "scrape_top5.py --top-n 10"
    $scrapeOut = & $Python -u (Join-Path $RepoRoot "scrape_top5.py") --top-n 10 2>&1
    $scrapeOut | Out-File $LogFile -Append -Encoding utf8
    if ($LASTEXITCODE -ne 0) {
        Write-Log "scrape failed (exit $LASTEXITCODE); skipping parse"
        exit 1
    }

    # Find the snapshot dir scrape_top5 just created (latest under simulation/<today>/)
    $today = Get-Date -Format "yyyy-MM-dd"
    $simRoot = Join-Path $RepoRoot "simulation\$today"
    if (-not (Test-Path $simRoot)) {
        Write-Log "no simulation dir for $today; aborting"
        exit 1
    }
    $latest = Get-ChildItem $simRoot -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $latest) {
        Write-Log "no snapshot under $simRoot; aborting"
        exit 1
    }
    $simDir  = $latest.FullName
    $trajDir = Join-Path $RepoRoot "trajectories\$today\$($latest.Name)"
    Write-Log "parse_replays.py --sim-dir $simDir --out-dir $trajDir"
    $parseOut = & $Python -u (Join-Path $RepoRoot "parse_replays.py") --sim-dir $simDir --out-dir $trajDir 2>&1
    $parseOut | Out-File $LogFile -Append -Encoding utf8

    Write-Log "=== run end (snapshot=$($latest.Name)) ==="
}
finally {
    Remove-Item $LockFile -Force -ErrorAction SilentlyContinue
}
