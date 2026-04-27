# install_scrape_task.ps1
#
# Create a Windows scheduled task that runs scrape_cron_wrapper.ps1
# hourly. Matches the install_cron.sh contract from the Linux side:
# self-expires after N days (default 7) so it can't run forever.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File install_scrape_task.ps1
#   powershell -ExecutionPolicy Bypass -File install_scrape_task.ps1 -Days 14 -EveryMinutes 30
#
# After install:
#   schtasks /Query /TN OrbitWarsScrape /FO LIST  — see status
#   tail -f .scrape_cron.log                       — watch progress
#   .\uninstall_scrape_task.ps1                    — stop early

param(
    [int]$Days = 7,
    [int]$EveryMinutes = 60,
    [string]$TaskName = "OrbitWarsScrape"
)

$ErrorActionPreference = "Stop"

$RepoRoot   = Split-Path -Parent $MyInvocation.MyCommand.Path
$Wrapper    = Join-Path $RepoRoot "scrape_cron_wrapper.ps1"
$ExpiryFile = Join-Path $RepoRoot ".scrape_cron_expiry"

if (-not (Test-Path $Wrapper)) {
    throw "wrapper not found: $Wrapper"
}

# Compute expiry (epoch seconds, now + Days)
$expiry = [int64](Get-Date -UFormat %s) + ($Days * 86400)
Set-Content -Path $ExpiryFile -Value $expiry
Write-Host "expiry: $((Get-Date 01-01-1970).AddSeconds($expiry).ToLocalTime())"

# Build the action — run the wrapper via powershell.exe with bypass policy
$startTime = (Get-Date).AddMinutes(2).ToString("HH:mm")
$cmd = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$Wrapper`""

# Force-overwrite if already exists (ignore "task not found" on first install)
try {
    $oldPref = $ErrorActionPreference
    $ErrorActionPreference = "SilentlyContinue"
    & schtasks /Delete /TN $TaskName /F 2>&1 | Out-Null
    $ErrorActionPreference = $oldPref
} catch { }
$LASTEXITCODE = 0

& schtasks /Create `
    /TN $TaskName `
    /TR $cmd `
    /SC MINUTE `
    /MO $EveryMinutes `
    /ST $startTime `
    /F | Out-Host

if ($LASTEXITCODE -ne 0) {
    throw "schtasks /Create failed (exit $LASTEXITCODE)"
}

Write-Host ""
Write-Host "task '$TaskName' installed:"
Write-Host "  every $EveryMinutes min, first run at $startTime"
Write-Host "  expires after $Days days"
Write-Host "  wrapper: $Wrapper"
Write-Host "  log:     $(Join-Path $RepoRoot '.scrape_cron.log')"
Write-Host ""
Write-Host "watch progress:"
Write-Host "  Get-Content -Wait $(Join-Path $RepoRoot '.scrape_cron.log')"
Write-Host ""
Write-Host "stop early:"
Write-Host "  .\uninstall_scrape_task.ps1"
