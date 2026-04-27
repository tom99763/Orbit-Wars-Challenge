# uninstall_scrape_task.ps1
# Remove the OrbitWarsScrape scheduled task and clean up its state files.

param([string]$TaskName = "OrbitWarsScrape")

$RepoRoot   = Split-Path -Parent $MyInvocation.MyCommand.Path
$ExpiryFile = Join-Path $RepoRoot ".scrape_cron_expiry"
$LockFile   = Join-Path $RepoRoot ".scrape_cron.lock"

& schtasks /Delete /TN $TaskName /F 2>&1 | Out-Host
Remove-Item $ExpiryFile -Force -ErrorAction SilentlyContinue
Remove-Item $LockFile   -Force -ErrorAction SilentlyContinue

Write-Host "scrape task removed. log file (.scrape_cron.log) kept for review."
