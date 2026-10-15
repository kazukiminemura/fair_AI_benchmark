# fairbench - one-command suite for a laptop review.
#
#   pwsh -File scripts/run-suite.ps1              # quick pass, ~5 minutes
#   pwsh -File scripts/run-suite.ps1 -Full        # publication settings
#   pwsh -File scripts/run-suite.ps1 -Targets "ort:CPUExecutionProvider","ort:DmlExecutionProvider@.venv-gpu"
#
# Run it plugged in, with the lid open, on the High performance or Balanced
# power plan, and close everything else first.

param(
    [switch]$Full,
    [switch]$Quick,
    [string[]]$Targets,
    [string[]]$Workloads,
    [int]$Repeats = 0,
    [switch]$NoServe
)

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"

# Prefer the project's uv environment when one exists, so the suite runs with
# the pinned toolchain; fall back to whatever `python` resolves to.
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$runner = if ((Get-Command uv -ErrorAction SilentlyContinue) -and (Test-Path "uv.lock")) {
    { param($a) uv run --no-sync fairbench @a }
} else {
    { param($a) python -m fairbench @a }
}

Write-Host ""
Write-Host " fairbench - pre-flight" -ForegroundColor Magenta
Write-Host ""

# --- power source ----------------------------------------------------------
$battery = Get-CimInstance Win32_Battery -ErrorAction SilentlyContinue | Select-Object -First 1
if ($battery) {
    if ($battery.BatteryStatus -in 1) {
        Write-Host "  [!!] On battery. Clocks are capped and the result will not be comparable." -ForegroundColor Red
        Write-Host "       Plug in, or add -AllowBattery once you know what you are doing."
        exit 2
    }
    Write-Host "  [ok] On AC power." -ForegroundColor Green
} else {
    Write-Host "  [ok] No battery (desktop or VM)." -ForegroundColor Green
    Write-Host "       Energy per inference will be unavailable; everything else still works."
}

# --- power plan ------------------------------------------------------------
$plan = (powercfg /getactivescheme) -join " "
if ($plan -match "a1841308") {
    Write-Host "  [!!] Power saver is active. Switch to Balanced or High performance." -ForegroundColor Red
    exit 2
}
Write-Host "  [ok] Power plan: $($plan.Trim())" -ForegroundColor Green

# --- background load -------------------------------------------------------
$load = (Get-Counter '\Processor(_Total)\% Processor Time' -MaxSamples 3 -SampleInterval 1).CounterSamples |
    Measure-Object -Property CookedValue -Average
$avg = [math]::Round($load.Average, 1)
if ($avg -gt 15) {
    Write-Host "  [!!] $avg% CPU busy. Close other work for a trustworthy run." -ForegroundColor Yellow
} else {
    Write-Host "  [ok] Background CPU load: $avg%" -ForegroundColor Green
}

# --- devices ---------------------------------------------------------------
Write-Host ""
& $runner @("devices")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# --- run -------------------------------------------------------------------
$runArgs = @("run")
if ($Full)      { $runArgs += "--full" }
elseif ($Quick) { $runArgs += "--quick" }
if ($Repeats -gt 0) { $runArgs += @("--repeats", "$Repeats") }
if ($Targets)   { $runArgs += "--targets";   $runArgs += $Targets }
if ($Workloads) { $runArgs += "--workloads"; $runArgs += $Workloads }
if ($NoServe)   { $runArgs += "--no-open" }

Write-Host ""
Write-Host " fairbench $($runArgs -join ' ')" -ForegroundColor Magenta
& $runner $runArgs
exit $LASTEXITCODE
