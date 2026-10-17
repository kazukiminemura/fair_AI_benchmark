<#
.SYNOPSIS
    Build the fairbench distributables.

.DESCRIPTION
    Produces two artefacts in `dist/`:

      1. `dist/fairbench-<version>-<platform>.zip`
         A frozen, self-contained build. No Python required on the target
         machine for the CPU path.

      2. `dist/fairbench-<version>-py3-none-any.whl`
         The normal Python package, for `uv tool install` / `pip install`.
         Recommended when the target already has Python, because the OpenVINO
         NPU plugin and the GPU extras install cleanly through the package
         manager instead of being baked in.

    Everything runs through uv, so the build environment is reproducible from
    uv.lock.

.PARAMETER SkipFrozen
    Only build the wheel (fast).

.PARAMETER SkipWheel
    Only build the frozen bundle.

.PARAMETER NoZip
    Leave the frozen bundle as a directory instead of zipping it.

.EXAMPLE
    pwsh -File scripts\build.ps1
#>
param(
    [switch]$SkipFrozen,
    [switch]$SkipWheel,
    [switch]$NoZip
)

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
& (Join-Path $PSScriptRoot 'check-web.ps1')
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "uv is required: https://docs.astral.sh/uv/getting-started/installation/" -ForegroundColor Red
    exit 1
}

$version = (Select-String -Path "fairbench\__init__.py" -Pattern '__version__ = "([^"]+)"').Matches[0].Groups[1].Value
$platform = if ($env:OS -eq "Windows_NT") { "win64" } elseif ($IsMacOS) { "macos" } else { "linux" }
$app = "dist\fairbench"

Write-Host ""
Write-Host " fairbench $version -> $platform" -ForegroundColor Magenta
Write-Host ""

Write-Host "[1/5] syncing the build environment from uv.lock" -ForegroundColor Cyan
uv sync
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if (-not $SkipWheel) {
    Write-Host "[2/5] building the wheel" -ForegroundColor Cyan
    uv build --wheel --out-dir dist
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} else {
    Write-Host "[2/5] wheel skipped" -ForegroundColor DarkGray
}

if (-not $SkipFrozen) {
    Write-Host "[3/5] freezing the executable" -ForegroundColor Cyan
    # Keep mutually exclusive ORT wheels out of the same environment. The
    # Windows bundle uses DirectML, which includes both CPU and GPU providers.
    $previousBuildEnvironment = $env:UV_PROJECT_ENVIRONMENT
    try {
        $env:UV_PROJECT_ENVIRONMENT = Join-Path $root '.venv-build'
        $syncArgs = @('sync', '--locked', '--no-install-project')
        if ($platform -eq 'win64') {
            $syncArgs += @('--extra', 'directml', '--no-install-package', 'onnxruntime')
        }
        uv @syncArgs
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        uv run --no-sync pyinstaller --noconfirm --clean --distpath dist --workpath build packaging\fairbench.spec
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    } finally {
        $env:UV_PROJECT_ENVIRONMENT = $previousBuildEnvironment
    }

    # Ship what the frozen build cannot bake in: the worker source, so a GPU
    # runtime living in another interpreter can be used, plus the docs.
    Write-Host "[4/5] staging the supporting files" -ForegroundColor Cyan
    New-Item -ItemType Directory -Force -Path "$app\src" | Out-Null
    if (Test-Path "$app\src\fairbench") { Remove-Item -Recurse -Force "$app\src\fairbench" }
    Copy-Item -Recurse -Force "fairbench" "$app\src\fairbench"
    Get-ChildItem -Path "$app\src" -Recurse -Directory -Filter "__pycache__" |
        Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

    Copy-Item -Force "README.md" "$app\README.md"
    Copy-Item -Force "docs\METHODOLOGY.md" "$app\METHODOLOGY.md"
    Copy-Item -Force "pyproject.toml" "$app\pyproject.toml"

    # Start the dashboard without requiring Python, uv, or a terminal command.
    $cmd = @(
        "@echo off",
        "cd /d `"%~dp0`"",
        '"%~dp0fairbench.exe" serve %*',
        "if errorlevel 1 pause"
    ) -join "`r`n"
    Set-Content -Encoding ASCII -Path "$app\start-fairbench.cmd" -Value $cmd

    if (-not $NoZip) {
        $zip = "dist\fairbench-$version-$platform.zip"
        if (Test-Path $zip) { Remove-Item -Force $zip }
        Compress-Archive -Path "$app\*" -DestinationPath $zip -CompressionLevel Optimal
    }
} else {
    Write-Host "[3/5] frozen build skipped" -ForegroundColor DarkGray
    Write-Host "[4/5] staging skipped" -ForegroundColor DarkGray
}

Write-Host "[5/5] done" -ForegroundColor Cyan
Write-Host ""
Get-ChildItem dist | ForEach-Object {
    if ($_.PSIsContainer) {
        $size = (Get-ChildItem -Recurse $_.FullName | Measure-Object -Property Length -Sum).Sum / 1MB
        "{0,-44} {1,8:N1} MB" -f ($_.Name + "\"), $size
    }
    else {
        "{0,-44} {1,8:N1} MB" -f $_.Name, ($_.Length / 1MB)
    }
}
