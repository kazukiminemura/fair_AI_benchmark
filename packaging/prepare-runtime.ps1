# Run on the end user's machine; Microsoft runtime DLLs are not shipped in the ZIP.
$ErrorActionPreference = 'Stop'
try {
    $systemDirectory = [Environment]::SystemDirectory
    foreach ($name in @('msvcp140.dll', 'msvcp140_1.dll', 'vcruntime140.dll', 'vcruntime140_1.dll')) {
        $path = Join-Path $systemDirectory $name
        if (-not (Test-Path -LiteralPath $path)) {
            throw 'Install the Microsoft Visual C++ x64 Redistributable, then launch again: https://aka.ms/vc14/vc_redist.x64.exe'
        }
        if ([version](Get-Item -LiteralPath $path).VersionInfo.FileVersion -lt [version]'14.40.33810.0') {
            throw 'Update the Microsoft Visual C++ x64 Redistributable: https://aka.ms/vc14/vc_redist.x64.exe'
        }
    }
    # NumPy's repaired wheels import this private filename. Copy the installed
    # user's runtime without modifying any third-party binary or distributing it.
    $privateDirectory = Join-Path $PSScriptRoot '_internal/numpy.libs'
    if (-not (Test-Path -LiteralPath $privateDirectory -PathType Container)) {
        throw 'numpy.libs is missing. Extract the complete fairbench ZIP before launching.'
    }
    $privateNames = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'LOCAL_RUNTIME_NAMES.json') -Raw | ConvertFrom-Json
    foreach ($name in $privateNames) {
        if ($name -notmatch '^msvcp140-[0-9a-f]{32}\.dll$') { throw 'Unexpected local runtime filename.' }
        Copy-Item -LiteralPath (Join-Path $systemDirectory 'msvcp140.dll') -Destination (Join-Path $privateDirectory $name) -Force
    }
} catch {
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}
