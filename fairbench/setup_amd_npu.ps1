param([string]$Installer)
$ErrorActionPreference = 'Stop'

# AMD's documented Windows installation flow (Ryzen AI Software 1.8.0).
# https://ryzenai.docs.amd.com/en/1.8.0/inst.html
# Keep the license agreement and installer UI visible to the user.
function Start-AMDInstaller([string]$File) {
    $signature = Get-AuthenticodeSignature -LiteralPath $File
    if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notmatch 'O="?Advanced Micro Devices,? Inc\.?"?(,|$)') {
        throw "Installer does not have a valid AMD signature: $File"
    }
    $process = Start-Process -FilePath $File -Verb RunAs -Wait -PassThru -WindowStyle Normal -WorkingDirectory (Split-Path -Parent $File)
    if ($process.ExitCode -eq 3010 -or $process.ExitCode -eq 1641) {
        throw 'Windows restart required. Restart, then run setup-npu again.'
    }
    if ($process.ExitCode -ne 0) { throw "Installer failed ($($process.ExitCode)): $File" }
}

try {
    $cpu = Get-CimInstance Win32_Processor
    if (-not ($cpu.Manufacturer -contains 'AuthenticAMD')) {
        throw 'Run this command on the AMD laptop. No system changes were made.'
    }
    $os = Get-CimInstance Win32_OperatingSystem
    if ([int]$os.BuildNumber -lt 22621 -or -not [Environment]::Is64BitOperatingSystem) {
        throw 'Ryzen AI requires Windows 11 x64 (see AMD supported configurations).'
    }
    $cache = Join-Path $env:LOCALAPPDATA 'fairbench\amd-npu'
    New-Item -ItemType Directory -Force -Path $cache | Out-Null

    # Use native installers rather than installing AMD's ORT into .venv.
    # Winget retains the publishers' agreement prompts.
    foreach ($dependency in @(
        @{ Command = 'conda'; Package = 'CondaForge.Miniforge3' },
        @{ Command = 'cmake'; Package = 'Kitware.CMake' }
    )) {
        if (-not (Get-Command $dependency.Command -ErrorAction SilentlyContinue)) {
            if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
                throw 'Install Windows App Installer (winget), then retry setup-npu.'
            }
            & winget install --id $dependency.Package --exact --source winget --interactive
            if ($LASTEXITCODE -ne 0) { throw "Prerequisite installation failed: $($dependency.Package)" }
        }
    }
    # Newly installed Miniforge must be visible to AMD's installer in this session.
    foreach ($prefix in @((Join-Path $env:USERPROFILE 'miniforge3'), (Join-Path $env:LOCALAPPDATA 'miniforge3'), (Join-Path $env:ProgramData 'miniforge3'))) {
        if (Test-Path -LiteralPath (Join-Path $prefix 'condabin')) {
            $env:PATH = "$prefix;$prefix\Scripts;$prefix\condabin;$env:PATH"
        }
    }
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    $hasVC = $false
    if (Test-Path -LiteralPath $vswhere) {
        $hasVC = [bool](& $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath)
    }
    if (-not $hasVC) {
        if (-not (Get-Command winget -ErrorAction SilentlyContinue)) { throw 'Visual Studio 2022 C++ tools are required. Install winget and retry.' }
        & winget install --id Microsoft.VisualStudio.2022.BuildTools --exact --source winget --interactive --override '--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended --norestart'
        if ($LASTEXITCODE -ne 0) { throw 'Visual Studio C++ tools installation failed. Complete installation and retry.' }
    }

    if (-not $Installer) {
        # Ask AMD for the SDK through its official agreement page. Never bypass it.
        $downloads = (New-Object -ComObject Shell.Application).NameSpace('shell:Downloads').Self.Path
        $started = Get-Date
        Start-Process 'https://account.amd.com/en/forms/downloads/ryzenai-eula-public-xef.html?filename=ryzen-ai-1.8.0.exe'
        Write-Host "Accept AMD's agreement and download ryzen-ai-1.8.0.exe to $downloads. Waiting up to 15 minutes..."
        do {
            $file = Get-ChildItem -LiteralPath $downloads -Filter 'ryzen-ai-1.8.0*.exe' -File |
                Where-Object { $_.LastWriteTime -ge $started } | Sort-Object LastWriteTime -Descending | Select-Object -First 1
            if ($file) { $Installer = $file.FullName; break }
            Start-Sleep -Seconds 3
        } while ((Get-Date) -lt $started.AddMinutes(15))
        if (-not $Installer) { throw 'Download not found. Retry with setup-npu --installer <path-to-downloaded-exe>.' }
    }
    $Installer = (Resolve-Path -LiteralPath $Installer).Path
    if ([IO.Path]::GetExtension($Installer) -ne '.exe') { throw 'SDK installer must be an .exe file.' }

    $archive = Join-Path $cache 'NPU_RAI_376_WHQL.zip'
    Write-Host 'Downloading the AMD NPU driver...'
    Invoke-WebRequest -UseBasicParsing -Uri 'https://download.amd.com/opendownload/RyzenAI/Driver/NPU_RAI_376_WHQL.zip' -OutFile $archive
    $driverDir = Join-Path $cache ([guid]::NewGuid().ToString())
    Expand-Archive -LiteralPath $archive -DestinationPath $driverDir
    $driver = @(Get-ChildItem -LiteralPath $driverDir -Recurse -Filter npu_sw_installer.exe -File)
    if ($driver.Count -ne 1) { throw 'AMD driver archive did not contain exactly one npu_sw_installer.exe.' }
    Write-Host 'Installing AMD NPU driver. Complete the administrator prompt and installer UI.'
    Start-AMDInstaller $driver[0].FullName
    Write-Host 'Installing Ryzen AI Software. Complete the license and Conda environment setup in the installer.'
    Start-AMDInstaller $Installer
} catch {
    Write-Error $_ -ErrorAction Continue
    exit 1
}
exit 0
