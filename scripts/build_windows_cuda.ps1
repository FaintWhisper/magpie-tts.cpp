# Build the magpie-tts.cpp fork (CUDA, shared DLL, CLI, tests).
# Usage: powershell -ExecutionPolicy Bypass -File scripts\build_windows_cuda.ps1 [-BuildDir <dir>] [-Tests ON|OFF]
param(
    [string]$BuildDir = "",
    [string]$Configuration = "Release",
    [string]$Tests = "ON",
    [string]$SourceDir = ""
)
$ErrorActionPreference = "Stop"

if (-not $SourceDir) { $SourceDir = Split-Path -Parent $PSScriptRoot }
if (-not $BuildDir)  { $BuildDir  = Join-Path $SourceDir "build-cuda" }

$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) { throw "Visual Studio Build Tools were not found." }
$vs = & $vswhere -latest -products * `
    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
    -property installationPath
if (-not $vs) { throw "Visual Studio C++ Build Tools were not found." }

Import-Module (Join-Path $vs "Common7\Tools\Microsoft.VisualStudio.DevShell.dll")
Enter-VsDevShell -VsInstallPath $vs -SkipAutomaticLocation `
    -DevCmdArguments "-arch=x64 -host_arch=x64"

cmake -S $SourceDir -B $BuildDir -G Ninja `
    "-DCMAKE_BUILD_TYPE=$Configuration" `
    -DMAGPIE_GGML_CUDA=ON `
    -DMAGPIE_SHARED=ON `
    "-DMAGPIE_BUILD_TESTS=$Tests" `
    -DMAGPIE_BUILD_CLI=ON
if ($LASTEXITCODE -ne 0) { throw "Magpie CMake configuration failed." }

cmake --build $BuildDir --parallel 8
if ($LASTEXITCODE -ne 0) { throw "Magpie build failed." }

Write-Host "BUILD OK: $BuildDir"
