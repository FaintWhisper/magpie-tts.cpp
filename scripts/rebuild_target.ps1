# Rebuild only test_codec_stream target (with VS env).
param([string]$SourceDir = "C:\Workspace\magpie-tts.cpp", [string]$BuildDir = "C:\Workspace\magpie-tts.cpp\build-cuda", [string]$Target = "test_codec_stream")
$ErrorActionPreference = "Stop"
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
Import-Module (Join-Path $vs "Common7\Tools\Microsoft.VisualStudio.DevShell.dll")
Enter-VsDevShell -VsInstallPath $vs -SkipAutomaticLocation -DevCmdArguments "-arch=x64 -host_arch=x64"
Set-Location $BuildDir
ninja $Target
if ($LASTEXITCODE -ne 0) { throw "build failed" }
Write-Host "TARGET OK: $Target"
