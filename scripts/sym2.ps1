# Compile and run the dbghelp RVA symbolizer inside VsDevShell.
param([string]$RvaHex = "39FB")
$ErrorActionPreference = "Continue"
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
Import-Module (Join-Path $vs "Common7\Tools\Microsoft.VisualStudio.DevShell.dll")
Enter-VsDevShell -VsInstallPath $vs -SkipAutomaticLocation -DevCmdArguments "-arch=x64 -host_arch=x64"
$src = Join-Path $env:TEMP "sym2.cpp"
if (-not (Test-Path $src)) { $src = "C:\Workspace\magpie-tts.cpp\build-cuda\sym2.cpp" }
cl /nologo /O2 $src /Fe:"$env:TEMP\sym2.exe" dbghelp.lib 2>&1 | Select-Object -Last 2
& "$env:TEMP\sym2.exe" "C:\Workspace\magpie-tts.cpp\build-cuda\tests\test_codec_stream.exe" $RvaHex
