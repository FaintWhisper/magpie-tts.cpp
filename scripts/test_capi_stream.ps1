# Compile + run the C-API streaming smoke tester inside VsDevShell.
$ErrorActionPreference = "Continue"
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
Import-Module (Join-Path $vs "Common7\Tools\Microsoft.VisualStudio.DevShell.dll")
Enter-VsDevShell -VsInstallPath $vs -SkipAutomaticLocation -DevCmdArguments "-arch=x64 -host_arch=x64" | Out-Null
cl /nologo /O2 /MT /I"C:\Workspace\magpie-tts.cpp\include" `
   "C:\Workspace\magpie-tts.cpp\tests\test_capi_stream.cpp" `
   /Fe:"C:\Workspace\magpie-tts.cpp\build-cuda\test_capi_stream.exe" `
   /link /LIBPATH:"C:\Workspace\magpie-tts.cpp\build-cuda" magpie-tts.lib 2>&1 | Select-Object -Last 3
$env:MAGPIE_DEVICE = "cuda"
& "C:\Workspace\magpie-tts.cpp\build-cuda\test_capi_stream.exe" "C:\Workspace\Playground\VoiceAgent\.cache\magpie\models\magpie-tts-multilingual-357m-q8_0.gguf" 2>&1 | Select-Object -Last 2
Write-Host ("rc=" + $LASTEXITCODE)
