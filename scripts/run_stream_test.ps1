# Run test_codec_stream.exe and report the true Windows exit code.
$env:MAGPIE_MODEL = "C:/Workspace/Playground/VoiceAgent/.cache/magpie/models/magpie-tts-multilingual-357m-q8_0.gguf"
Remove-Item Env:MAGPIE_DEVICE -ErrorAction SilentlyContinue
$p = Start-Process -FilePath "C:/Workspace/magpie-tts.cpp/build-cuda/tests/test_codec_stream.exe" `
    -WorkingDirectory "C:/Workspace/magpie-tts.cpp/build-cuda" `
    -RedirectStandardOutput "C:/Workspace/magpie-tts.cpp/build-cuda/ps_o.txt" `
    -RedirectStandardError  "C:/Workspace/magpie-tts.cpp/build-cuda/ps_e.txt" `
    -NoNewWindow -PassThru -Wait
Write-Host ("ExitCode: 0x{0:X} ({0})" -f $p.ExitCode)
Get-Content "C:/Workspace/magpie-tts.cpp/build-cuda/ps_e.txt" | Select-Object -Last 8
