# Long-text VRAM probe: offline `say` with a long passage (big codec working set).
param([string]$Mode = "say")
$cli = "C:\Workspace\magpie-tts.cpp\build-cuda\examples\cli\magpie-cli.exe"
$model = "C:\Workspace\Playground\VoiceAgent\.cache\magpie\models\magpie-tts-multilingual-357m-q8_0.gguf"
$env:MAGPIE_DEVICE = "cuda"
$long = "This is a much longer passage to force many more decoder frames and a proportionally larger codec working set for the offline single pass decode path, so that the difference between the two strategies becomes obvious in the numbers."
$baseline = [int](& nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
$peak = $baseline
$args = @($Mode, "--model", "`"$model`"", "--text", "`"$long`"",
    "--lang", "en", "--speaker", "Aria", "--seed", "1234",
    "--output", "`"aria_long.wav`"", "--chunk-frames", "4")
$p = Start-Process -FilePath $cli -ArgumentList $args `
    -WorkingDirectory "C:\Workspace\magpie-tts.cpp\build-cuda" `
    -RedirectStandardOutput "$env:TEMP\vram_out.txt" `
    -RedirectStandardError  "$env:TEMP\vram_err.txt" `
    -NoNewWindow -PassThru
while (-not $p.HasExited) {
    $u = [int](& nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    if ($u -gt $peak) { $peak = $u }
    Start-Sleep -Milliseconds 150
}
Write-Host ("{0}-long: exit={1} delta={2} MiB" -f $Mode, $p.ExitCode, ($peak - $baseline))
Get-Content "$env:TEMP\vram_out.txt" | Select-Object -Last 2
