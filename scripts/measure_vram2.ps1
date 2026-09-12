# Measure peak GLOBAL VRAM delta while magpie-cli runs (per-process query is N/A on WDDM).
# Usage: powershell -File measure_vram2.ps1 -Mode say|stream -Output <wav>
param(
    [string]$Mode = "say",
    [string]$Output = "aria.wav"
)
$cli = "C:\Workspace\magpie-tts.cpp\build-cuda\examples\cli\magpie-cli.exe"
$model = "C:\Workspace\Playground\VoiceAgent\.cache\magpie\models\magpie-tts-multilingual-357m-q8_0.gguf"
$env:MAGPIE_DEVICE = "cuda"

$baseline = [int](& nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
$peak = $baseline

$p = Start-Process -FilePath $cli -ArgumentList @($Mode, "--model", "`"$model`"",
    "--text", "`"Hola, soy Aria. Esta frase se genera completa y despues se decodifica de una sola vez.`"",
    "--lang", "es", "--speaker", "Aria", "--seed", "1234",
    "--output", "`"$Output`"", "--chunk-frames", "4") `
    -WorkingDirectory "C:\Workspace\magpie-tts.cpp\build-cuda" `
    -RedirectStandardOutput "$env:TEMP\vram_out.txt" `
    -RedirectStandardError  "$env:TEMP\vram_err.txt" `
    -NoNewWindow -PassThru

while (-not $p.HasExited) {
    $u = [int](& nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    if ($u -gt $peak) { $peak = $u }
    Start-Sleep -Milliseconds 150
}
Write-Host ("{0}: exit={1} baseline={2} MiB peak={3} MiB delta={4} MiB" -f `
    $Mode, $p.ExitCode, $baseline, $peak, ($peak - $baseline))
Get-Content "$env:TEMP\vram_out.txt" | Select-Object -Last 2
