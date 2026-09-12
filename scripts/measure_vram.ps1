# Measure peak VRAM of a magpie-cli run by polling per-process usage.
# Usage: powershell -File measure_vram.ps1 -Mode say|stream -Output <wav>
param(
    [string]$Mode = "say",
    [string]$Output = "aria.wav"
)
$ErrorActionPreference = "Continue"
$cli = "C:\Workspace\magpie-tts.cpp\build-cuda\examples\cli\magpie-cli.exe"
$model = "C:\Workspace\Playground\VoiceAgent\.cache\magpie\models\magpie-tts-multilingual-357m-q8_0.gguf"
$env:MAGPIE_DEVICE = "cuda"

$p = Start-Process -FilePath $cli -ArgumentList @($Mode, "--model", "`"$model`"",
    "--text", "`"Hola, soy Aria. Esta frase se genera completa y despues se decodifica de una sola vez.`"",
    "--lang", "es", "--speaker", "Aria", "--seed", "1234",
    "--output", "`"$Output`"", "--chunk-frames", "4") `
    -WorkingDirectory "C:\Workspace\magpie-tts.cpp\build-cuda" `
    -RedirectStandardOutput "$env:TEMP\vram_out.txt" `
    -RedirectStandardError  "$env:TEMP\vram_err.txt" `
    -NoNewWindow -PassThru

$peak = 0
while (-not $p.HasExited) {
    try {
        $rows = & nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>$null
        foreach ($row in $rows) {
            $parts = $row -split ",\s*"
            if ($parts.Count -ge 2 -and $parts[0].Trim() -eq "$($p.Id)") {
                $mb = [int]$parts[1].Trim()
                if ($mb -gt $peak) { $peak = $mb }
            }
        }
    } catch {}
    Start-Sleep -Milliseconds 200
}
Write-Host ("{0}: PID {1} exit={2} peak-VRAM={3} MiB" -f $Mode, $p.Id, $p.ExitCode, $peak)
Get-Content "$env:TEMP\vram_out.txt" | Select-Object -Last 2
