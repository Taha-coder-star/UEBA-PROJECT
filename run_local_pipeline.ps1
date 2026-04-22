param(
    [switch]$SkipClean,
    [switch]$TrainOnly
)

$ErrorActionPreference = "Stop"
Set-Location "E:\dlp-project"

if ($TrainOnly) {
    python scripts\run_local_pipeline.py --train-only
}
elseif ($SkipClean) {
    python scripts\run_local_pipeline.py --skip-clean
}
else {
    python scripts\run_local_pipeline.py
}
