param(
    [switch]$InstallDeps,
    [switch]$LaunchApp,
    [switch]$SkipClean
)

$ErrorActionPreference = "Stop"
Set-Location "E:\dlp-project"

$arguments = @("scripts\run_everything_from_scratch.py")
if ($InstallDeps) { $arguments += "--install-deps" }
if ($LaunchApp) { $arguments += "--launch-app" }
if ($SkipClean) { $arguments += "--skip-clean" }

python @arguments
