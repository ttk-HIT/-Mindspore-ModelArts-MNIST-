param(
    [string]$DataPath = ".\MNIST_Data",
    [string]$CkptPath = ".\ckpt",
    [string]$LogPath = ".\logs\train_cpu.log"
)

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Resolve-Path (Join-Path $scriptDir "..")
$resolvedLogPath = Join-Path $projectRoot $LogPath
$resolvedCkptPath = Join-Path $projectRoot $CkptPath

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $resolvedLogPath) | Out-Null
New-Item -ItemType Directory -Force -Path $resolvedCkptPath | Out-Null

Push-Location $projectRoot
try {
    python train.py --data_path $DataPath --device_target CPU --ckpt_path $CkptPath 2>&1 | Tee-Object -FilePath $resolvedLogPath
} finally {
    Pop-Location
}
