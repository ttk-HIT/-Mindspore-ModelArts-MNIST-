param(
    [string]$DataPath = ".\MNIST_Data",
    [string]$CkptPath = ".\ckpt\checkpoint_lenet.ckpt",
    [string]$LogPath = ".\logs\eval_cpu.log"
)

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Resolve-Path (Join-Path $scriptDir "..")
$resolvedLogPath = Join-Path $projectRoot $LogPath

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $resolvedLogPath) | Out-Null

Push-Location $projectRoot
try {
    python eval.py --data_path $DataPath --device_target CPU --ckpt_path $CkptPath 2>&1 | Tee-Object -FilePath $resolvedLogPath
} finally {
    Pop-Location
}
