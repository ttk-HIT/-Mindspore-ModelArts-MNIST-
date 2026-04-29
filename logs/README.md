# 日志说明

本目录用于保存 LeNet 训练与评估日志。

推荐运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_train_cpu.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\run_eval_cpu.ps1 -CkptPath .\ckpt\你的模型文件.ckpt
```

当前工作区未安装 `mindspore`，因此尚未生成真实训练日志。可先参考 `env_check.txt` 查看本地检查结果。
