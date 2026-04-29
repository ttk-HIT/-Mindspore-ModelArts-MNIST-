#LeNet MNIST（MindSpore）

本目录是从 `materials/lenet` 独立整理出的可运行副本，已补全以下内容：

- `src/lenet.py`：LeNet-5 网络结构
- `train.py`：损失函数、优化器、模型构建与训练调用
- `eval.py`：损失函数、模型构建与评估调用
- `scripts/run_train_cpu.ps1`：Windows PowerShell 训练脚本
- `scripts/run_eval_cpu.ps1`：Windows PowerShell 评估脚本
- `logs/`：日志目录与环境检查记录

## 运行前准备

1. 安装 MindSpore。
2. 安装 `requirements.txt` 中的依赖。
3. 按如下结构准备 MNIST 数据：

```text
MNIST_Data/
  train/
    train-images.idx3-ubyte
    train-labels.idx1-ubyte
  test/
    t10k-images.idx3-ubyte
    t10k-labels.idx1-ubyte
```

## 本地运行

训练：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_train_cpu.ps1 -DataPath .\MNIST_Data -CkptPath .\ckpt
```

评估：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_eval_cpu.ps1 -DataPath .\MNIST_Data -CkptPath .\ckpt\checkpoint_lenet-10_1875.ckpt
```

## 当前状态

当前工作区未安装 `mindspore`，因此我已完成代码补全和脚本整理，但没有在此环境中直接跑出训练精度。你后续在装好 MindSpore 的机器上执行后，日志会写入 `logs/`，模型会保存到 `ckpt/`。
