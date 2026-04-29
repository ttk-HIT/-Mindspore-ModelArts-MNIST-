# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
######################## eval lenet example ########################
eval lenet according to model file:
python eval.py --data_path /YourDataPath --ckpt_path Your.ckpt
"""

import argparse
import ast
import json
import os

import mindspore.nn as nn
from mindspore import context
from mindspore.nn.metrics import Accuracy
from mindspore.train import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.config import mnist_cfg as cfg
from src.dataset import create_dataset
from src.lenet import LeNet5


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MindSpore Lenet Example")
    parser.add_argument(
        "--device_target",
        type=str,
        default="CPU",
        choices=["Ascend", "GPU", "CPU"],
        help="device where the code will be implemented (default: CPU)",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./MNIST_Data",
        help="path where the dataset is saved",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="",
        help="must provide path where the trained ckpt file is saved",
    )
    parser.add_argument(
        "--dataset_sink_mode",
        type=ast.literal_eval,
        default=False,
        help="dataset_sink_mode is False or True",
    )

    args = parser.parse_args()
    
    # If no ckpt_path provided, auto-find the latest checkpoint
    if not args.ckpt_path:
        ckpt_dir = "./ckpt"
        if os.path.exists(ckpt_dir) and os.path.isdir(ckpt_dir):
            ckpt_files = [f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")]
            if ckpt_files:
                # Sort by modification time and pick the latest
                ckpt_files.sort(key=lambda x: os.path.getmtime(os.path.join(ckpt_dir, x)))
                args.ckpt_path = os.path.join(ckpt_dir, ckpt_files[-1])
                print(f"Auto-detected latest checkpoint: {args.ckpt_path}")
            else:
                raise ValueError("No checkpoint files found in ./ckpt directory. Please provide --ckpt_path.")
        else:
            raise ValueError("Please provide --ckpt_path for evaluation.")

    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)

    network = LeNet5(cfg.num_classes)
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    model = Model(network, net_loss, metrics={"Accuracy": Accuracy()})

    print("============== Starting Testing ==============")
    param_dict = load_checkpoint(args.ckpt_path)
    load_param_into_net(network, param_dict)
    ds_eval = create_dataset(os.path.join(args.data_path, "test"), cfg.batch_size, 1)
    acc = model.eval(ds_eval, dataset_sink_mode=args.dataset_sink_mode)
    print("============== {} ==============".format(acc))
    
    # Save evaluation results to file
    result_path = "eval_result.json"
    result = {
        "ckpt_path": args.ckpt_path,
        "accuracy": acc["Accuracy"],
        "data_path": args.data_path,
        "device_target": args.device_target
    }
    with open(result_path, "w") as f:
        json.dump(result, f, indent=4)
    print("Evaluation result saved to {}".format(result_path))
