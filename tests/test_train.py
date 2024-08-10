"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import argparse
import os
import glob
import random
import yaml

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import wandb

import medlvlm.tasks as tasks
from medlvlm.common.config import Config
from medlvlm.common.dist_utils import get_rank, init_distributed_mode
from medlvlm.common.logger import setup_logger
from medlvlm.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
from medlvlm.common.registry import registry
from medlvlm.common.utils import now

# imports modules for registration
from medlvlm.models import *
from tests.test_evaluate import *
from medlvlm.processors import *
from medlvlm.runners import *
from medlvlm.tasks import *

def list_of_str(arg):
    return list(map(str, arg.split(',')))

def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--cfg-path", required=True, help="path to train configuration file.")
    parser.add_argument("--cfg-eval-path", required=False, help="path to evaluation configuration file.")
    parser.add_argument("--eval-dataset", type=list_of_str, default='val_vindrcxr', help="dataset to evaluate")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()

    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def get_runner_class(cfg):
    """
    Get runner class from config. Default to epoch-based runner.
    """
    runner_cls = registry.get_runner_class(cfg.run_cfg.get("runner", "runner_base"))

    return runner_cls


def main():
    # allow auto-dl completes on main process without timeout when using NCCL backend.
    # os.environ["NCCL_BLOCKING_WAIT"] = "1"

    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    job_id = now()
    args = parse_args()
    cfg = Config(args)

    init_distributed_mode(cfg.run_cfg)
    setup_seeds(cfg)

    # set after init_distributed_mode() to only log on master.
    setup_logger()
    cfg.pretty_print()

    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)
    model = task.build_model(cfg)

    if cfg.run_cfg.wandb_log:
        wandb.login(key=cfg.run_cfg.wandb_token)
        wandb.init(project="ars2text", name=cfg.run_cfg.job_name)
        wandb.watch(model)

    runner = get_runner_class(cfg)(
        cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
    )
    runner.train()

    if hasattr(args, 'cfg_eval_path'):
        args.cfg_path = args.cfg_eval_path

        model_path = "{}/{}".format(cfg.run_cfg.output_dir, job_id)
        ckpt_paths = glob.glob(os.path.join(model_path, "*.pth"))
        ckpt_names = [os.path.basename(ckp_path) for ckp_path in ckpt_paths]
        last_ckpt_name = sorted(ckpt_names, key=lambda x: int(x.split(".")[0].split("_")[-1]))[-1]
        last_ckpt_path = os.path.join(model_path, last_ckpt_name)

        with open(args.cfg_path) as f:
            eval_cfg = yaml.load(f, Loader=yaml.FullLoader)
            eval_cfg["model"]["ckpt"] = last_ckpt_path

        with open(args.cfg_path, "w") as f:
            yaml.dump(
                eval_cfg, stream=f, default_flow_style=False, sort_keys=False
            )
        
        print("Evaluating...........")
        evaluate(args)
        print("Done!")

if __name__ == "__main__":
    main()