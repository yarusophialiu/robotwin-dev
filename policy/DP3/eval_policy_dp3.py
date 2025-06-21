import sys

sys.path.insert(0, "./policy/3D-Diffusion-Policy/3D-Diffusion-Policy")
sys.path.append("./")

import torch
import sapien.core as sapien
import traceback
import os
import numpy as np
from envs import *
import hydra
import pathlib

from dp3_policy import *

import yaml
from datetime import datetime
import importlib

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)

TASK = None


def main(cfg):
    global TASK
    TASK = cfg.task.name
    print("Task name:", TASK)
    checkpoint_num = cfg.checkpoint_num
    expert_data_num = cfg.expert_data_num
    seed = cfg.training.seed
    setting = cfg.setting

    with open(f"./task_config/{cfg.raw_task_name}.yml", "r", encoding="utf-8") as f:
        model_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    dp3 = DP3(cfg, checkpoint_num)


def test_policy(Demo_class, model_config, task_args, dp3, st_seed, test_num=20):
    Demo_class.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **task_args)
    Demo_class.apply_dp3(dp3, task_args)


if __name__ == "__main__":
    from test_render import Sapien_TEST

    Sapien_TEST()
    main()
