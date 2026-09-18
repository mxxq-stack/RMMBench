import os
import sys

import numpy as np
import open3d
import cv2
from dm_control import viewer

from RMMBench.envs import load_env
from RMMBench.tasks import *
import json

from RMMBench.tasks.condition import get_leaf_met_progress
from RMMBench.utils.utils import extract_containers,get_geom_ids_by_prefix,get_entity_mask_from_seg,get_placement_top_down_view

from pathlib import Path

from RMMBench.robots.single_arm.franka import Franka

task = 'pick_chocolate'

robot = "pandaomron"


env = load_env(task,robot=robot,time_limit=1000)



env.reset()
target_entity = env.task.config_manager.target_entity

viewer.launch(env)  # resets the env once automatically

env.close()

