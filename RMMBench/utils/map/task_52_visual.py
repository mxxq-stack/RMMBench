import os
import sys

import numpy as np
import open3d
import cv2
from dm_control import viewer

from RMMBench.envs import load_env
from RMMBench.tasks import *
import json

from RMMBench.utils.utils import extract_containers,get_geom_ids_by_prefix,get_entity_mask_from_seg,get_placement_top_down_view

from pathlib import Path

from RMMBench.robots.single_arm.franka import Franka
# Check whether the point cloud is empty

task = 'pick_whisk_knife_0'

robot = "pandaomron"


env = load_env(task,robot=robot,time_limit=1000)
# env = load_env(task,robot=robot,episode_config=config,time_limit=1000)


env.reset()


rgb_data = env.get_observation()
# Cameras 0-2: fixed camera views, corresponding to right, left, and forward respectively; cameras 3 and 4 correspond to wrist and hand
n= 1
vlm_img_input = [cv2.cvtColor(rgb_data["rgb"][i], cv2.COLOR_BGR2RGB) for i in range(6)]
img_path = "/Users/lh/work/RMMBench/test_img/all_cam_vis"
cv2.imwrite(os.path.join(img_path, f"wrist_img_{n}.png"), vlm_img_input[3])
cv2.imwrite(os.path.join(img_path, f"right_img_{n}.png"), vlm_img_input[0])
cv2.imwrite(os.path.join(img_path, f"left_img_{n}.png"), vlm_img_input[1])
cv2.imwrite(os.path.join(img_path, f"opposite_img_{n}.png"), vlm_img_input[4])
cv2.imwrite(os.path.join(img_path, f"head_img_{n}.png"), vlm_img_input[5])
cv2.imwrite(os.path.join(img_path, f"top_img_{n}.png"), vlm_img_input[2])
# viewer.launch(env)# automatically performs one reset

env.close()
exit()

