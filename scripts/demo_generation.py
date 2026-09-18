"""
The scripts to launch auto scene load and key-point based trajectory generation.
"""
import numpy as np
import os
import open3d as o3d
import mediapy
import argparse
import traceback
from dm_control import viewer
from dm_control.mjcf import physics
from sympy.strategies.branch import condition
from tqdm import tqdm
from datetime import datetime
from scipy.spatial.transform import Rotation as R
from RMMBench.robots import Franka
from RMMBench.tasks import *
from RMMBench.utils.data_utils import save_single_data, process_observations
from RMMBench.utils.utils import get_hanan_l1_distance,find_key_by_value, get_logger,quaternion_to_euler
from RMMBench.envs import load_env
import cv2
from RMMBench.utils.skill_lib import SkillLib
from RMMBench.configs import name2config
os.environ["MUJOCO_GL"] = "egl"
import time

def get_args():
    parser = argparse.ArgumentParser(description='Generate trajectory for a task')
    parser.add_argument('--task-name', default= [ "nav_select_tool_galley_0"], type=str, help='task name, can be a single task (str) or a list of tasks (list)')
    parser.add_argument('--record-video', default=True, help='record video')
    parser.add_argument('--save-dir', default="")
    parser.add_argument('--n-sample', default=1, type=int, help='number of samples to generate')
    parser.add_argument('--start-id', default=0, type=int, help='start index for data storage')
    parser.add_argument('--robot', default="pandaomron", type=str, help='robot name')
    parser.add_argument('--debug', action="store_true", default=False, help='debug mode')
    parser.add_argument('--early-stop', action="store_true", default=False, help='whether use early stop when skill failed to carry out')
    parser.add_argument('--max-episode', default=100, type=int, help='max episode number in the directory')
    parser.add_argument('--eval-unseen', default=False, action="store_true", help='evaluate unseen object categories')
    args = parser.parse_args()
    # Normalize task-name to a list
    if isinstance(args.task_name, list):
        args.task_list = args.task_name
    else:
        args.task_list = [args.task_name]
    return args


def generate_trajectory(args, task_name, index, logger):
    env = load_env(task_name, robot=args.robot, eval=args.eval_unseen)

    env.reset()

    init_ee_pos, init_ee_quat = env.robot.get_end_effector_pos(env.physics), env.robot.get_end_effector_quat(env.physics)



    robot_init_info = env.robot.robot_init_info

    target_entity = env.task.config_manager.target_entity

    instruction = env.task.get_instruction()


    # register the expert sequence
    skill_seq = env.get_expert_skill_sequence()

    # start auto trajectory generation
    observations, waypoints = [], []
    obs = [env.get_observation()]
    task_dir=None
    if skill_seq is not None:  # normal case
        for i, skill in enumerate(skill_seq):
            if skill.func.__name__ == "moveforward":
                obs, waypoint, stage_success, task_success, _ = skill(env)
            else:
                obs, waypoint, stage_success, task_success = skill(env)



            if not task_dir:
                task_dir = os.path.join(args.save_dir, task_name)

                time_img = time.strftime("%Y-%m-%d_%H.%M.%S", time.localtime())

                task_dir = os.path.join(task_dir, f"{time_img}_{target_entity}")

                img_dir = os.path.join(task_dir, f"{target_entity}_img")
                os.makedirs(img_dir, exist_ok=True)
                os.makedirs(task_dir, exist_ok=True)
            env.update_pcd_generator()


            if args.debug:
                for o in obs: observations.append(dict(rgb=o["rgb"]))
            else:
                observations.extend(obs)
                waypoints.extend(waypoint)
            if args.early_stop and not stage_success:
                logger.warning(f"{skill} failed, early quit...")
                break
            if i == len(skill_seq)-1:
                break

    else:  # TODO: some special tasks should be handled based on the feedback
        raise NotImplementedError("No expert skill sequence found")
    if args.record_video:
        frames = []
        fs = []

        for obs in observations:
            frame = np.vstack([np.hstack([obs["rgb"][5],obs["rgb"][2]]), np.hstack([obs["rgb"][3], obs["rgb"][4]])])
            f = obs["rgb"][2]

            fs.append(f)
            frames.append(frame)
        if not os.path.exists(task_dir):
            os.makedirs(task_dir)

        timename = time.strftime("%H.%M.%S", time.localtime())

        mediapy.write_video(os.path.join(task_dir, f"{target_entity}_{task_success}_{timename}.mp4"),
                            frames, fps=10)
        mediapy.write_video(os.path.join(task_dir, f"forward.mp4"),
                            fs, fps=10)


    if not env.task.task_success:
        logger.warning("Task failed, skip saving data")
        return
    else:
        logger.info("Task success, saving data")


    env.close()

        
if __name__ == "__main__":
    args = get_args()
    logger = get_logger()
    for task_name in args.task_list:
        logger.info(f"Starting task: {task_name}")
        for i in tqdm(range(args.n_sample), desc=f"Task {task_name}"):
            i += args.start_id
            try:

                generate_trajectory(args, task_name, i, logger)
            except Exception as e:
                err = traceback.TracebackException.from_exception(e)
                print("".join(err.format()))
                continue
