"""
The scripts to launch auto scene load and key-point based trajectory generation.
"""

import requests
import re
import json
import random
from RMMBench.configs.prompt.vlm_prompt import prompt_messages

import numpy as np
import os

os.environ["DISPLAY"] = ":1"
import open3d as o3d
import mediapy
import argparse
import traceback
from dm_control import viewer
from tqdm import tqdm
from datetime import datetime
from scipy.spatial.transform import Rotation as R
from RMMBench.robots import Franka
from RMMBench.tasks import *
from RMMBench.utils.data_utils import save_single_data, process_observations
from RMMBench.utils.utils import find_key_by_value, get_logger,get_direction_from_euler
from RMMBench.envs import load_env
import cv2
from RMMBench.utils.skill_lib import SkillLib
from RMMBench.configs import name2config
from RMMBench.configs.restart_prompt_generate.vlm_prompt import VLAMessageHandler

import time

from RMMBench.tasks.primitive.base import PrimitiveTask

os.environ["MUJOCO_GL"] = "egl"

#history_maxlen
def get_args():
    parser = argparse.ArgumentParser(description='Generate trajectory for a task')
    parser.add_argument('--vlm-url', default="http://10.146.207.96:8080", type=str, help='10.118.2.117:8080')
    parser.add_argument('--max-skills-num', default=20, type=int, help='number of skills to generate')
    parser.add_argument('--history-maxlen', default=8, type=int, help='number of history maxlen')
    parser.add_argument('--task-name', default="cook_steak", type=str, help='task name')
    parser.add_argument('--record-video', default=True, help='record video')
    parser.add_argument('--save-dir', default="/home/hadoop-aipnlp/work_lhp/RMMBench/VLM_video/test_graspnet")
    parser.add_argument('--n-sample', default=1, type=int, help='number of samples to generate')
    parser.add_argument('--start-id', default=0, type=int, help='start index for data storage')
    parser.add_argument('--robot', default="pandaomron", type=str, help='robot name')
    parser.add_argument('--debug', action="store_true", default=False, help='debug mode')
    parser.add_argument('--early-stop', action="store_true", default=False,
                        help='whether use early stop when skill failed to carry out')
    parser.add_argument('--max-episode', default=100, type=int, help='max episode number in the directory')
    parser.add_argument('--eval-unseen', default=False, action="store_true", help='evaluate unseen object categories')
    args = parser.parse_args()
    return args


def get_all_hdf5_files(directory):
    hdf5_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.hdf5'):
                hdf5_files.append(os.path.join(root, file))
    return hdf5_files


def create_quad_view_with_cross(image_path):
    """Create a quad view and draw red cross divider lines in the middle"""
    # # Stitch the four views
    # top_row = np.hstack([images[0], images[1]])
    # bottom_row = np.hstack([images[2], images[3]])
    # quad_view = np.vstack([top_row, bottom_row])
    image = cv2.imread(image_path)
    # Get the size of a single image
    h, w = 480, 480

    # Draw the red cross divider lines
    line_width = 3
    color = (0, 0, 255)  # red

    # Vertical line (from top to bottom)
    cv2.line(image, (w, 0), (w, 2 * h), color, line_width)
    # Horizontal line (from left to right)
    cv2.line(image, (0, h), (2 * w, h), color, line_width)

    return image


def generate_trajectory(args, index, logger):
    env = load_env(args.task_name, robot=args.robot, eval=args.eval_unseen)
    # --- [Key change 1]: Initialize the class instance ---
    # Pass in the history_maxlen obtained from argparse
    handler = VLAMessageHandler(history_maxlen=args.history_maxlen, img_size=(480, 480))

    init_ee_pos, init_ee_quat = env.robot.get_end_effector_pos(env.physics), env.robot.get_end_effector_quat(
        env.physics)
    env.reset()

    # Get the basic task info
    instruction = env.task.get_instruction()
    target_entity = env.task.config_manager.target_entity
    target_container = env.task.config_manager.target_container
    if env.task.config_manager.target_bbox:
        target_object_info = env.task.config_manager.target_object_info
    if env.task.config_manager.seen_container and len(env.task.config_manager.seen_container) == 1:
        target_container_name = env.task.config_manager.seen_container[0]

    # Directory preparation
    task_dir = os.path.join(args.save_dir, args.task_name)
    time_img = time.strftime("%Y-%m-%d_%H.%M.%S", time.localtime())
    task_dir = os.path.join(task_dir, f"{time_img}_{target_entity}")
    img_dir = os.path.join(task_dir, f"{target_entity}_img")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(task_dir, exist_ok=True)

    # Skill mapping table
    allowed_skills = {
        'pick': lambda **kwargs: partial(SkillLib.pick, env=env, **kwargs),
        'place': lambda **kwargs: partial(SkillLib.place, env=env, **kwargs),
        'lift': lambda **kwargs: partial(SkillLib.lift, env=env, **kwargs),
        'pull': lambda **kwargs: partial(SkillLib.pull, env=env, **kwargs),
        'push': lambda **kwargs: partial(SkillLib.push, env=env, **kwargs),
        'close_gripper': lambda **kwargs: partial(SkillLib.close_gripper, env=env, **kwargs),
        'open_gripper': lambda **kwargs: partial(SkillLib.open_gripper, env=env, **kwargs),
        'reset': lambda **kwargs: partial(SkillLib.observe, env=env, target_pos=init_ee_pos, target_quat=init_ee_quat,
                                          **kwargs),
        'open_door': lambda **kwargs: partial(SkillLib.open_door, env=env, target_container_name=target_container_name,
                                              **kwargs),
        'close_door': lambda **kwargs: partial(SkillLib.close_door, env=env,
                                               target_container_name=target_container_name, **kwargs),
        'moveforward': lambda **kwargs: partial(SkillLib.moveforward, env=env, target_object_info=target_object_info,
                                                **kwargs),
        'rotate_right': lambda **kwargs: partial(SkillLib.rotate_right, env=env, **kwargs),
        'rotate_left': lambda **kwargs: partial(SkillLib.rotate_left, env=env, **kwargs),
        'end': lambda **kwargs: partial(SkillLib.end, env=env, wait_time=20, **kwargs)
    }

    observations, waypoints = [], []
    last_action = None
    meta_data = []
    n = args.max_skills_num

    while n > 0:
        i = args.max_skills_num - n

        # Get the latest observation
        rgb_data = env.get_observation()
        # Cameras 0-2: fixed camera views, corresponding to right, left, forward respectively; cameras 3, 4 correspond to wrist, hand
        vlm_img_input = [cv2.cvtColor(rgb_data["rgb"][i], cv2.COLOR_BGR2RGB) for i in range(6)]

        #bool flag deciding whether to replace the head camera / static camera
        is_position = env.task.navig_condition
        # --- [Key change 2]: Call the handler to request the VLM ---
        # Note: this automatically maintains the internal vlm_response history
        result = handler.request_vlm(
            vlm_url=args.vlm_url,
            img=vlm_img_input,
            n=i,
            instruction=instruction,
            action=last_action,
            down_to_head=is_position
        )

        if not result:
            logger.error('VLM 响应失败且重试耗尽。')
            break

        bbox, action, target_object, content, pts = result
        last_action = action  # update the action for the next round's feedback

        # Draw and save the BBox selected by the VLM (visual debugging)
        if bbox and action in ["pick", "place", "moveforward", "rotate_right", "rotate_left"]:
            temp_wrist = vlm_img_input[3].copy()
            cv2.polylines(temp_wrist, [pts.astype(int)], isClosed=True, color=(0, 0, 255), thickness=2)
            cv2.imwrite(os.path.join(img_dir, f"vlm_action_{i}_{action}.png"), temp_wrist)

        # Execute the robot skill
        try:
            skill_factory = allowed_skills[action]
            if action == "pick":
                # Keep your original bbox offset correction logic
                pick_bbox = [bbox[0] - 10, bbox[1] - 5, bbox[2] + 5, bbox[3] + 5]
                obs, waypoint, stage_success, task_success, viz = skill_factory(target_entity_name=target_entity,
                                                                                bbox=pick_bbox)()
                # Save the GraspNet visualization
                cv2.imwrite(os.path.join(img_dir, f"grasp_viz_{i}.png"), cv2.cvtColor(viz, cv2.COLOR_BGR2RGB))
            elif action == "place":
                obs, waypoint, stage_success, task_success = skill_factory(target_container_name=target_container,
                                                                           bbox=bbox)()
            else:
                obs, waypoint, stage_success, task_success = skill_factory()()
        except Exception as e:
            logger.error(f'技能执行失败: {e}')
            break

        # Record Meta
        meta_data.append({
            "step": i + 1,
            "action": action,
            "target": target_object,
            "bbox": bbox,
            "response": content
        })

        observations.extend(obs)
        waypoints.extend(waypoint)

        if action == "end" or (args.early_stop and not stage_success):
            break
        n -= 1

    # Save JSON and video
    with open(os.path.join(task_dir, "meta.json"), "w") as f:
        json.dump(meta_data, f, indent=4)

    if args.record_video:
        frames = []
        for o in observations:
            #left、top ; wrist、head
            frame = np.vstack([np.hstack([o["rgb"][5],o["rgb"][2]]), np.hstack([o["rgb"][3], o["rgb"][4]])])
            # print("frame:",frame)

            text = f"{instruction}"
            cv2.putText(
                frame, text, (10, 20),  # position (x, y)
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,  # font and size
                (255, 255, 255), 2,  # color and line width (white)
                cv2.LINE_AA
            )

            frames.append(frame)
        if not os.path.exists(task_dir):
            os.makedirs(task_dir)

        timename = time.strftime("%H.%M.%S", time.localtime())

        mediapy.write_video(os.path.join(task_dir, f"{target_entity}_{task_success}_{timename}.mp4"),
                            frames, fps=10)
        print("video_dir:", task_dir)

    if not task_success:
        logger.warning("Task failed, skip saving data")
        return
    else:
        logger.info("Task success, saving data")

        # timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # data_to_save = process_observations(observations)
        # #data_to_save[keys]: dict_keys(['q_state', 'q_velocity', 'q_acceleration', 'rgb', 'depth', 'segmentation', 'robot_mask', 'instrinsic', 'extrinsic', 'masked_point_cloud', 'point_cloud_points', 'point_cloud_colors', 'ee_state', 'grasped_obj_name', 'trajectory'])
        #
        #
        # print("data_to_save[keys]:",data_to_save.keys())
        # robot_position = env.robot.robot_config["position"]
        # robot_frame_waypoints = [np.array(waypoint) - np.concatenate([robot_position, np.zeros(5)]) for waypoint in waypoints]
        # data_to_save["trajectory"] = robot_frame_waypoints
        # data_to_save["entities"] = meta_info["entities"]
        # print("meta_info[entities]:",meta_info["entities"])
        # data_to_save["target_entity"] = meta_info["target_entity"]
        # print("meta_info[target_entity]:",meta_info["target_entity"])
        # data_to_save["episode_config"] = json.dumps(episode_config)
        # print("meta_info[episode_config]:",episode_config)
        # data_to_save["instruction"] =meta_info["instruction"]
        # save_single_data(data_to_save,
        #                  save_dir=task_dir,
        #                  filename=f"data_{index}_{target_entity}_{timename}.hdf5",
        #                  )
    env.close()

    env.close()


if __name__ == "__main__":
    args = get_args()
    logger = get_logger()
    for i in tqdm(range(args.n_sample)):
        i += args.start_id
        try:
            h5_files = get_all_hdf5_files(os.path.join(args.save_dir, args.task_name))
            if len(h5_files) >= args.max_episode:
                logger.info(f"Task {args.task_name} has reached the maximum episode number, skip")
                break
            generate_trajectory(args, i, logger)
        except Exception as e:
            err = traceback.TracebackException.from_exception(e)
            print("".join(err.format()))
            continue