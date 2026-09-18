"""
Skill Library for data generation.
"""
import math
import open3d as o3d
import requests
import re
import os
import json
import time
import random
import base64

import numpy as np
import cv2
import random
import time

from sympy.utilities.iterables import rotate_right

# graspnet
import torch
from scipy.spatial.transform import Rotation as R

from RMMBench.utils.utils import get_graspnet_pcd, run_graspnet_on_pcd, filter_grasps_collision, filter_grasps_position, \
    select_topk_grasps, graspnet_rot_correction, graspnet_grasp_to_pose, prepare_grasp_graspnet
from RMMBench.utils.utils import vis_grasps, vis_gripper

from RMMBench.utils.utils import degrees_to_radians_builtin, flatten_list, find_keypoint_and_prepare_grasp, \
    extract_base_name, distance, quaternion_to_euler, euler_to_quaternion, quaternion_from_axis_angle, \
    quaternion_multiply, get_placement_top_down_view, matrix_to_quaternion, compute_lmax_hanan_grid
from RMMBench.algorithms.motion_planning.rrt import rrt_motion_planning
from RMMBench.algorithms.utils import interpolate_path, qauternion_slerp, interpolate_pos, interpolate_radians


# =============================================================================
# Safe observation switch (IndexError protection)
# -----------------------------------------------------------------------------
# SAFE_OBSERVATION_MODE = True  (default)
#   Occasional IndexError in env.get_observation() (e.g. dm_control segmentation
#   rendering out-of-bounds) is caught and retried; if all retries fail, the
#   previous observation is returned and the program does not crash.
#
# SAFE_OBSERVATION_MODE = False
#   Disables the protection; IndexError is raised normally, which makes it
#   easier to locate problems while debugging.
#
# Can also be toggled at runtime:
#   import RMMBench.utils.codelab_skill_lib as sl
#   sl.SAFE_OBSERVATION_MODE = False   # disable protection
#   sl.SAFE_OBSERVATION_MODE = True    # re-enable
# =============================================================================
SAFE_OBSERVATION_MODE: bool = True

# Maximum number of retries (only takes effect when SAFE_OBSERVATION_MODE=True)
_SAFE_OBS_MAX_RETRIES: int = 3


def safe_get_observation(env, _last_obs=None):
    """
    Safe wrapper around env.get_observation().

    When SAFE_OBSERVATION_MODE=True:
      - Catches IndexError (occasional errors such as dm_control segmentation rendering out-of-bounds)
      - Retries up to _SAFE_OBS_MAX_RETRIES times
      - If all retries fail, returns the previous observation if available; otherwise returns None and prints a warning
    When SAFE_OBSERVATION_MODE=False:
      - Calls env.get_observation() directly; exceptions propagate (handy for debugging)

    Parameters
    ----------
    env       : environment object
    _last_obs : previous observation (optional), used as the fallback return value when all retries fail
    """
    if not SAFE_OBSERVATION_MODE:
        return env.get_observation()

    last_exc = None
    for attempt in range(1, _SAFE_OBS_MAX_RETRIES + 1):
        try:
            return env.get_observation()
        except IndexError as e:
            last_exc = e
            print(
                f"[safe_get_observation] IndexError (attempt {attempt}/{_SAFE_OBS_MAX_RETRIES}): {e}"
            )
    # All retries failed
    print(
        f"[safe_get_observation] WARNING: 所有重试均失败，最后异常: {last_exc}。"
        f"{'返回上一帧观测作为兜底。' if _last_obs is not None else '无上一帧观测，返回 None。'}"
    )
    return _last_obs


PRIOR_EULERS = [[np.pi, 0, -np.pi / 2],  # face down, horizontal
                [np.pi, 0, 0],  # face down, vertical
                [-np.pi / 2, -np.pi / 2, 0],  # face forward, horizontal
                [-np.pi / 2, 0, 0],  # face forward, vertical
                ]


class SkillLib:

    @staticmethod
    def look_right(env, gripper_state=None, yaw=30, target_velocity=0.157, tolerance=0.01):
        yaw = -degrees_to_radians_builtin(yaw)

        init_qpos = env.task.robot.default_qpos
        link_init_pos = init_qpos[:7]
        base_current_pos = env.robot.get_mobilebase_distance(env.physics)
        base_target_pos = base_current_pos + [0, yaw, 0, 0, 0]  # forward:[01000];
        arm_current_angles = env.robot.get_qpos(env.physics)[:7]  # keep only the arm joint angles
        gripper_closed = env.robot.get_ee_open_state(env.physics)
        if gripper_state is None:
            if gripper_closed:
                gripper_state = np.zeros(2)
            else:
                gripper_state = np.ones(2) * 0.04
        current_action = np.concatenate([link_init_pos, gripper_state, base_current_pos])
        observations = [env.get_observation()]
        waypoints = []
        task_success = False
        base_radians = [base_current_pos[1], base_target_pos[1]]

        # base_radians = [0, -1]
        interplate_base_path = interpolate_radians(base_radians, target_velocity)
        inerplate_path = []
        for _ in range(len(interplate_base_path)):
            point = np.concatenate([current_action[:10], [interplate_base_path[_]], current_action[11:]])
            timestep = env.step(point)
            if timestep.last():
                task_success = True
                break
            obs = safe_get_observation(env, observations[-1])
            observations.append(obs)
            waypoints.append(point)
        # curr = env.robot.get_mobilebase_distance(env.physics)
        # print("curr_1:",curr[1])
        if distance(interplate_base_path[-1], env.robot.get_mobilebase_distance(env.physics)[-1]) < tolerance:
            stage_success = True
        stage_success = True
        return observations, waypoints, stage_success, task_success

    @staticmethod
    def look_left(env, gripper_state=None, yaw=30, target_velocity=0.157, tolerance=0.01):
        return SkillLib.look_right(env,
                                   gripper_state=gripper_state,
                                   yaw=-yaw,
                                   target_velocity=target_velocity,
                                   tolerance=tolerance)

    @staticmethod
    def look_forward(env, gripper_state=None, yaw=0, target_velocity=0.157, tolerance=0.01):
        init_qpos = env.task.robot.default_qpos
        link_init_pos = init_qpos[:7]
        base_current_pos = env.robot.get_mobilebase_distance(env.physics)

        base_target_pos = base_current_pos + [0, 0, 0, 0, 0]  # forward:[01000];
        base_target_pos[1] = 0

        arm_current_angles = env.robot.get_qpos(env.physics)[:7]  # keep only the arm joint angles
        gripper_closed = env.robot.get_ee_open_state(env.physics)
        if gripper_state is None:
            if gripper_closed:
                gripper_state = np.zeros(2)
            else:
                gripper_state = np.ones(2) * 0.04
        current_action = np.concatenate([link_init_pos, gripper_state, base_current_pos])
        observations = [env.get_observation()]
        waypoints = []
        task_success = False
        base_radians = [base_current_pos[1], base_target_pos[1]]

        # base_radians = [0, -1]
        interplate_base_path = interpolate_radians(base_radians, target_velocity)
        inerplate_path = []
        for _ in range(len(interplate_base_path)):
            point = np.concatenate([current_action[:10], [interplate_base_path[_]], current_action[11:]])
            timestep = env.step(point)
            if timestep.last():
                task_success = True
                break
            obs = safe_get_observation(env, observations[-1])
            observations.append(obs)
            waypoints.append(point)
        curr = env.robot.get_mobilebase_distance(env.physics)
        print("curr_1:", curr[1])
        if distance(interplate_base_path[-1], env.robot.get_mobilebase_distance(env.physics)[-1]) < tolerance:
            stage_success = True
        stage_success = True
        return observations, waypoints, stage_success, task_success

    @staticmethod
    def moveforward(env, gripper_state=None, forward=0, side=0, yaw=0, target_velocity=0.1, tolerance=0.01,
                    target_object_info=None,
                    disturbance_points=None,  # disturbance point set [[x1,y1], [x2,y2], ...]
                    L_th=1.2,  # distance threshold
                    l_default=0.4,  # default travel distance (boundary case)
                    robot_radius=0.3):  # robot radius

        """
        [0, forward, side, yaw]
        Hanan-grid-based movement distance decision logic.
        Every moveforward the robot walks on the Hanan grid, moving forward
        along the current heading (one of the four directions).
        """
        target_object_info = env.task.config_manager.target_object_info
        current_robot_info = env.robot.get_link_base_info(env.physics)
        init_robot_info = env.task.init_robot_info

        curr_yaw = current_robot_info["euler"][2]
        init_yaw = init_robot_info["euler"][2]

        # Compute relative_yaw to determine the current heading (one of the four directions)
        relative_yaw = (curr_yaw - init_yaw + math.pi) % (2 * math.pi) - math.pi

        # ========== Hanan-grid movement distance decision logic ==========
        # Build the stay-point set = target points + disturbance points (2D coordinates only)
        stay_points = []
        if target_object_info is not None:
            if isinstance(target_object_info, list):
                for info in target_object_info:
                    stay_points.append([info["position"][0], info["position"][1]])
            else:
                stay_points.append([target_object_info["position"][0], target_object_info["position"][1]])

        if disturbance_points is not None:
            for dp in disturbance_points:
                if len(dp) >= 2:
                    stay_points.append([dp[0], dp[1]])

        # Get the obstacle bbox list (from the RoboCasa scene)
        obstacle_bboxes = []
        boundary = None
        if hasattr(env.task, 'get_robocasa_scene_class') and env.task.get_robocasa_scene_class is not None:
            navigation_bbox_info = env.task.get_robocasa_scene_class.get_obstacles_bbox(env.physics)
            obstacle_bboxes = navigation_bbox_info.get("bbox_list", [])
            boundary = navigation_bbox_info.get("boundary", [])
        # Determine the movement direction from relative_yaw (one of the four directions)
        # Case 1 (0°): +X direction (forward positive)
        # Case 4 (180°): -X direction (forward negative)
        # Case 2 (90°): +Y direction (side negative, left)
        # Case 3 (-90°): -Y direction (side positive, right)
        move_direction = None
        if abs(relative_yaw) < 0.1:  # +X
            move_direction = '+x'
        elif abs(abs(relative_yaw) - 3.14) < 0.1:  # -X
            move_direction = '-x'
        elif abs(relative_yaw - 1.57) < 0.1:  # +Y
            move_direction = '+y'
        elif abs(relative_yaw + 1.57) < 0.1:  # -Y
            move_direction = '-y'

        if move_direction is not None:
            # Compute Lmax on the Hanan grid
            current_pos = [current_robot_info["position"][0], current_robot_info["position"][1]]

            # Check whether the robot is currently within a stay point (if so, remove it from stay_points)
            stay_points_filtered = []
            for sp in stay_points:
                manhattan_dist = abs(current_pos[0] - sp[0]) + abs(current_pos[1] - sp[1])
                if manhattan_dist > robot_radius * 2:  # Outside the current stay-point range; ignore this point
                    stay_points_filtered.append(sp)

            # Determine the world-frame direction directly from curr_yaw (0 -> +x, 3.14/-3.14 -> -x, 1.57 -> +y, -1.57 -> -y)
            world_move_direction = None
            if abs(curr_yaw) < 0.1 or abs(curr_yaw - 2 * math.pi) < 0.1:  # 0 or 2π, i.e. +X direction
                world_move_direction = '+x'
            elif abs(abs(curr_yaw) - math.pi) < 0.1:  # ±π, i.e. -X direction
                world_move_direction = '-x'
            elif abs(curr_yaw - math.pi / 2) < 0.1:  # π/2, i.e. +Y direction
                world_move_direction = '+y'
            elif abs(curr_yaw + math.pi / 2) < 0.1:  # -π/2, i.e. -Y direction
                world_move_direction = '-y'

            # ===== DEBUG: print direction decisions and point-set info =====
            print(
                f"[moveforward DEBUG] curr_yaw={curr_yaw:.4f}, init_yaw={init_yaw:.4f}, relative_yaw={relative_yaw:.4f}")
            print(f"[moveforward DEBUG] move_direction={move_direction}, world_move_direction={world_move_direction}")
            print(f"[moveforward DEBUG] current_pos={current_pos}")
            print(f"[moveforward DEBUG] stay_points={stay_points}")
            print(f"[moveforward DEBUG] stay_points_filtered={stay_points_filtered}")

            # Compute Lmax (along the current direction): distance to the nearest stay point or obstacle ahead
            if world_move_direction is not None:
                Lmax = compute_lmax_hanan_grid(current_pos, world_move_direction, obstacle_bboxes,
                                               stay_points_filtered, robot_radius, boundary)
            else:
                Lmax = float('inf')
                print(f"[moveforward DEBUG] 无法确定世界坐标方向，使用默认值inf")

            print(f"[moveforward DEBUG] Lmax={Lmax}")

            # Decide the actual travel distance based on L_th
            if Lmax == float('inf') or Lmax > 100:  # Boundary case: no points and no obstacles ahead
                actual_dist = l_default
                print(f"[moveforward DEBUG] 使用l_default={l_default} (前方无点无障碍)")
            elif Lmax < L_th:
                actual_dist = max(0, Lmax)  # Walk up to the stay point or obstacle
                print(f"[moveforward DEBUG] Lmax<L_th, 行走距离={actual_dist}")
            else:
                actual_dist = Lmax / 2  # Stop halfway
                print(f"[moveforward DEBUG] Lmax>=L_th, 中途停留, 行走距离={actual_dist}")

            # Set forward or side directly from move_direction (pure Hanan-grid movement, ignoring the passed-in forward/side)
            if move_direction == '+x':
                forward = actual_dist
            elif move_direction == '-x':
                forward = -actual_dist
            elif move_direction == '+y':
                side = -actual_dist  # +Y direction corresponds to a negative side value (left)
            elif move_direction == '-y':
                side = actual_dist  # -Y direction corresponds to a positive side value (right)

        # Current arm orientation
        current_euler = current_robot_info["euler"]
        # Current rotation - 1.57
        # env.robot.get_mobilebase_distance(env.physics)
        print("link_base_pos:", env.robot.get_base_position(env.physics))
        print("get_base_quat:", env.robot.get_link_base_quat(env.physics))

        # env.robot.set_base_orientation(ori=[0, 0, 1.57])
        # Base [0, forward, side, yaw]
        base_current_pos = env.robot.get_mobilebase_distance_continue_look(env.physics)  # current value
        print("base_current_pos:", base_current_pos)
        print("current_euler[2]:", current_euler[2])

        print("forward:", forward)  # forward must be an incremental value, added on top of the current value
        print("side:", side)
        move_distance = round(max(abs(forward), abs(side)), 2)

        # move_distance = max(abs(forward),abs(side))
        # print("move_distance:", move_distance)
        # env.close()
        # exit()

        base_current_pos_reset_look = env.robot.get_mobilebase_distance(env.physics)
        base_target_pos = list(np.array(base_current_pos_reset_look) + np.array([0, 0, forward, side, yaw]))
        print("base_target_pos:", base_target_pos)

        arm_current_angles = env.robot.get_qpos(env.physics)[:7]  # keep only the arm joint angles

        gripper_closed = env.robot.get_ee_open_state(env.physics)
        if gripper_state is None:
            if gripper_closed:
                gripper_state = np.zeros(2)
            else:
                gripper_state = np.ones(2) * 0.04

        current_action = np.concatenate([arm_current_angles, gripper_state, base_current_pos])

        observations = [env.get_observation()]
        waypoints = []
        task_success = False

        base_path = [base_current_pos[1:4], np.array(base_target_pos[1:4])]
        print("base_path:", base_path)

        interplate_base_path = interpolate_pos(base_path, target_velocity)

        inerplate_path = []
        for _ in range(len(interplate_base_path)):

            # point = np.concatenate([current_action[:10],[current_action[10]], interplate_base_path[_], [current_action[-1]]])
            point = np.concatenate([current_action[:10], interplate_base_path[_], [current_action[-1]]])
            print("point:", point)
            timestep = env.step(point)
            if timestep.last():
                task_success = True
                break
            obs = safe_get_observation(env, observations[-1])
            observations.append(obs)
            waypoints.append(point)

        # waypoints.append(waypoint)
        # if distance(interplate_base_path[-1], env.robot.get_mobilebase_distance(env.physics)[1:3]) < tolerance:
        #     stage_success = True
        stage_success = True
        print("link_base_pos_end:", env.robot.get_base_position(env.physics))
        return observations, waypoints, stage_success, task_success, move_distance

    @staticmethod
    def rotate_right(env, gripper_state=None, foward=0, side=0, yaw=-1.57, target_velocity=0.157, tolerance=0.01,
                     rotation_center_offset=0.208):  # the rotation center is 0.208 m in front of the base
        # print("get_base_quat:",env.robot.get_link_base_quat(env.physics))
        # Current value
        init_base_position = env.robot.get_link_base_info(env.physics)
        init_qpos = env.task.robot.default_qpos
        link_init_pos = init_qpos[:7]

        base_current_pos = env.robot.get_mobilebase_distance_continue_look(env.physics)
        current_yaw = base_current_pos[-1]
        # print("base_current_pos:",base_current_pos)

        # ==== Rotation-center compensation computation ====
        # Since the rotation center lies rotation_center_offset in front of the base,
        # the base center traces an arc while rotating.
        # A compensation must be computed so the base center stays in place;
        # the base center offset relative to the rotation center after rotation needs to be compensated back.

        target_yaw = current_yaw + yaw

        # Compute the base center's position change relative to the rotation center before and after the rotation
        # The rotation center is rotation_center_offset in front of the base (along the current yaw direction)
        # The base center relative to the rotation center is (0, -rotation_center_offset) (in the rotation-center frame)

        # Base center position in the world frame before rotation (relative to the rotation center)
        cos_curr = math.cos(current_yaw)
        sin_curr = math.sin(current_yaw)
        # Offset of the rotation center in the world frame (used for the computation)
        center_x_curr = rotation_center_offset * cos_curr
        center_y_curr = rotation_center_offset * sin_curr

        # After rotation, the base center's position relative to the rotation center
        # (in the rotation-center frame the base center is always behind)
        cos_target = math.cos(target_yaw)
        sin_target = math.sin(target_yaw)
        center_x_target = rotation_center_offset * cos_target
        center_y_target = rotation_center_offset * sin_target

        # Compensation = the offset of the rotation-center position (keeps the base center in place)
        # To keep the base center stationary, the rotation center must orbit the base center,
        # i.e. forward/side need to compensate the rotation-center offset
        compensate_forward = center_x_target - center_x_curr
        compensate_side = center_y_target - center_y_curr

        # Note: the base frame is [height, forward, side, yaw]
        # forward corresponds to X, side corresponds to Y (but the positive side direction needs verification)
        base_current_pos_reset_look = env.robot.get_mobilebase_distance(env.physics)
        base_target_pos = base_current_pos_reset_look + [0, 0, compensate_forward + foward, compensate_side + side, yaw]
        # print("base_target_pos:", base_target_pos)
        # print("ee original angle:", env.robot.get_end_effector_quat(env.physics))

        arm_current_angles = env.robot.get_qpos(env.physics)[:7]  # keep only the arm joint angles

        gripper_closed = env.robot.get_ee_open_state(env.physics)
        if gripper_state is None:
            if gripper_closed:
                gripper_state = np.zeros(2)
            else:
                gripper_state = np.ones(2) * 0.04

        # mobile_base = [0, 0, 1, 0]  # h-f-s-y
        # current_action = np.concatenate([arm_current_angles, gripper_state, base_current_pos])
        current_action = np.concatenate([link_init_pos, gripper_state, base_current_pos])

        observations = [env.get_observation()]
        waypoints = []
        task_success = False

        base_radians = [base_current_pos[-1], base_target_pos[-1]]
        interplate_base_path = interpolate_radians(base_radians, target_velocity)
        # print("interplate_base_path:", interplate_base_path)
        # print("len(interplate_base_path):", len(interplate_base_path))

        # Generate the compensated trajectory (interpolated)
        n_steps = len(interplate_base_path)
        # interpolate_look = [[base_current_pos[1]], [base_target_pos[1]]]
        # Resample
        start_val = base_current_pos[1]
        end_val = base_target_pos[1]
        # np.linspace automatically includes start_val and end_val, uniformly generating n_steps points
        resampled_path = np.linspace(start_val, end_val, num=n_steps).tolist()

        for i in range(n_steps):
            # Compute the compensation for the current step
            t = i / (n_steps - 1) if n_steps > 1 else 1.0
            interp_yaw = current_yaw + yaw * t
            cos_interp = math.cos(interp_yaw)
            sin_interp = math.sin(interp_yaw)
            center_x_interp = rotation_center_offset * cos_interp
            center_y_interp = rotation_center_offset * sin_interp

            # Compensation: make the rotation center orbit the base center so the base center stays still
            # Goal: base center position = initial position + compensation
            # Rotation center position = base center + offset * (cos(yaw), sin(yaw))
            # To keep the base center stationary, the rotation center must move in the opposite direction
            step_comp_forward = -(center_x_interp - center_x_curr)
            step_comp_side = -(center_y_interp - center_y_curr)

            # Build the action [arm_qpos(7), gripper(2), height, forward, side, yaw]
            # Note: the side direction may be inverted relative to world Y; the sign needs to be tried
            # base_action = [base_current_pos[0],  # height unchanged
            #                base_current_pos[1] + step_comp_forward + foward * t,  # forward + compensation
            #                base_current_pos[2] - step_comp_side + side * t,  # side - compensation (mind the sign)
            #                interplate_base_path[i]]  # yaw
            # without look reset
            # base_action = [base_current_pos[0],  # height unchanged
            #                base_current_pos[1],
            #                base_current_pos[2] + step_comp_forward + foward * t,  # forward + compensation
            #                base_current_pos[3] - step_comp_side + side * t,  # side - compensation (mind the sign)
            #                interplate_base_path[i]]  # yaw
            # with look reset
            base_action = [base_current_pos[0],  # height unchanged
                           resampled_path[i],
                           base_current_pos[2] + step_comp_forward + foward * t,  # forward + compensation
                           base_current_pos[3] - step_comp_side + side * t,  # side - compensation (mind the sign)
                           interplate_base_path[i]]  # yaw

            point = np.concatenate([current_action[:9], base_action])
            # print("point:", point)

            timestep = env.step(point)
            if timestep.last():
                task_success = True
                break
            obs = safe_get_observation(env, observations[-1])
            observations.append(obs)
            waypoints.append(point)
        # print("current angle:", env.robot.get_mobilebase_distance(env.physics))
        # print("ee current angle:", env.robot.get_end_effector_quat(env.physics))
        # waypoints.append(waypoint)
        if distance(interplate_base_path[-1], env.robot.get_mobilebase_distance(env.physics)[-1]) < tolerance:
            stage_success = True
        stage_success = True
        # print("get_base_quat_end:",env.robot.get_link_base_quat(env.physics))

        return observations, waypoints, stage_success, task_success

    @staticmethod
    def rotate_left(env, gripper_state=None, foward=0, side=0, yaw=1.57, target_velocity=0.157, tolerance=0.01,
                    rotation_center_offset=0.208):
        return SkillLib.rotate_right(
            env=env,
            gripper_state=gripper_state,
            foward=foward,
            side=side,
            yaw=yaw,
            target_velocity=target_velocity,
            tolerance=tolerance,
            rotation_center_offset=rotation_center_offset
        )

    @staticmethod
    def step_trajectory(env,
                        points,
                        quats,
                        gripper_state,
                        max_n_substep=1,
                        tolerance=0.01,
                        end_dpos=None):
        """
        Universal step function for data generation.
        Input:
            env: LM4ManipEnv, for success detection
            points: np.array (n, 3), target positions
            quates: np.array(n, 4), target quaternions
            gripper_state: np.array(2), gripper state
            max_n_step: int, max number of substep for each step
            tolerance: float, tolerance for the qpos error between the target and the current qpos
        Return:
            observations: list of observations
            waypoints: list of waypoints
            stage_success: bool, whether the stage is successful
            task_success: bool, whether the task is successful
        """
        observations = []
        waypoints = []
        stage_success = False
        task_success = False
        for i, (point, quat) in enumerate(zip(points, quats)):

            success, action = env.robot.get_qpos_from_ee_pos(physics=env.physics, pos=point, quat=quat)
            # if not success: # a wrong action beyond the embodied limit
            #     return None, None, False, False
            # if len(action) == 11:
            #     action = np.concatenate([action[:7], gripper_state,action[7:]])

            if len(action) == 7:
                action = np.concatenate([action, gripper_state])
            else:
                action = np.concatenate([action[:7], gripper_state, action[7:]])

            waypoint = np.concatenate([point, quaternion_to_euler(quat), gripper_state])
            if i == len(points) - 1:
                if end_dpos is not None:
                    action = np.concatenate([end_dpos, gripper_state, action[9:]])
                    max_n_substep = 100

            substep_count = 0
            while substep_count < max_n_substep:
                dpos = env.robot.get_qpos(env.physics)[:]
                if np.all(np.abs(dpos[:7] - action[:7]) < 0.01):
                    break
                timestep = env.step(action)
                substep_count += 1

                if timestep.last():
                    task_success = True
                    break

                current_qpos = np.array(env.task.robot.get_qpos(env.physics)).reshape(-1)

                if len(action) == 9:
                    diff = current_qpos - np.array(action[:7])
                    if np.max(diff) < tolerance and np.min(diff) > -tolerance:
                        break

                elif len(action) == 13:
                    target_qpos = np.concatenate([action[:7], action[9:]])
                    diff = current_qpos - np.array(target_qpos)
                    if np.max(diff) < tolerance and np.min(diff) > -tolerance:
                        break
            # print("substep_count:",substep_count)
            if task_success:
                break
            obs = safe_get_observation(env, observations[-1])
            observations.append(obs)
            waypoints.append(waypoint)
        assert len(observations) == len(
            waypoints), f"observations and waypoints should have the same length, {len(observations)} and {len(waypoints)}"
        if distance(points[-1], env.robot.get_end_effector_pos(env.physics)) < tolerance:
            stage_success = True
        return observations, waypoints, stage_success, task_success

    @staticmethod
    def moveto(env,
               target_pos,
               target_quat=None,
               target_velocity=0.05,
               gripper_state=None,
               **kwargs
               ):
        start_pos, start_quat = env.robot.get_end_effector_pos(env.physics), env.robot.get_end_effector_quat(
            env.physics)
        observations = [env.get_observation()]
        waypoints = []
        task_success = False
        gripper_closed = env.robot.get_ee_open_state(env.physics)
        if gripper_state is None:
            if gripper_closed:
                gripper_state = np.zeros(2)
            else:
                gripper_state = np.ones(2) * 0.04
        # env_pcd = observations[0]["masked_point_cloud"]
        # obstacle_pcd = np.asarray(env_pcd.points)

        obstacle_pcd = np.asarray(env.get_obstacle_pcd().points)

        if target_quat is None:
            target_quat = start_quat
        target_pos = target_pos
        motion_planning_path = rrt_motion_planning(tuple(start_pos),
                                                   tuple(target_pos),
                                                   obstacle_pcd)
        if motion_planning_path is None:
            motion_planning_path = [start_pos, target_pos]
        quats_in_path = []
        for t in np.linspace(0, 1, len(motion_planning_path), endpoint=False):
            quats_in_path.append(qauternion_slerp(start_quat, target_quat, t))
        interplate_path, interplate_quat = interpolate_path(np.array(motion_planning_path),
                                                            np.array(quats_in_path),
                                                            target_velocity)
        new_obs, new_waypoints, stage_success, task_success = SkillLib.step_trajectory(env,
                                                                                       interplate_path,
                                                                                       interplate_quat,
                                                                                       gripper_state,
                                                                                       **kwargs)
        # if new_obs is None:
        # return None, None, False, False
        observations.extend(new_obs)
        waypoints.extend(new_waypoints)
        observations.pop(-1)
        assert len(observations) == len(
            waypoints), f"observations and waypoints should have the same length, {len(observations)} and {len(waypoints)}"
        return observations, waypoints, stage_success, task_success

    @staticmethod
    def pick(env,
             target_entity_name=None,
             topk_grasps=4,
             prepare_quat=None,
             motion_planning_kwargs=dict(),
             voxel_size=0.01,
             approach_dist=0.05,
             collision_thresh=0.01,
             bbox=None,
             select_grasp=None,
             task_name=None,
             vlm_handler=None,
             grasp_seed=42,  # new parameter: random seed, default 42
             prior_eulers=PRIOR_EULERS,
             specific_keypoint=None,
             **kwargs):
        """
        Use GraspNet to detect grasps from the env masked point cloud, then try to execute them.
        - net: pretrained GraspNet model loaded by get_net()
        - env: LM4ManipDMEnv
        Returns same signature as original pick: observations, waypoints, stage_success, task_success

        Current logic:
        1. Use graspnet to generate grasps from wrist camera
        2. Filter grasps by camera z-coordinates, keep the grasps that are more likely to be on the object
        3. Rank grasps by camera y-coordinates (grasp point on object)
        4. Use pre-defined quaternion along with graspnet translation for grasp execution
        """
        print("pick_bbox:", bbox)

        # Fix the random seed to ensure consistent GraspNet output
        if grasp_seed is not None:
            np.random.seed(grasp_seed)
            random.seed(grasp_seed)
            torch.manual_seed(grasp_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(grasp_seed)

        # SkillLib.open_gripper(env)
        start_pos = np.array(env.robot.get_end_effector_pos(env.physics))
        start_quat = np.array(env.robot.get_end_effector_quat(env.physics))
        distance = env.robot.get_ee_open_distance(env.physics)
        print("gripper_open_distance1", distance)
        '''Add bounding box HERE!'''
        # test_eval_pick.json:bbox:[242, 174, 298, 222]
        # bbox = [242, 174, 298, 222]
        bounding_box = bbox  # for select_carrot wrist camera
        end_points, cloud, cam_intr, cam_extr, is_cloud = get_graspnet_pcd(env, bbox=bounding_box, cam_id=3)
        if is_cloud is False:
            print("The point cloud is empty")
            return [env.get_observation()], [], False, False, None, None, False

        gg = run_graspnet_on_pcd(end_points, cloud, bbox=bounding_box, intr=cam_intr)
        print("len(gg):", len(gg))

        # Check whether GraspNet returned an empty result
        if len(gg) == 0:
            return [env.get_observation()], [], False, False, None, None, True

        gg_col = filter_grasps_collision(gg, np.array(cloud.points), voxel_size=voxel_size, approach_dist=approach_dist,
                                         collision_thresh=collision_thresh)

        # Check whether any grasp candidates remain after collision filtering
        if len(gg_col) == 0:
            print("Warning: All grasps filtered out by collision detection")
            return [env.get_observation()], [], False, False, None, None, True

        gg_pos = None
        y_p1 = 60
        y_p2 = 95
        z_p = 10
        x_p = [30, 70]
        # y_p1 = 35
        # y_p2 = 65
        # z_p = 10

        # Set the physical limits
        LIMIT_MIN = 0
        LIMIT_MAX = 100
        max_expand_iter = 50  # maximum number of relaxation iterations, to prevent an infinite loop

        expand_iter = 0
        while gg_pos == None:
            # Run the filtering function
            current_grasps = filter_grasps_position(gg_col, x_p=x_p, y_p1=y_p1, y_p2=y_p2, z_p=z_p)

            if len(current_grasps) < topk_grasps:
                y_p1 = max(LIMIT_MIN, y_p1 - 1)
                z_p = min(LIMIT_MAX, z_p + 1)
                new_x0 = max(LIMIT_MIN, x_p[0] - 1)
                new_x1 = min(LIMIT_MAX, x_p[1] + 1)
                x_p = [new_x0, new_x1]
                expand_iter += 1

                # Check whether relaxation has hit the limit and is still not enough; fall back directly
                is_at_limit = (y_p1 == LIMIT_MIN and z_p == LIMIT_MAX and
                               x_p[0] == LIMIT_MIN and x_p[1] == LIMIT_MAX)
                if expand_iter >= max_expand_iter or is_at_limit:
                    if len(current_grasps) > 0:
                        gg_pos = current_grasps
                        print(
                            f"Warning: Using relaxed grasps after {expand_iter} iterations, got {len(current_grasps)} grasps")
                    elif len(gg_col) > 0:
                        # Nothing left at all; use all collision-filtered results directly
                        gg_pos = gg_col
                        print(f"Warning: Fallback to collision-only grasps, got {len(gg_col)} grasps")
                    else:
                        # No grasp candidates at all
                        return [env.get_observation()], [], False, False, None, None, True
                    break
            else:
                gg_pos = filter_grasps_position(gg_col, x_p=x_p, y_p1=y_p1, y_p2=y_p2, z_p=z_p)
        print(f"y_p1:{y_p1},z_p:{z_p},x_p:{x_p}")
        candidate_grasps = select_topk_grasps(gg_pos, topk=3)
        # vis_grasps(candidate_grasps, cloud)

        # rgb_img = env.render(camera_id=3, height=480, width=480)
        # viz = vis_gripper(rgb_img, candidate_grasps[0].translation, R, cam_intr, gripper_w=candidate_grasps[0].width, gripper_d=candidate_grasps[0].depth)
        # plt.imsave(f"graspnet_visual.jpg", viz)

        observations = [env.get_observation()]  # start observation
        waypoints = []

        for i, g in enumerate(candidate_grasps):
            # try:
            grasp_rotation = graspnet_rot_correction(g.rotation_matrix)
            rgb_img = env.render(camera_id=3, height=480, width=480)

            # debug_dir = "/home/sankuai/work/RMMBench/B_test_data/test_img"
            # cv2.imwrite(os.path.join(debug_dir, f"test1.png"),rgb_img)

            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
            # cv2.imwrite(os.path.join(debug_dir, f"test2.png"),rgb_img)

            viz_1 = vis_gripper(rgb_img, g.translation, grasp_rotation, cam_intr, g.width, g.depth)
            viz_list = [viz_1]
            viz_dict = [{"translation": g.translation, "rotation": grasp_rotation}]

            # 11111111111111111 check the grasp translation
            check_contain = True
            for name in ["1"]:
                if name not in target_entity_name:
                    target_entity = env.task.entities[target_entity_name]
                    key_pos_fix, _, _ = find_keypoint_and_prepare_grasp(env, target_entity, prior_eulers,
                                                                        specific_keypoint_id=specific_keypoint,
                                                                        move_vector=prepare_quat)

                    # key_pos1, key_quat1 = graspnet_grasp_to_pose(g.translation, grasp_rotation, cam_extr)
                else:
                    check_contain = False
                    key_pos_fix = g.translation
                # plt.imsave(f"graspnet_visual.jpg", viz)

            # Safely get the second grasp point (if it exists)
            if len(candidate_grasps) > 1:
                g1 = candidate_grasps[1]
                grasp_rotation_g1 = graspnet_rot_correction(g1.rotation_matrix)
                # cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
                viz_2 = vis_gripper(rgb_img, g1.translation, grasp_rotation_g1, cam_intr, g1.width, g1.depth)
                viz_list.append(viz_2)
                viz_dict.append({"translation": g1.translation, "rotation": grasp_rotation_g1})
                key_pos2, key_quat2 = graspnet_grasp_to_pose(g1.translation, grasp_rotation_g1, cam_extr)
                print("key_quat2:", key_quat2)

            # #horizontal grasp
            # Build a vertical-grasp visualization that stays consistent in the robot base frame
            # Requirements: 1. vertical, pointing down (Z toward world -Z); 2. X axis pointing in the robot's forward direction

            # Visualize using the corrected coordinates
            # for name in ["pan"]:
            #     if name not in target_entity_name:
            print("cam_extr:", cam_extr)
            # --- [DEBUG] raw-data checks ---
            R_W_C = cam_extr[:3, :3]  # camera pose rotation (Camera-to-World)
            t_W_C = cam_extr[:3, 3]  # camera position (Camera-to-World)
            # 1. Define the MuJoCo (OpenGL) to OpenCV coordinate conversion
            # MuJoCo: X right, Y up, Z back -> OpenCV: X right, Y down, Z forward
            R_mjt_to_ocv = np.array([
                [1, 0, 0],
                [0, -1, 0],
                [0, 0, -1]
            ])
            # 2. World -> camera transform matrix
            # Logic: first use R_W_C.T to go to the MuJoCo camera frame, then convert to the OpenCV frame
            R_world_to_cam_proper = R_mjt_to_ocv @ R_W_C.T

            # 3. Build the grasp pose in the world frame (this part keeps the original definition)
            base_current_pos = env.robot.get_mobilebase_distance(env.physics)
            base_yaw = base_current_pos[-1]

            z_axis_world = np.array([0, 0, -1])  # vertical, pointing down
            y_axis_world = np.array([-np.cos(base_yaw), -np.sin(base_yaw), 0])  # wide face facing the robot
            y_axis_world /= np.linalg.norm(y_axis_world)
            x_axis_world = np.cross(y_axis_world, z_axis_world)
            x_axis_world /= np.linalg.norm(x_axis_world)

            R_world_gripper = np.column_stack([x_axis_world, y_axis_world, z_axis_world])

            # Rotate -90 degrees around the Z axis (kept from the earlier logic; comment this line out if the pose is off by 90 degrees)
            Rz_neg90 = o3d.geometry.get_rotation_matrix_from_xyz([0, 0, -np.pi / 2])
            R_world_gripper_final = R_world_gripper @ Rz_neg90

            # 4. Convert uniformly to the OpenCV camera space
            r_vertical_in_cam = R_world_to_cam_proper @ R_world_gripper_final
            key_pos_in_cam = R_world_to_cam_proper @ (key_pos_fix - t_W_C)
            # prepare_key_pos_cam = R_world_to_cam_proper @ (prepare_key_pos - t_W_C)
            viz_3 = vis_gripper(rgb_img, key_pos_in_cam, r_vertical_in_cam, cam_intr, g.width, g.depth)
            viz_list.append(viz_3)
            viz_dict.append({"translation": key_pos_in_cam, "rotation": r_vertical_in_cam})

            grasp_choice = None
            if select_grasp is not None:
                grasp_choice, grasp_content = vlm_handler.request_grasp(viz_list, extract_base_name(target_entity_name))
                print("grasp_choice:", grasp_choice)

            if grasp_choice == 3 and check_contain:
                target_entity = env.task.entities[target_entity_name]
                key_pos, prepare_key_pos, key_quat = find_keypoint_and_prepare_grasp(env, target_entity, prior_eulers,
                                                                                     specific_keypoint_id=specific_keypoint,
                                                                                     move_vector=prepare_quat)
                key_pos, prepare_pos, key_quat = np.array(key_pos), np.array(prepare_key_pos), np.array(key_quat)

            else:
                # pos quat generate
                translation, rotation = g.translation, grasp_rotation
                if grasp_choice is not None:
                    translation, rotation = viz_dict[grasp_choice - 1]["translation"], viz_dict[grasp_choice - 1][
                        "rotation"]

                key_pos, key_quat = graspnet_grasp_to_pose(translation, rotation,
                                                           cam_extr)  # convert grasp to pose in world frame
                key_pos, prepare_pos, key_quat = prepare_grasp_graspnet(env,
                                                                        key_pos,
                                                                        key_quat,
                                                                        g.depth,
                                                                        move_vector=prepare_quat,
                                                                        grasp_choice=grasp_choice
                                                                        )  # using graspnet quats

            # except Exception as e:
            #     print("Failed to convert grasp:", e)
            #     continue

            obstacle_pcd = np.asarray(env.get_obstacle_pcd().points)
            # object_offset
            entity = env.task.entities[target_entity_name]
            entity_mjcf = entity.mjcf_model.worldbody
            object_pos = env.physics.bind(entity_mjcf).xpos
            init2prepare_path = rrt_motion_planning(tuple(start_pos),
                                                    tuple(prepare_pos),
                                                    obstacle_pcd,
                                                    object_pos,
                                                    **motion_planning_kwargs)
            if init2prepare_path is None:
                init2prepare_path = [start_pos, prepare_pos]

            quats_in_path = [start_quat for _ in range(len(init2prepare_path) - 1)]
            quats_in_path.append(key_quat)
            init2prepare_path.append(tuple(key_pos))
            path = np.array(init2prepare_path)
            quats_in_path.append(key_quat)

            interplate_path, interplate_quat = interpolate_path(path, quats_in_path, target_velocity=0.05)
            stage_success = False
            task_success = False

            gripper_state = np.ones(2) * 0.04
            new_obs, new_waypoints, _, task_success = SkillLib.step_trajectory(env,
                                                                               interplate_path,
                                                                               interplate_quat,
                                                                               gripper_state)
            observations.extend(new_obs)

            # save robot
            # img_end = observations[-1]["rgb"][5]
            # cv2.imwrite(os.path.join(img_dir,f"robot{i}.png"),img_end)
            # exit()

            waypoints.extend(new_waypoints)

            # if task_success:
            #     # If already successful (rare before close), pop last observation like original code
            #     observations.pop(-1)
            #     return observations, waypoints, True, task_success

            new_obs, new_waypoints, _, task_success = SkillLib.close_gripper(env)

            observations.extend(new_obs)
            waypoints.extend(new_waypoints)
            # if env.task.entities[target_entity_name].is_grasped(env.physics, env.robot):
            #     stage_success = True
            observations.pop(-1)
            assert len(observations) == len(waypoints)
            return observations, waypoints, stage_success, task_success, viz_list, grasp_choice, is_cloud

    # @staticmethod
    # def place(env,
    #           target_container_name,
    #           target_pos=None,
    #           target_quat=None,
    #           motion_planning_kwargs=dict()):
    #     """
    #     general place function for data generation
    #     param:
    #         env: LM4manipEnv object
    #         target_entity_name: str, target entity name
    #         target_pos: np.array, target position. If None, will propose a target position automatically
    #         target_quat: np.array, target quaternion. If None, will propose a target quaternion automatically
    #     return:
    #         observations: list of obs
    #         waypoints: list of actions
    #         key_frame: list of key action such as move to prepare point, grasp
    #     """
    #     target_container = env.task.entities[target_container_name]
    #     start_pos, start_quat = env.robot.get_end_effector_pos(env.physics), env.robot.get_end_effector_quat(env.physics)
    #     if target_pos is None:
    #         place_points = target_container.get_place_point(env.physics)
    #         if not place_points:
    #             print("can not find valid place point, reset the env")
    #             return None
    #         if isinstance(place_points, list):
    #             place_point = random.choice(place_points)
    #         target_pos = place_point
    #     if target_quat is None:
    #         # if no target_quat is provided, use the default quat
    #         target_quat = env.robot.get_end_effector_quat(env.physics)

    #     obstacle_pcd = np.asarray(env.get_obstacle_pcd().points)
    #     # np.save("obstacle_pcd.npy", obstacle_pcd)

    #     start_pos, start_quat, target_pos, target_quat = np.array(start_pos), np.array(start_quat), np.array(target_pos), np.array(target_quat)
    #     #FIXME if can not find a path, consider change another algorithm
    #     #FIXME optimize the path with min margin to obstacles for safer moving
    #     init2target_path = rrt_motion_planning(tuple(start_pos),
    #                                             tuple(target_pos),
    #                                             obstacle_pcd,
    #                                             **motion_planning_kwargs)
    #     offset = env.robot.ee_offset(env.physics) # for avoid the collision
    #     if init2target_path is None:
    #         print("can not find a path to target position, use default lift")
    #         # default solution is lifting
    #         if start_pos[2] <= target_pos[2]:
    #             mid_point = np.array([start_pos[0], start_pos[1], target_pos[2]])
    #         else:
    #             mid_point = np.array([target_pos[0], target_pos[1], start_pos[2]])

    #         init2target_path = [start_pos, mid_point, target_pos]
    #     path = np.array(init2target_path)
    #     path += offset
    #     path_point_len = len(init2target_path)
    #     quats = [start_quat for _ in range(path_point_len)]
    #     quats[-1] = target_quat
    #     interplate_path, interplate_quat = interpolate_path(path, quats)
    #     observations= [env.get_observation()]
    #     waypoints = []
    #     stage_success = True
    #     task_success = False
    #     new_obs, new_waypoints, _, task_success = SkillLib.step_trajectory(env,
    #                                                             interplate_path,
    #                                                             interplate_quat,
    #                                                             np.zeros(2))
    #     observations.extend(new_obs)
    #     waypoints.extend(new_waypoints)
    #     if task_success:
    #         observations.pop(-1)
    #         assert len(observations) == len(waypoints), f"observations and waypoints should have the same length, {len(observations)} and {len(waypoints)}"
    #         return observations, waypoints, True, task_success
    #     # grasp
    #     new_obs, new_waypoints, _, task_success = SkillLib.open_gripper(env)
    #     observations.extend(new_obs)
    #     waypoints.extend(new_waypoints)

    #     observations.pop(-1)
    #     assert len(observations) == len(waypoints), f"observations and waypoints should have the same length, {len(observations)} and {len(waypoints)}"
    #     for entity in env.task.entities.values():
    #         if hasattr(entity, "is_grasped") and entity.is_grasped(env.physics, env.robot):
    #             stage_success = False
    #     return observations, waypoints, stage_success, task_success

    @staticmethod
    def open_door(env,
                  target_container_name):
        """
        Open the door of the target container
        Input:
            env: LM4manipEnv object
            target_container_name: str, target container name
        Return:
            observations: list of observations
            waypoints: list of waypoints
            trajectory: list of trajectory
            trajectory_quats: list of trajectory quaternions
        """
        target_container = env.task.entities[target_container_name]
        start_pos, start_quat = env.robot.get_end_effector_pos(env.physics), env.robot.get_end_effector_quat(
            env.physics)

        trajectory = target_container.get_open_trajectory(env.physics)
        trajectory_quats = []
        door_joint = target_container.door_joint
        rotation_axis = env.physics.bind(door_joint).xaxis
        # rotation_anchor = env.physics.bind(door_joint).xanchor
        observations = [env.get_observation()]
        waypoints = []
        stage_success = False
        task_success = False
        for i in range(len(trajectory)):
            rot_quat = quaternion_from_axis_angle(rotation_axis, -0.1 * (i + 1))
            new_quat = quaternion_multiply(start_quat, rot_quat)
            trajectory_quats.append(new_quat)
        # init_qpos = np.array(env.robot.get_qpos(env.physics)).reshape(-1)
        interplate_path, interplate_quat = interpolate_path(trajectory, trajectory_quats)
        new_obs, new_waypoints, _, task_success = SkillLib.step_trajectory(env,
                                                                           interplate_path,
                                                                           interplate_quat,
                                                                           np.zeros(2))
        observations.extend(new_obs)
        waypoints.extend(new_waypoints)
        qpos = np.array(env.robot.get_qpos(env.physics)).reshape(-1)
        pos, quat = env.robot.get_end_effector_pos(env.physics), env.robot.get_end_effector_quat(env.physics)
        for _ in range(10):
            action = np.concatenate([qpos, np.ones(2) * (0.04 / 10) * (i + 1)])
            timestep = env.step(action)
            if timestep.last():
                task_success = True
                break
            obs = safe_get_observation(env, observations[-1])
            observations.append(obs)
            waypoints.append(np.concatenate([pos, quaternion_to_euler(quat), np.ones(2) * 0.04]))
        observations.pop(-1)
        assert len(observations) == len(
            waypoints), f"observations and waypoints should have the same length, {len(observations)} and {len(waypoints)}"
        if env.task.entities[target_container_name].is_open(env.physics):
            stage_success = True
        return observations, waypoints, stage_success, task_success

    @staticmethod
    def close_door(env, target_container_name, gripper_state=np.zeros(2)):
        target_container = env.task.entities[target_container_name]
        start_pos, start_quat = env.robot.get_end_effector_pos(env.physics), env.robot.get_end_effector_quat(
            env.physics)

        trajectory = target_container.get_close_trajectory(env.physics)
        trajectory_quats = [start_quat for _ in range(len(trajectory))]

        observations = [env.get_observation()]
        waypoints = []
        stage_success = False
        task_success = False
        if len(trajectory) == 0:
            return observations, waypoints, True, task_success
        interplate_path, interplate_quat = interpolate_path(trajectory, trajectory_quats)
        obs, new_waypoints, _, task_success = SkillLib.step_trajectory(env,
                                                                       interplate_path,
                                                                       interplate_quat,
                                                                       gripper_state)
        observations.extend(obs)
        waypoints.extend(new_waypoints)
        observations.pop(-1)
        assert len(observations) == len(
            waypoints), f"observations and waypoints should have the same length, {len(observations)} and {len(waypoints)}"
        if env.task.entities[target_container_name].is_closed(env.physics):
            stage_success = True
        return observations, waypoints, stage_success, task_success

    # #graspnet
    #     @staticmethod
    #     def pick(env,
    #              vlm_handler=None,
    #              target_entity_name=None,
    #              topk_grasps=5,
    #              prepare_quat=None,
    #              motion_planning_kwargs=dict(),
    #              voxel_size=0.01,
    #              approach_dist=0.05,
    #              collision_thresh=0.01,
    #              bbox = None,
    #              grasp_task = None,
    #              task_name=None,
    #              **kwargs):
    #         """
    #         Use GraspNet to detect grasps from the env masked point cloud, then try to execute them.
    #         - net: pretrained GraspNet model loaded by get_net()
    #         - env: LM4ManipDMEnv
    #         Returns same signature as original pick: observations, waypoints, stage_success, task_success

    #         Current logic:
    #         1. Use graspnet to generate grasps from wrist camera
    #         2. Filter grasps by camera z-coordinates, keep the grasps that are more likely to be on the object
    #         3. Rank grasps by camera y-coordinates (grasp point on object)
    #         4. Use pre-defined quaternion along with graspnet translation for grasp execution
    #         """
    #         # if grasp_task is not None:
    #         #     prepare_quat = np.array([0, 0, -1])
    #         target_entity = env.task.entities[target_entity_name]
    #         start_pos = np.array(env.robot.get_end_effector_pos(env.physics))
    #         start_quat = np.array(env.robot.get_end_effector_quat(env.physics))

    #         '''Add bounding box HERE!'''
    #         bounding_box = bbox  # for select_carrot wrist camera
    #         # bounding_box = [190, 50, 275, 120]

    #         # for select_carrot wrist camera
    #         end_points, cloud, cam_intr, cam_extr = get_graspnet_pcd(env, bbox=bounding_box, cam_id=3)
    #         gg = run_graspnet_on_pcd(end_points, cloud, bbox=bounding_box, intr=cam_intr)
    #         gg_col = filter_grasps_collision(gg, np.array(cloud.points), voxel_size=voxel_size, approach_dist=approach_dist,
    #                                          collision_thresh=collision_thresh)
    #         gg_pos = None
    #         y_p = 40
    #         z_p = 20
    #         while gg_pos == None:
    #             if len(filter_grasps_position(gg_col, y_p=y_p, z_p=z_p)) < topk_grasps:
    #                 z_p += 5
    #                 z_p += 10
    #             else:
    #                 gg_pos = filter_grasps_position(gg_col, y_p=y_p, z_p=z_p)
    #         candidate_grasps = select_topk_grasps(gg_pos, topk=1)
    #         print("candidate_grasps:",candidate_grasps)
    #         # vis_grasps(candidate_grasps, cloud)

    #         observations = [env.get_observation()]  # start observation
    #         waypoints = []
    #         viz_list = []
    #         for g in candidate_grasps:
    #             print("g.rotation_matrix")
    #             try:

    #                 grasp_rotation = graspnet_rot_correction(g.rotation_matrix)

    #                 rgb_img = env.render(camera_id=3, height=480, width=480)
    #                 viz_1 = vis_gripper(rgb_img, g.translation, grasp_rotation, cam_intr, g.width, g.depth)
    #                 viz_list.append(viz_1)

    #                 r_perfect = np.array([
    #                     0.99932894, -0.00563362, -0.03619311,
    #                     -0.03582271, 0.05585626, -0.99779598,
    #                     0.00764282, 0.99842293, 0.05561697
    #                 ]).reshape(3, 3)
    #                 R_W_C = cam_extr[:3, :3]
    #                 print(R_W_C.shape)  # output should be (3, 3)
    #                 r_perfect_in_cam = R_W_C.T @ r_perfect
    #                 viz_2 = vis_gripper(rgb_img, g.translation, r_perfect_in_cam, cam_intr, g.width, g.depth)
    #                 viz_list.append(viz_2)

    #                 # Build the rotation matrix Rx that rotates 180 degrees around the X axis
    #                 Rx_flip = o3d.geometry.get_rotation_matrix_from_xyz([-np.pi/2, 0.0, 0.0])
    #                 # Flip-align the pose in the camera frame
    #                 r_perfect_in_cam_ = Rx_flip @ r_perfect_in_cam
    #                 viz_3 = vis_gripper(rgb_img, g.translation, r_perfect_in_cam_, cam_intr, g.width, g.depth)
    #                 viz_list.append(viz_3)

    #                 # key_pos, key_quat = graspnet_grasp_to_pose(g.translation, grasp_rotation,
    #                 #                                            cam_extr)  # convert grasp to pose in world frame
    #                 key_pos, key_quat = graspnet_grasp_to_pose(g.translation, r_perfect_in_cam,
    #                                             cam_extr)
    #                 key_pos, prepare_pos, key_quat = prepare_grasp_graspnet(env, key_pos, key_quat, g.depth,
    #                                                                         move_vector=prepare_quat)  # using graspnet quats
    #             except Exception as e:
    #                 print("Failed to convert grasp:", e)
    #                 continue

    #             # if grasp_task is not None:
    #             #     option,content = vlm_handler.request_grasp(
    #             #         viz_list=viz_list,
    #             #         target_entity=target_entity_name
    #             #     )
    #             #     # 1. Define the path
    #             #     file_path = os.path.join("/home/sankuai/work/RMMBench/B_test_data", f'{task_name}_data.json')

    #             #     # 2. Extract the current maximum step
    #             #     current_step = 0
    #             #     if os.path.isfile(file_path):
    #             #         with open(file_path, 'r', encoding='utf-8') as f:
    #             #             lines = f.readlines()
    #             #             if lines:
    #             #                     # Read the last line to get the latest step
    #             #                 try:
    #             #                     last_entry = json.loads(lines[-1].strip())
    #             #                     current_step = last_entry.get('step', 0)
    #             #                 except json.JSONDecodeError:
    #             #                     # Guard against blank or corrupted lines at the end of the file
    #             #                     print("Warning: Last line is not valid JSON, scanning all lines...")
    #             #                     for line in reversed(lines):
    #             #                         try:
    #             #                             last_entry = json.loads(line.strip())
    #             #                             current_step = last_entry.get('step', 0)
    #             #                             break
    #             #                         except: continue

    #             #     # 3. Build and append the new data
    #             #     data_entry = {
    #             #         "step": current_step + 1,
    #             #         "content": content
    #             #     }

    #             #     # Append directly in 'a' mode, one JSON object per line
    #             #     with open(file_path, 'a', encoding='utf-8') as f:
    #             #         json_record = json.dumps(data_entry, ensure_ascii=False)
    #             #         f.write(json_record + '\n')

    #             obstacle_pcd = np.asarray(env.get_obstacle_pcd().points)
    #             init2prepare_path = rrt_motion_planning(tuple(start_pos),
    #                                                     tuple(prepare_pos),
    #                                                     obstacle_pcd,
    #                                                     **motion_planning_kwargs)
    #             if init2prepare_path is None:
    #                 init2prepare_path = [start_pos, prepare_pos]

    #             quats_in_path = [start_quat for _ in range(len(init2prepare_path) - 1)]
    #             quats_in_path.append(key_quat)
    #             init2prepare_path.append(tuple(key_pos))
    #             path = np.array(init2prepare_path)
    #             quats_in_path.append(key_quat)

    #             interplate_path, interplate_quat = interpolate_path(path, quats_in_path, target_velocity=0.05)
    #             stage_success = False
    #             task_success = False

    #             gripper_state = np.ones(2) * 0.04
    #             new_obs, new_waypoints, _, task_success = SkillLib.step_trajectory(env,
    #                                                                                interplate_path,
    #                                                                                interplate_quat,
    #                                                                                gripper_state)
    #             observations.extend(new_obs)
    #             waypoints.extend(new_waypoints)

    #             if task_success:
    #                 # If already successful (rare before close), pop last observation like original code
    #                 observations.pop(-1)
    #                 return observations, waypoints, True, task_success

    #             new_obs, new_waypoints, _, task_success = SkillLib.close_gripper(env)
    #             observations.extend(new_obs)
    #             waypoints.extend(new_waypoints)
    #             if env.task.entities[target_entity_name].is_grasped(env.physics, env.robot):
    #                 stage_success = True
    #             observations.pop(-1)
    #             assert len(observations) == len(waypoints)
    #             return observations, waypoints, stage_success, task_success,viz_list

    # @staticmethod
    # def pick(env,
    #          target_entity_name=None,
    #          target_pos=None,
    #          target_quat=None,
    #          prepare_distance=-0.1,
    #          prepare_quat=None,
    #          prior_eulers=PRIOR_EULERS,
    #          specific_keypoint=None,
    #          target_velocity=0.05,
    #          motion_planning_kwargs=dict(),
    #          bbox=None,
    #          topk_grasps=5,
    #          voxel_size=0.01,
    #          approach_dist=0.05,
    #          collision_thresh=0.01,
    #          **kwargs):
    #     """
    #     Use GraspNet to detect grasps from the env masked point cloud, then try to execute them.
    #     - net: pretrained GraspNet model loaded by get_net()
    #     - env: LM4ManipDMEnv
    #     Returns same signature as original pick: observations, waypoints, stage_success, task_success

    #     Current logic:
    #     1. Use graspnet to generate grasps from wrist camera
    #     2. Filter grasps by camera z-coordinates, keep the grasps that are more likely to be on the object
    #     3. Rank grasps by camera y-coordinates (grasp point on object)
    #     4. Use pre-defined quaternion along with graspnet translation for grasp execution
    #     """

    #     target_entity = env.task.entities[target_entity_name]
    #     if target_pos is None or target_quat is None:
    #         key_pos, prepare_key_pos, key_quat = find_keypoint_and_prepare_grasp(env, target_entity, prior_eulers, specific_keypoint_id=specific_keypoint, move_vector=prepare_quat)
    #         if key_pos is None or prepare_key_pos is None:
    #             print("can not find valid keypoint and prepare point, reset the env")
    #             return None
    #     else:
    #         key_pos, key_quat = target_pos, target_quat
    #         if prepare_quat is None:
    #             gripper_pcd, move_quat = env.robot.gripper_pcd(key_pos, key_quat)
    #         else:
    #             move_quat = prepare_quat
    #         prepare_key_pos = key_pos + move_quat * prepare_distance
    #     start_pos, start_quat = env.robot.get_end_effector_pos(env.physics), env.robot.get_end_effector_quat(env.physics)
    #     # env_pcd = env.get_observation()["masked_point_cloud"]
    #     # obstacle_pcd = np.asarray(env_pcd.points)

    #     obstacle_pcd = np.asarray(env.get_obstacle_pcd().points)

    #     start_pos, start_quat, key_quat, prepare_pos, key_pos = np.array(start_pos), np.array(start_quat), np.array(key_quat), np.array(prepare_key_pos), np.array(key_pos)
    #     # motion planning -> path(start, prepare_point) & path(prepare_point, key_point)

    #     init2prepare_path = rrt_motion_planning(tuple(start_pos),
    #                                             tuple(prepare_pos),
    #                                             obstacle_pcd,
    #                                             **motion_planning_kwargs)
    #     if init2prepare_path is None:
    #         init2prepare_path = [start_pos, prepare_pos]
    #     quats_in_path = [start_quat for _ in range(len(init2prepare_path)-1)]
    #     quats_in_path.append(key_quat)
    #     # quats_in_path = []
    #     # for t in np.linspace(0, 1, len(init2prepare_path), endpoint=False):
    #     #     quats_in_path.append(qauternion_slerp(start_quat, key_quat, t))
    #     init2prepare_path.append(tuple(key_pos))
    #     path = np.array(init2prepare_path)
    #     quats_in_path.append(key_quat)

    #     interplate_path, interplate_quat = interpolate_path(path, quats_in_path, target_velocity)

    #     waypoints = []
    #     stage_success = False
    #     task_success = False
    #     observations = [env.get_observation()]
    #     # move along the interplated path
    #     gripper_state = np.ones(2) * 0.04
    #     new_obs, new_waypoints, _, task_success = SkillLib.step_trajectory(env,
    #                                                                interplate_path,
    #                                                                interplate_quat,
    #                                                                gripper_state,
    #                                                                **kwargs)
    #     observations.extend(new_obs)
    #     waypoints.extend(new_waypoints)
    #     if task_success:
    #         observations.pop(-1)
    #         return observations, waypoints, True, task_success
    #     # grasp
    #     new_obs, new_waypoints, _, task_success = SkillLib.close_gripper(env)

    #     observations.extend(new_obs)
    #     waypoints.extend(new_waypoints)

    #     observations.pop(-1)

    #     # #rgb: dict_keys(['q_state', 'q_velocity', 'q_acceleration',
    #     # # 'rgb', 'depth', 'segmentation', 'robot_mask', 'instrinsic',
    #     # # 'extrinsic', 'masked_point_cloud', 'point_cloud', 'ee_state', 'grasped_obj_name'])
    #     # img_path = "/Users/lh/work/RMMBench/test_img/pick_carrot/observations.png"
    #     # rgb = observations[-1]
    #     #
    #     # four_p = np.vstack([np.hstack(rgb["rgb"][:2]), np.hstack(rgb["rgb"][2:4])])
    #     # four_p = cv2.cvtColor(four_p, cv2.COLOR_BGR2RGB)
    #     # cv2.imwrite(img_path,four_p)

    #     assert len(observations) == len(waypoints), f"observations and waypoints should have the same length, {len(observations)} and {len(waypoints)}"
    #     if env.task.entities[target_entity_name].is_grasped(env.physics, env.robot):
    #         stage_success = True

    #     return observations, waypoints, stage_success, task_success

    @staticmethod
    def place(env,
              target_container_name=None,
              bbox=None,
              target_pos=None,
              target_quat=None,
              placement_th=0.6,
              open_bbox=1,
              motion_planning_kwargs=dict()):
        """
        general place function for data generation
        param:
            env: LM4manipEnv object
            target_entity_name: str, target entity name
            target_pos: np.array, target position. If None, will propose a target position automatically
            target_quat: np.array, target quaternion. If None, will propose a target quaternion automatically
        return:
            observations: list of obs
            waypoints: list of actions
            key_frame: list of key action such as move to prepare point, grasp
        """
        # SkillLib.lift(env,gripper_state=np.zeros(2),lift_height=lift_height)
        SkillLib.lift(env)
        print("target_pos:", target_pos)
        print("target_entity_name:", target_container_name)
        # print("flatten_list(target_container_name)[0]:",flatten_list(target_container_name)[0])
        if isinstance(target_container_name, list):
            target_container_name = random.choice(target_container_name)

        if target_pos is not None and target_container_name is not None:
            target_container = env.task.entities[target_container_name]
        start_pos, start_quat = env.robot.get_end_effector_pos(env.physics), env.robot.get_end_effector_quat(
            env.physics)

        # if target_pos is None:
        # place_points = target_container.get_place_point(env.physics)

        # bbox=[256 ,73 ,311, 134]
        # bbox = [0,480,0,480]
        if bbox is not None:
            print("default_placepoint:", target_pos)

            if target_container_name is not None:
                for fixture in ["sink"]:
                    if fixture in target_container_name:
                        placement_th += 0.25
            print("placement_th:", placement_th)

            place_point = get_placement_top_down_view(env, bbox=bbox, cam_id=5, placement_offset=placement_th,
                                                      target_container_name=target_container_name)
            # if not place_points:
            #     print("can not find valid place point, reset the env")
            #     return None
            # if isinstance(place_points, list):
            #     place_point = random.choice(place_points)
            print("place_point:", place_point)
        elif target_pos is None:
            place_points = target_container.get_place_point(env.physics)
            place_point = random.choice(place_points)
        else:
            place_point = target_pos
        target_pos = place_point

        if target_quat is None:
            # if no target_quat is provided, use the default quat
            target_quat = env.robot.get_end_effector_quat(env.physics)

        print("最终target_pos:", target_pos)
        obstacle_pcd = np.asarray(env.get_obstacle_pcd().points)
        # np.save("obstacle_pcd.npy", obstacle_pcd)

        start_pos, start_quat, target_pos, target_quat = np.array(start_pos), np.array(start_quat), np.array(
            target_pos), np.array(target_quat)
        # FIXME if can not find a path, consider change another algorithm
        # FIXME optimize the path with min margin to obstacles for safer moving
        init2target_path = rrt_motion_planning(tuple(start_pos),
                                               tuple(target_pos),
                                               obstacle_pcd,
                                               **motion_planning_kwargs)
        offset = env.robot.ee_offset(env.physics)  # for avoid the collision

        if init2target_path is None:
            print("can not find a path to target position, use default lift")
            # default solution is lifting
            if start_pos[2] <= target_pos[2]:
                mid_point = np.array([start_pos[0], start_pos[1], target_pos[2]])
            else:
                mid_point = np.array([target_pos[0], target_pos[1], start_pos[2]])

            init2target_path = [start_pos, mid_point, target_pos]
        path = np.array(init2target_path)
        path += offset
        path_point_len = len(init2target_path)
        quats = [start_quat for _ in range(path_point_len)]
        quats[-1] = target_quat
        interplate_path, interplate_quat = interpolate_path(path, quats)
        observations = [env.get_observation()]
        waypoints = []
        stage_success = True
        task_success = False
        new_obs, new_waypoints, _, task_success = SkillLib.step_trajectory(env,
                                                                           interplate_path,
                                                                           interplate_quat,
                                                                           np.zeros(2))
        observations.extend(new_obs)
        waypoints.extend(new_waypoints)
        if task_success:
            observations.pop(-1)
            assert len(observations) == len(
                waypoints), f"observations and waypoints should have the same length, {len(observations)} and {len(waypoints)}"
            return observations, waypoints, True, task_success
        # grasp
        new_obs, new_waypoints, _, task_success = SkillLib.open_gripper(env)
        observations.extend(new_obs)
        waypoints.extend(new_waypoints)

        observations.pop(-1)
        assert len(observations) == len(
            waypoints), f"observations and waypoints should have the same length, {len(observations)} and {len(waypoints)}"
        for entity in env.task.entities.values():
            if hasattr(entity, "is_grasped") and entity.is_grasped(env.physics, env.robot):
                stage_success = False
        return observations, waypoints, stage_success, task_success

    @staticmethod
    def move_forward(env, target_pos=None, target_quat=None, gripper_state=None, move_distance=0.3,
                     target_velocity=0.05):
        """
        mobile_base move function.
        """
        # Get the base pos/quat; no need to worry about timestep since we interpolate
        start_pos, start_quat = env.robot.get_mobile_base_pos(env.physics), env.robot.get_mobile_base_quat(
            env.physics)

        if target_pos is None: target_pos = np.array(start_pos) + np.array([0, -move_distance, 0])
        if target_quat is None: target_quat = start_quat

        # Interpolate by velocity - only concerns the end pos
        interplate_path, interplate_quat = interpolate_path([start_pos, target_pos],
                                                            [np.array(start_quat), np.array(target_quat)],
                                                            target_velocity)
        observations = [env.get_observation()]

        waypoints = []
        task_success = False
        # Get the gripper opening
        if gripper_state is None:
            gripper_closed = env.robot.get_ee_open_state(env.physics)
            if gripper_closed:
                gripper_state = np.zeros(2)
            else:
                gripper_state = np.ones(2) * 0.04

        new_obs, new_waypoints, stage_success, task_success = SkillLib.step_trajectory(env,
                                                                                       interplate_path,
                                                                                       interplate_quat,
                                                                                       gripper_state)
        observations.extend(new_obs)
        waypoints.extend(new_waypoints)
        observations.pop(-1)
        assert len(observations) == len(
            waypoints), f"observations and waypoints should have the same length, {len(observations)} and {len(waypoints)}"
        return observations, waypoints, stage_success, task_success

    @staticmethod
    def open_door(env,
                  target_container_name):
        """
        Open the door of the target container
        Input:
            env: LM4manipEnv object
            target_container_name: str, target container name
        Return:
            observations: list of observations
            waypoints: list of waypoints
            trajectory: list of trajectory
            trajectory_quats: list of trajectory quaternions
        """
        target_container = env.task.entities[target_container_name]
        start_pos, start_quat = env.robot.get_end_effector_pos(env.physics), env.robot.get_end_effector_quat(
            env.physics)

        trajectory = target_container.get_open_trajectory(env.physics)
        trajectory_quats = []
        door_joint = target_container.door_joint
        rotation_axis = env.physics.bind(door_joint).xaxis
        # rotation_anchor = env.physics.bind(door_joint).xanchor
        observations = [env.get_observation()]
        waypoints = []
        stage_success = False
        task_success = False
        for i in range(len(trajectory)):
            rot_quat = quaternion_from_axis_angle(rotation_axis, -0.1 * (i + 1))
            new_quat = quaternion_multiply(start_quat, rot_quat)
            trajectory_quats.append(new_quat)
        # init_qpos = np.array(env.robot.get_qpos(env.physics)).reshape(-1)
        interplate_path, interplate_quat = interpolate_path(trajectory, trajectory_quats)
        new_obs, new_waypoints, _, task_success = SkillLib.step_trajectory(env,
                                                                           interplate_path,
                                                                           interplate_quat,
                                                                           np.zeros(2))
        observations.extend(new_obs)
        waypoints.extend(new_waypoints)
        qpos = np.array(env.robot.get_qpos(env.physics)).reshape(-1)
        pos, quat = env.robot.get_end_effector_pos(env.physics), env.robot.get_end_effector_quat(env.physics)
        for _ in range(10):
            if len(qpos) > 8:
                action = np.concatenate([qpos[:7], np.ones(2) * (0.04 / 10) * (_ + 1), qpos[7:]])
            else:
                action = np.concatenate([qpos, np.ones(2) * (0.04 / 10) * (_ + 1)])
            timestep = env.step(action)
            if timestep.last():
                task_success = True
                break
            obs = safe_get_observation(env, observations[-1])
            observations.append(obs)
            waypoints.append(np.concatenate([pos, quaternion_to_euler(quat), np.ones(2) * 0.04]))
        observations.pop(-1)
        assert len(observations) == len(
            waypoints), f"observations and waypoints should have the same length, {len(observations)} and {len(waypoints)}"
        if env.task.entities[target_container_name].is_open(env.physics):
            stage_success = True
        return observations, waypoints, stage_success, task_success

    @staticmethod
    def close_door(env, target_container_name, gripper_state=np.zeros(2)):
        target_container = env.task.entities[target_container_name]
        start_pos, start_quat = env.robot.get_end_effector_pos(env.physics), env.robot.get_end_effector_quat(
            env.physics)

        trajectory = target_container.get_close_trajectory(env.physics)
        trajectory_quats = [start_quat for _ in range(len(trajectory))]

        observations = [env.get_observation()]
        waypoints = []
        stage_success = False
        task_success = False
        if len(trajectory) == 0:
            return observations, waypoints, True, task_success
        interplate_path, interplate_quat = interpolate_path(trajectory, trajectory_quats)
        obs, new_waypoints, _, task_success = SkillLib.step_trajectory(env,
                                                                       interplate_path,
                                                                       interplate_quat,
                                                                       gripper_state)
        observations.extend(obs)
        waypoints.extend(new_waypoints)
        observations.pop(-1)
        assert len(observations) == len(
            waypoints), f"observations and waypoints should have the same length, {len(observations)} and {len(waypoints)}"
        if env.task.entities[target_container_name].is_closed(env.physics):
            stage_success = True
        return observations, waypoints, stage_success, task_success

    @staticmethod
    def open_drawer(env,
                    target_container_name,
                    pick_prior_eulers=[[-np.pi / 2, 0, 0]],
                    drawer_id=0):
        """
        common open drawer function
        Input:
            drawer_id: 0-2 means top to bottom drawer.
            pick_prioer_euelrs: list of list, prior eulers for pick
        """
        observations = [env.get_observation()]
        waypoints = []
        stage_success = False
        task_success = False
        target_container = env.task.entities[target_container_name]
        # grasp handle
        new_obs, new_waypoints, _, success_ = SkillLib.pick(env, target_container_name, prior_eulers=pick_prior_eulers,
                                                            specific_keypoint=drawer_id)

        observations.extend(new_obs)
        waypoints.extend(new_waypoints)
        task_success = task_success or success_
        # open drawer
        start_pos, start_quat = env.robot.get_end_effector_pos(env.physics), env.robot.get_end_effector_quat(
            env.physics)
        trajectory = target_container.get_drawer_open_trajectory(env.physics, drawer_id)
        trajectory_quats = [start_quat for _ in range(len(trajectory))]
        trajectory, trajectory_quats = interpolate_path(trajectory, trajectory_quats)
        new_obs, new_waypoints, _, success_ = SkillLib.step_trajectory(env, trajectory, trajectory_quats, np.zeros(2))

        observations.extend(new_obs)
        waypoints.extend(new_waypoints)
        task_success = task_success or success_
        observations.pop(-1)

        assert len(observations) == len(
            waypoints), f"observations and waypoints should have the same length, {len(observations)} and {len(waypoints)}"
        # TODO check the drawer state
        return observations, waypoints, True, task_success

    @staticmethod
    def press(env, target_pos, target_quat=None, move_vector=[0, 0, 0.1],
              max_n_substep=100):  # TODO move vector to determine the press direction
        prepare_pos = target_pos + np.array(move_vector) if move_vector is not None else target_pos
        observations, waypoints, _, _ = SkillLib.moveto(env,
                                                        prepare_pos,
                                                        target_quat,
                                                        max_n_substep=max_n_substep)
        # close gripper
        qpos = np.array(env.robot.get_qpos(env.physics)).reshape(-1)
        for i in range(10):
            gripper_state = np.ones(2) * (0.04 - i / 10 * 0.04)
            action = np.concatenate([qpos, gripper_state])
            timestep = env.step(action)
            if timestep.last():
                break
            obs = safe_get_observation(env, observations[-1])
            observations.append(obs)
            waypoints.append(np.concatenate([env.robot.get_end_effector_pos(env.physics),
                                             quaternion_to_euler(env.robot.get_end_effector_quat(env.physics)),
                                             gripper_state]))
        new_obs, new_waypoints, stage_success, task_success = SkillLib.moveto(env, target_pos, target_quat,
                                                                              max_n_substep=max_n_substep)
        observations.extend(new_obs)
        waypoints.extend(new_waypoints)
        assert len(observations) == len(
            waypoints), f"observations and waypoints should have the same length, {len(observations)} and {len(waypoints)}"
        return observations, waypoints, stage_success, task_success

    @staticmethod
    def pull(env, target_pos=None, target_quat=None, gripper_state=None, pull_distance=0.3):
        """
        Common pull function.
        """
        start_pos, start_quat = env.robot.get_end_effector_pos(env.physics), env.robot.get_end_effector_quat(
            env.physics)
        if target_pos is None: target_pos = np.array(start_pos) + np.array([0, -pull_distance, 0])
        if target_quat is None: target_quat = start_quat
        interplate_path, interplate_quat = interpolate_path([start_pos, target_pos],
                                                            [np.array(start_quat), np.array(target_quat)])
        observations = [env.get_observation()]
        waypoints = []
        task_success = False
        if gripper_state is None:
            gripper_closed = env.robot.get_ee_open_state(env.physics)
            if gripper_closed:
                gripper_state = np.zeros(2)
            else:
                gripper_state = np.ones(2) * 0.04
        new_obs, new_waypoints, stage_success, task_success = SkillLib.step_trajectory(env,
                                                                                       interplate_path,
                                                                                       interplate_quat,
                                                                                       gripper_state)
        observations.extend(new_obs)
        waypoints.extend(new_waypoints)
        observations.pop(-1)
        assert len(observations) == len(
            waypoints), f"observations and waypoints should have the same length, {len(observations)} and {len(waypoints)}"
        return observations, waypoints, stage_success, task_success

    @staticmethod
    def push(env, target_pos=None, target_quat=None, gripper_state=None, push_distance=0.3):
        obs, waypoints, stage_success, task_success = SkillLib.pull(env, target_pos, target_quat, gripper_state,
                                                                    -push_distance)
        return obs, waypoints, stage_success, task_success

    @staticmethod
    def pour(env, target_delta_qpos=np.pi, target_q_velocity=np.pi / 40, n_repeat_step=2, tolerance=0.01):
        """
        Common pour function.
        """
        waypoints = []
        stage_success = False
        task_success = False
        observations = [env.get_observation()]

        init_qpos = np.array(env.robot.get_qpos(env.physics))
        gripper_closed = env.robot.get_ee_open_state(env.physics)
        if gripper_closed:
            gripper_state = np.zeros(2)
        else:
            gripper_state = np.ones(2) * 0.04
        timesteps = int(target_delta_qpos / target_q_velocity)
        for i in range(timesteps):
            action = np.array(init_qpos).reshape(-1)
            action[-1] += target_q_velocity * i
            action = np.concatenate([action, gripper_state])
            for _ in range(n_repeat_step):
                timestep = env.step(action)
                if timestep.last():
                    task_success = True
                    break
                current_qpos = np.array(env.task.robot.get_qpos(env.physics)).reshape(-1)
                if np.max(current_qpos - np.array(action[:7])) < tolerance \
                        and np.min(current_qpos - np.array(action[:7])) > -tolerance:
                    break
            waypoint = np.concatenate([env.robot.get_end_effector_pos(env.physics),
                                       quaternion_to_euler(env.robot.get_end_effector_quat(env.physics)),
                                       gripper_state])
            obs = safe_get_observation(env, observations[-1])
            observations.append(obs)
            waypoints.append(waypoint)
            if task_success:
                break
        observations.pop(-1)
        assert len(observations) == len(
            waypoints), f"observations and waypoints should have the same length, {len(observations)} and {len(waypoints)}"
        return observations, waypoints, True, task_success

    @staticmethod
    def lift(env, target_pos=None, target_quat=None, gripper_state=np.zeros(2), lift_height=0.2):
        """
        Common lift function.
        """
        start_pos, start_quat = env.robot.get_end_effector_pos(env.physics), env.robot.get_end_effector_quat(
            env.physics)
        if target_pos is None:
            target_pos = np.array(start_pos) + np.array([0, 0, lift_height])
        if target_quat is None:
            target_quat = start_quat
        interplate_path, interplate_quat = interpolate_path([start_pos, target_pos],
                                                            [np.array(start_quat), np.array(target_quat)])
        observations = [env.get_observation()]
        waypoints = []
        if gripper_state is None:
            gripper_closed = env.robot.get_ee_open_state(env.physics)
            if gripper_closed:
                gripper_state = np.zeros(2)
            else:
                gripper_state = np.ones(2) * 0.04
        obs, new_waypoints, stage_success, task_success = SkillLib.step_trajectory(env,
                                                                                   interplate_path,
                                                                                   interplate_quat,
                                                                                   gripper_state)
        observations.extend(obs)
        waypoints.extend(new_waypoints)
        observations.pop(-1)
        assert len(observations) == len(
            waypoints), f"observations and waypoints should have the same length, {len(observations)} and {len(waypoints)}"
        return observations, waypoints, stage_success, task_success

    @staticmethod
    def observe(env,
                target_pos=[-0.19955923, -0.19029609, 1.27651386],
                target_quat=[-0.017108027748582093, -0.6941570090325536, 0.7192555094587563, -0.022910135546478864],
                motion_planning_kwargs=dict()):
        """
        Return to the initial TCP pose
        """

        start_pos, start_quat = env.robot.get_end_effector_pos(env.physics), env.robot.get_end_effector_quat(
            env.physics)

        gripper_state = np.ones(2) * 0.04
        ee_state = env.robot.get_ee_open_state(env.physics)
        if ee_state:
            gripper_state = np.zeros(2)

        obstacle_pcd = np.asarray(env.get_obstacle_pcd().points)
        # np.save("obstacle_pcd.npy", obstacle_pcd)

        start_pos, start_quat, target_pos, target_quat = np.array(start_pos), np.array(start_quat), np.array(
            target_pos), np.array(target_quat)
        # FIXME if can not find a path, consider change another algorithm
        # FIXME optimize the path with min margin to obstacles for safer moving
        init2target_path = rrt_motion_planning(tuple(start_pos),
                                               tuple(target_pos),
                                               obstacle_pcd,
                                               **motion_planning_kwargs)
        offset = env.robot.ee_offset(env.physics)  # for avoid the collision

        if init2target_path is None:
            print("can not find a path to target position, use default lift")
            # default solution is lifting
            if start_pos[2] <= target_pos[2]:
                mid_point = np.array([start_pos[0], start_pos[1], target_pos[2]])
            else:
                mid_point = np.array([target_pos[0], target_pos[1], start_pos[2]])

            init2target_path = [start_pos, mid_point, target_pos]
        path = np.array(init2target_path)
        path += offset
        path_point_len = len(init2target_path)
        quats = [start_quat for _ in range(path_point_len)]
        quats[-1] = target_quat
        interplate_path, interplate_quat = interpolate_path(path, quats)
        observations = [env.get_observation()]
        waypoints = []
        stage_success = True
        task_success = False
        new_obs, new_waypoints, _, task_success = SkillLib.step_trajectory(env,
                                                                           interplate_path,
                                                                           interplate_quat,
                                                                           gripper_state,
                                                                           end_dpos=[0, -1.04, 0, -2.56, 0, 1.55,
                                                                                     -2.32])
        observations.extend(new_obs)
        waypoints.extend(new_waypoints)
        if task_success:
            observations.pop(-1)
            assert len(observations) == len(
                waypoints), f"observations and waypoints should have the same length, {len(observations)} and {len(waypoints)}"
            return observations, waypoints, True, task_success

        observations.pop(-1)
        assert len(observations) == len(
            waypoints), f"observations and waypoints should have the same length, {len(observations)} and {len(waypoints)}"
        for entity in env.task.entities.values():
            if hasattr(entity, "is_grasped") and entity.is_grasped(env.physics, env.robot):
                stage_success = False
        return observations, waypoints, stage_success, task_success

    @staticmethod
    def close_gripper(env, repeat=5):
        qpos = np.array(env.robot.get_qpos(env.physics)).reshape(-1)

        observations = [env.get_observation()]
        waypoints = []
        success = False
        for i in range(10):
            gripper_state = np.ones(2) * (0.04 - i * 0.04 / 10)
            action = np.concatenate([qpos[:7], gripper_state, qpos[7:]])
            for _ in range(repeat):
                timestep = env.step(action)
                if timestep.last():
                    success = True
                    obs = safe_get_observation(env, observations[-1])
                    observations.append(obs)
                    waypoints.append(np.concatenate([env.robot.get_end_effector_pos(env.physics),
                                                     quaternion_to_euler(env.robot.get_end_effector_quat(env.physics)),
                                                     gripper_state]))
                    break

            obs = safe_get_observation(env, observations[-1])
            observations.append(obs)
            waypoints.append(np.concatenate([env.robot.get_end_effector_pos(env.physics),
                                             quaternion_to_euler(env.robot.get_end_effector_quat(env.physics)),
                                             gripper_state]))
        # _, _, _, _ = SkillLib.wait(env, wait_time=5,gripper_state=np.zeros(2))

        observations.pop(-1)
        assert len(observations) == len(
            waypoints), f"observations and waypoints should have the same length, {len(observations)} and {len(waypoints)}"
        return observations, waypoints, True, success

    @staticmethod
    def open_gripper(env, repeat=4):
        """
         qpos = np.array(env.robot.get_qpos(env.physics)).reshape(-1)
        pos, quat = env.robot.get_end_effector_pos(env.physics), env.robot.get_end_effector_quat(env.physics)
        for _ in range(10):
            action = np.concatenate([qpos, np.ones(2)*(0.04/10)*(i+1)])
            timestep = env.step(action)
            if timestep.last():
                task_success = True
                break

        """
        qpos = np.array(env.robot.get_qpos(env.physics)).reshape(-1)
        observations = [env.get_observation()]
        waypoints = []
        task_success = False
        stage_success = False
        for i in range(20):
            gripper_state = np.ones(2) * (i + 1) / 10 * 0.04
            if len(qpos) > 8:
                action = np.concatenate([qpos[:7], gripper_state, qpos[7:]])
            else:
                action = np.concatenate([qpos, np.ones(2) * (0.04 / 10) * (i + 1)])
            for _ in range(repeat):
                timestep = env.step(action)
                if timestep.last():
                    task_success = True
                    obs = safe_get_observation(env, observations[-1])
                    observations.append(obs)
                    waypoints.append(np.concatenate([env.robot.get_end_effector_pos(env.physics),
                                                     quaternion_to_euler(env.robot.get_end_effector_quat(env.physics)),
                                                     gripper_state]))
            obs = safe_get_observation(env, observations[-1])
            observations.append(obs)
            waypoints.append(np.concatenate([env.robot.get_end_effector_pos(env.physics),
                                             quaternion_to_euler(env.robot.get_end_effector_quat(env.physics)),
                                             gripper_state]))
            if timestep.last():
                task_success = True
                break
        observations.pop(-1)
        # _, _, _, _ = SkillLib.wait(env, wait_time=20,gripper_state=np.ones(2)*0.04)

        assert len(observations) == len(
            waypoints), f"observations and waypoints should have the same length, {len(observations)} and {len(waypoints)}"
        if env.robot.get_ee_open_state(env.physics):
            stage_success = True
        return observations, waypoints, stage_success, task_success

    @staticmethod
    def flip(env, gripper_state=None, target_q_velocity=np.pi / 40, max_n_substep=30, tolerance=0.01):
        qpos = np.array(env.robot.get_qpos(env.physics)).reshape(-1)
        observations = [env.get_observation()]
        waypoints = []
        if gripper_state is None:
            gripper_closed = env.robot.get_ee_open_state(env.physics)
            if gripper_closed:
                gripper_state = np.zeros(2)
            else:
                gripper_state = np.ones(2) * 0.04
        timestep = int(np.pi / target_q_velocity)
        success = False
        for i in range(timestep):
            action = np.array(qpos).copy()
            action[-1] += target_q_velocity * i
            action = np.concatenate([action, gripper_state])
            for _ in range(max_n_substep):
                timestep = env.step(action)
                if timestep.last():
                    success = True
                    break
                current_qpos = np.array(env.task.robot.get_qpos(env.physics)).reshape(-1)
                if np.max(current_qpos - np.array(action[:7])) < tolerance \
                        and np.min(current_qpos - np.array(action[:7])) > -tolerance:
                    break
            if success:
                break
            waypoint = np.concatenate([env.robot.get_end_effector_pos(env.physics),
                                       quaternion_to_euler(env.robot.get_end_effector_quat(env.physics)),
                                       gripper_state])
            obs = safe_get_observation(env, observations[-1])
            observations.append(obs)
            waypoints.append(waypoint)
        observations.pop(-1)
        assert len(observations) == len(
            waypoints), f"observations and waypoints should have the same length, {len(observations)} and {len(waypoints)}"
        return observations, waypoints, success

    @staticmethod
    def open_laptop(env, target_entity_name):
        laptop = env.task.entities[target_entity_name]
        start_pos, start_quat = env.robot.get_end_effector_pos(env.physics), env.robot.get_end_effector_quat(
            env.physics)

        trajectory = laptop.get_open_trajectory(env.physics)
        trajectory_quats = []
        screen_joint = laptop.screen_joint
        rotation_axis = env.physics.bind(screen_joint).xaxis
        # rotation_anchor = env.physics.bind(door_joint).xanchor
        observations = [env.get_observation()]
        waypoints = []
        stage_success = False
        task_success = False
        for i in range(len(trajectory)):
            rot_quat = quaternion_from_axis_angle(rotation_axis, 0.04 * (i + 1))
            new_quat = quaternion_multiply(start_quat, rot_quat)
            trajectory_quats.append(new_quat)
        # init_qpos = np.array(env.robot.get_qpos(env.physics)).reshape(-1)
        interplate_path, interplate_quat = interpolate_path(trajectory, trajectory_quats)
        new_obs, new_waypoints, _, task_success = SkillLib.step_trajectory(env,
                                                                           interplate_path,
                                                                           interplate_quat,
                                                                           np.zeros(2))
        observations.extend(new_obs)
        waypoints.extend(new_waypoints)
        if task_success:
            observations.pop(-1)
            assert len(observations) == len(
                waypoints), f"observations and waypoints should have the same length, {len(observations)} and {len(waypoints)}"
            return observations, waypoints, True, task_success

        new_obs, new_waypoints, _, task_success = SkillLib.open_gripper(env)
        observations.extend(new_obs)
        waypoints.extend(new_waypoints)
        observations.pop(-1)
        assert len(observations) == len(
            waypoints), f"observations and waypoints should have the same length, {len(observations)} and {len(waypoints)}"
        if env.task.entities[target_entity_name].is_open(env.physics):
            stage_success = True
        return observations, waypoints, stage_success, task_success

    @staticmethod
    def wait(env, wait_time=20, gripper_state=None):
        # print(f"wait for {wait_time} steps")
        current_qpos = np.array(env.robot.get_qpos(env.physics)).reshape(-1)
        if gripper_state is None:
            gripper_closed = env.robot.get_ee_open_state(env.physics)
            if gripper_closed:
                gripper_state = np.zeros(2)
            else:
                gripper_state = np.ones(2) * 0.04
        observations = [env.get_observation()]
        waypoints = []
        task_success = False
        start_time = time.time()
        for _ in range(wait_time):
            # print(f"iteration {_}")
            if len(current_qpos) == 7:
                action = np.concatenate([current_qpos[:7], gripper_state])

            elif len(current_qpos) == 11:
                action = np.concatenate([current_qpos[:7], gripper_state, current_qpos[7:]])
            timestep = env.step(action)
            if timestep.last():
                print(f"第{_}次判定成功")
                task_success = True
                break
            obs = safe_get_observation(env, observations[-1])
            observations.append(obs)
            waypoints.append(np.concatenate([env.robot.get_end_effector_pos(env.physics),
                                             quaternion_to_euler(env.robot.get_end_effector_quat(env.physics)),
                                             gripper_state]))
        end_time = time.time()
        print(f"wait for{start_time - end_time} s")
        observations.pop(-1)
        assert len(observations) == len(
            waypoints), f"observations and waypoints should have the same length, {len(observations)} and {len(waypoints)}"
        return observations, waypoints, True, task_success

    @staticmethod
    def end(env, wait_time=1, gripper_state=None):
        print(f"wait for {wait_time} steps")
        current_qpos = np.array(env.robot.get_qpos(env.physics)).reshape(-1)

        if gripper_state is None:
            gripper_closed = env.robot.get_ee_open_state(env.physics)
            if gripper_closed:
                gripper_state = np.zeros(2)
            else:
                gripper_state = np.ones(2) * 0.04
        observations = [env.get_observation()]
        waypoints = []
        task_success = False
        start_time = time.time()
        env.task.skill_end = True

        for _ in range(wait_time):
            action = np.concatenate([current_qpos[:7], gripper_state, current_qpos[7:]])
            print("action:", action)
            timestep = env.step(action)
            if timestep.last():
                print(f"第{_}次判定成功")
                task_success = True
                break
            obs = safe_get_observation(env, observations[-1])
            observations.append(obs)
            waypoints.append(np.concatenate([env.robot.get_end_effector_pos(env.physics),
                                             quaternion_to_euler(env.robot.get_end_effector_quat(env.physics)),
                                             gripper_state]))
        end_time = time.time()
        print(f"wait for{start_time - end_time} s")
        observations.pop(-1)
        assert len(observations) == len(
            waypoints), f"observations and waypoints should have the same length, {len(observations)} and {len(waypoints)}"
        return observations, waypoints, True, task_success

    @staticmethod
    def move_offset(env, offset, target_quat=None, gripper_state=None):
        start_pos, start_quat = env.robot.get_end_effector_pos(env.physics), env.robot.get_end_effector_quat(
            env.physics)
        target_pos = np.array(start_pos) + np.array(offset)
        if target_quat is None: target_quat = start_quat
        observations, waypoints, stage_success, task_success = SkillLib.moveto(env, target_pos, target_quat,
                                                                               gripper_state=gripper_state)
        return observations, waypoints, stage_success, task_success