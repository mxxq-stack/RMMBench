"""
Skill Library for data generation.
"""
import math
import os
import random
import time

import cv2
import numpy as np

from RMMBench.algorithms.motion_planning.rrt import rrt_motion_planning
from RMMBench.algorithms.utils import interpolate_path, qauternion_slerp, interpolate_pos, interpolate_radians
from RMMBench.utils.utils import euler_to_quaternion,degrees_to_radians_builtin, compute_lmax_hanan_grid, find_keypoint_and_prepare_grasp, \
    find_interactive_site_and_prepare, distance, quaternion_to_euler, quaternion_from_axis_angle, quaternion_multiply, \
    get_placement_top_down_view, quaternion_to_matrix, extract_base_name


PRIOR_EULERS=[
[0, -np.pi/2, 0],
]


class SkillLib:

    @staticmethod
    def moveforward(env, gripper_state=None, forward=0, side=0, yaw=0, target_velocity=0.1, tolerance=0.01,
                    target_object_info=None,
                    disturbance_points=None,
                    L_th=1.5,
                    l_default=0,
                    robot_radius=0.2):

        """
        [0, forward, side, yaw]
        Hanan-grid-based travel distance decision logic.
        Each moveforward execution moves the robot along the Hanan grid, advancing
        in the current heading (one of the four directions).
        """
        _,_,_,_= SkillLib.observe(env)

        target_object_info = env.task.config_manager.target_object_info

        current_robot_info = env.robot.get_link_base_info(env.physics)
        init_robot_info = env.task.init_robot_info

        curr_yaw = current_robot_info["euler"][2]
        init_yaw = init_robot_info["euler"][2]

        relative_yaw = (curr_yaw - init_yaw + math.pi) % (2 * math.pi) - math.pi

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

        obstacle_bboxes = []
        boundary = None
        if hasattr(env.task, 'get_robocasa_scene_class') and env.task.get_robocasa_scene_class is not None:
            navigation_bbox_info = env.task.get_robocasa_scene_class.get_obstacles_bbox(env.physics)
            obstacle_bboxes = navigation_bbox_info.get("bbox_list", [])
            boundary = navigation_bbox_info.get("boundary", [])

        move_direction = None
        if abs(relative_yaw) < 0.1:
            move_direction = '+x'
        elif abs(abs(relative_yaw) - 3.14) < 0.1:
            move_direction = '-x'
        elif abs(relative_yaw - 1.57) < 0.1:
            move_direction = '+y'
        elif abs(relative_yaw + 1.57) < 0.1:
            move_direction = '-y'

        if move_direction is not None:
            current_pos = [current_robot_info["position"][0], current_robot_info["position"][1]]

            stay_points_filtered = []
            for sp in stay_points:
                manhattan_dist = abs(current_pos[0] - sp[0]) + abs(current_pos[1] - sp[1])
                if manhattan_dist > robot_radius * 2:
                    stay_points_filtered.append(sp)

            world_move_direction = None
            if abs(curr_yaw) < 0.1 or abs(curr_yaw - 2 * math.pi) < 0.1:
                world_move_direction = '+x'
            elif abs(abs(curr_yaw) - math.pi) < 0.1:
                world_move_direction = '-x'
            elif abs(curr_yaw - math.pi / 2) < 0.1:
                world_move_direction = '+y'
            elif abs(curr_yaw + math.pi / 2) < 0.1:
                world_move_direction = '-y'


            if world_move_direction is not None:
                Lmax = compute_lmax_hanan_grid(current_pos, world_move_direction, obstacle_bboxes,
                                               stay_points_filtered, robot_radius, boundary)
            else:
                Lmax = float('inf')


            if Lmax == float('inf') or Lmax > 100:
                actual_dist = l_default
            elif Lmax == 0:
                actual_dist = 0
            elif Lmax < L_th:
                actual_dist = max(0, Lmax)
            else:
                actual_dist = Lmax / 2


            if move_direction == '+x':
                forward = actual_dist
            elif move_direction == '-x':
                forward = -actual_dist
            elif move_direction == '+y':
                side = -actual_dist
            elif move_direction == '-y':
                side = actual_dist
        current_euler = current_robot_info["euler"]

        base_current_pos = env.robot.get_mobilebase_distance(env.physics)

        move_distance = round(max(abs(forward), abs(side)), 2)


        base_current_pos_reset_look = env.robot.get_mobilebase_distance(env.physics)
        base_target_pos = list(np.array(base_current_pos_reset_look) + np.array([0, 0, forward, side, yaw]))

        arm_current_angles = env.robot.get_qpos(env.physics)[:7]

        gripper_closed = env.robot.get_ee_open_state(env.physics)
        if gripper_state is None:
            if gripper_closed:
                gripper_state = np.zeros(2)
            else:
                gripper_state = np.ones(2) * 0.04

        observations = [env.get_observation()]
        waypoints = []
        task_success = False

        current_action = np.concatenate([arm_current_angles, gripper_state, base_current_pos])

        base_path = [base_current_pos[1:4], np.array(base_target_pos[1:4])]

        interplate_base_path = interpolate_pos(base_path, target_velocity)

        inerplate_path = []
        for _ in range(len(interplate_base_path)):

            point = np.concatenate([current_action[:10], interplate_base_path[_], [current_action[-1]]])


            timestep = env.step(point)
            if timestep.last():
                task_success = True
                break
            obs = env.get_observation()
            observations.append(obs)
            waypoints.append(point)

        stage_success = True
        end_robot_info = env.robot.get_link_base_info(env.physics)
        if "y" not in  move_direction:
            true_distance = current_robot_info["position"][1]-end_robot_info["position"][1]
        else:
            true_distance = current_robot_info["position"][0]-end_robot_info["position"][0]

        return observations, waypoints, stage_success, task_success,move_distance


    @staticmethod
    def rotate_right(env, gripper_state=None, foward=0, side=0, yaw=-1.57, target_velocity=0.157, tolerance=0.01,
                     rotation_center_offset=0.208):
        _,_,_,_= SkillLib.observe(env)
        init_base_position = env.robot.get_link_base_info(env.physics)
        init_qpos = env.task.robot.default_qpos
        link_init_pos = init_qpos[:7]

        base_current_pos = env.robot.get_mobilebase_distance_continue_look(env.physics)
        current_yaw = base_current_pos[-1]


        target_yaw = current_yaw + yaw


        cos_curr = math.cos(current_yaw)
        sin_curr = math.sin(current_yaw)
        center_x_curr = rotation_center_offset * cos_curr
        center_y_curr = rotation_center_offset * sin_curr

        cos_target = math.cos(target_yaw)
        sin_target = math.sin(target_yaw)
        center_x_target = rotation_center_offset * cos_target
        center_y_target = rotation_center_offset * sin_target

        compensate_forward = center_x_target - center_x_curr
        compensate_side = center_y_target - center_y_curr

        base_current_pos_reset_look=env.robot.get_mobilebase_distance(env.physics)
        base_target_pos = base_current_pos_reset_look + [0, 0,compensate_forward + foward, compensate_side + side, yaw]

        arm_current_angles = env.robot.get_qpos(env.physics)[:7]

        gripper_closed = env.robot.get_ee_open_state(env.physics)
        if gripper_state is None:
            if gripper_closed:
                gripper_state = np.zeros(2)
            else:
                gripper_state = np.ones(2) * 0.04

        observations = [env.get_observation()]
        waypoints = []
        task_success = False


        current_action = np.concatenate([link_init_pos, gripper_state, base_current_pos])

        base_radians = [base_current_pos[-1], base_target_pos[-1]]
        interplate_base_path = interpolate_radians(base_radians, target_velocity)

        n_steps = len(interplate_base_path)
        start_val = base_current_pos[1]
        end_val = base_target_pos[1]
        resampled_path = np.linspace(start_val, end_val, num=n_steps).tolist()

        for i in range(n_steps):
            t = i / (n_steps - 1) if n_steps > 1 else 1.0
            interp_yaw = current_yaw + yaw * t
            cos_interp = math.cos(interp_yaw)
            sin_interp = math.sin(interp_yaw)
            center_x_interp = rotation_center_offset * cos_interp
            center_y_interp = rotation_center_offset * sin_interp

            step_comp_forward = -(center_x_interp - center_x_curr)
            step_comp_side = -(center_y_interp - center_y_curr)

            base_action = [base_current_pos[0],
                           resampled_path[i],
                           base_current_pos[2] + step_comp_forward + foward * t,
                           base_current_pos[3] - step_comp_side + side * t,
                           interplate_base_path[i]]


            point = np.concatenate([current_action[:9], base_action])

            timestep = env.step(point)
            if timestep.last():
                task_success = True
                break
            obs = env.get_observation()
            observations.append(obs)
            waypoints.append(point)
        if distance(interplate_base_path[-1], env.robot.get_mobilebase_distance(env.physics)[-1]) < tolerance:
            stage_success = True
        stage_success = True

        return observations, waypoints, stage_success, task_success

    @staticmethod
    def look_right(env, gripper_state=None,  yaw=45, target_velocity=0.157, tolerance=0.01):
        yaw = -degrees_to_radians_builtin(yaw)


        init_qpos = env.task.robot.default_qpos
        link_init_pos = init_qpos[:7]
        base_current_pos = env.robot.get_mobilebase_distance_continue_look(env.physics)
        base_target_pos = base_current_pos + [0, yaw, 0,0, 0]
        arm_current_angles = env.robot.get_qpos(env.physics)[:7]
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

        interplate_base_path = interpolate_radians(base_radians, target_velocity)
        inerplate_path = []
        for _ in range(len(interplate_base_path)):
            point = np.concatenate([current_action[:10], [interplate_base_path[_]],current_action[11:]])
            timestep = env.step(point)
            if timestep.last():
                task_success = True
                break
            obs = env.get_observation()
            observations.append(obs)
            waypoints.append(point)
        curr = env.robot.get_mobilebase_distance_continue_look(env.physics)
        if distance(interplate_base_path[-1], env.robot.get_mobilebase_distance_continue_look(env.physics)[-1]) < tolerance:
            stage_success = True
        stage_success = True
        return observations, waypoints, stage_success, task_success


    @staticmethod
    def look_left(env, gripper_state=None, yaw=45, target_velocity=0.157,tolerance=0.01):
        return SkillLib.look_right(env,
           gripper_state=gripper_state,
           yaw=-yaw,
           target_velocity=target_velocity,
           tolerance=tolerance)

    @staticmethod
    def look_forward(env, gripper_state=None, yaw=0, target_velocity=0.157,tolerance=0.01):
        init_qpos = env.task.robot.default_qpos
        link_init_pos = init_qpos[:7]
        base_current_pos = env.robot.get_mobilebase_distance(env.physics)

        base_target_pos = base_current_pos + [0, 0, 0, 0, 0]
        base_target_pos[1]=0

        arm_current_angles = env.robot.get_qpos(env.physics)[:7]
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

        interplate_base_path = interpolate_radians(base_radians, target_velocity)
        inerplate_path = []
        for _ in range(len(interplate_base_path)):
            point = np.concatenate([current_action[:10], [interplate_base_path[_]], current_action[11:]])
            timestep = env.step(point)
            if timestep.last():
                task_success = True
                break
            obs = env.get_observation()
            observations.append(obs)
            waypoints.append(point)
        curr = env.robot.get_mobilebase_distance(env.physics)
        if distance(interplate_base_path[-1], env.robot.get_mobilebase_distance(env.physics)[-1]) < tolerance:
            stage_success = True
        stage_success = True
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
            if task_success:
                break
            obs = env.get_observation()
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
        start_pos, start_quat = env.robot.get_end_effector_pos(env.physics), env.robot.get_end_effector_quat(env.physics)
        observations = [env.get_observation()]
        waypoints = []
        task_success = False
        gripper_closed = env.robot.get_ee_open_state(env.physics)
        if gripper_state is None:
            if gripper_closed: gripper_state = np.zeros(2)
            else: gripper_state = np.ones(2) * 0.04

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
        observations.extend(new_obs)
        waypoints.extend(new_waypoints)
        observations.pop(-1)
        assert len(observations) == len(waypoints), f"observations and waypoints should have the same length, {len(observations)} and {len(waypoints)}"
        return observations, waypoints, stage_success, task_success

    @staticmethod
    def _vis_gripper(rgb_img, p_grasp_cam, r_grasp_cam, intrinsic, gripper_w=0.08, gripper_d=0.035, color=(0, 255, 0)):
        """Overlay the gripper wireframe onto rgb_img (RGB format) and return the overlaid image."""
        img_viz = rgb_img.copy()
        hw = gripper_w / 2
        depth_base = 0.02
        pts_local = np.array([
            [0,   0, 0],
            [ hw, 0, -depth_base],
            [-hw, 0, -depth_base],
            [ hw, 0, -depth_base + gripper_d],
            [-hw, 0, -depth_base + gripper_d],
        ])
        pts_cam = (r_grasp_cam @ pts_local.T).T + p_grasp_cam
        pts_2d_homo = (intrinsic @ pts_cam.T).T
        pts_2d = (pts_2d_homo[:, :2] / pts_2d_homo[:, 2:3]).astype(int)
        cv2.line(img_viz, tuple(pts_2d[1]), tuple(pts_2d[2]), color, 2)
        cv2.line(img_viz, tuple(pts_2d[1]), tuple(pts_2d[3]), color, 2)
        cv2.line(img_viz, tuple(pts_2d[2]), tuple(pts_2d[4]), color, 2)
        return img_viz

    @staticmethod
    def visualize_grasp_poses(env, target_entity_name, base_yaw,
                              grasp_direction="vertical",
                              specific_keypoint=None,
                              keypoint_offset=0.02,
                              save_path=None):
        """
        Construct a world-frame rotation matrix directly with cross products, visualize the
        current grasp pose, overlay it on the wrist camera image, and save it.
        The axis definitions align naturally with _vis_gripper's local frame; no extra correction needed.

        Args:
            base_yaw: Robot chassis heading (radians), taken from euler[2] of get_link_base_info.
            grasp_direction: "vertical" or "horizontal".
            keypoint_offset: Distance (meters) to offset key_pos along the pre-grasp direction
                             (opposite the approach direction); upward for vertical,
                             toward the robot for horizontal.
            save_path: Image save path; if None, do not save and return the image array.
        """
        rgb_img = env.render(camera_id=3, height=480, width=480)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        _, _, _, intr, extr = env.get_rgbd(3)

        R_W_C = extr[:3, :3]
        t_W_C = extr[:3, 3]
        R_mjt_to_ocv = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        R_world_to_cam = R_mjt_to_ocv @ R_W_C.T

        target_entity = env.task.entities[target_entity_name]
        keypoints = target_entity.get_grasped_keypoints(env.physics)
        idx = specific_keypoint if specific_keypoint is not None else 0
        key_pos = np.array(keypoints[idx])

        if grasp_direction == "vertical":
            z_axis = np.array([0, 0, -1])
            y_axis = np.array([-np.cos(base_yaw), -np.sin(base_yaw), 0])
            y_axis /= np.linalg.norm(y_axis)
            x_axis = np.cross(y_axis, z_axis)
            x_axis /= np.linalg.norm(x_axis)
        else:
            z_axis = np.array([np.cos(base_yaw), np.sin(base_yaw), 0])
            z_axis /= np.linalg.norm(z_axis)
            y_axis = np.array([0, 0, 1])
            x_axis = np.cross(y_axis, z_axis)
            x_axis /= np.linalg.norm(x_axis)
        R_world_gripper = np.column_stack([x_axis, y_axis, z_axis])

        key_pos_vis = key_pos + (-z_axis) * keypoint_offset

        p_cam = R_world_to_cam @ (key_pos_vis - t_W_C)
        r_cam = R_world_to_cam @ R_world_gripper

        gripper_d = 0.035 if grasp_direction == "vertical" else 0.035 + 0.015
        img = SkillLib._vis_gripper(rgb_img.copy(), p_cam, r_cam, intr, gripper_d=gripper_d, color=(0, 255, 0))

        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        return img

    @staticmethod
    def pick(env,
             target_entity_name=None,
             target_pos=None,
             target_quat=None,
             prepare_distance=-0.15,
             grasp_direction=None,
             specific_keypoint=None,
             target_velocity=0.05,
             motion_planning_kwargs=dict(),
             body_name=None,
             use_graspnet=False,
             topk_grasps=2,
             grasp_seed=None,
             vlm_handler=None,
             **kwargs):
        """
        Keypoint-based pick skill (no GraspNet).

        Flow:
          1. Dynamically generate prior_eulers from base_yaw + grasp_direction
          2. find_keypoint_and_prepare_grasp → key_pos, prepare_pos, key_quat
          3. RRT: start → prepare_pos  (orientation slerped from start_quat to key_quat)
          4. Straight line: prepare_pos → key_pos  (keep key_quat, pure translation approach)
          5. close_gripper
          6. lift

        Args:
            grasp_direction: "vertical" (vertically downward) or "horizontal" (horizontal, facing straight ahead of the robot)
            site_name: full site name of an interactive manipulation point (e.g. knob); when passed, target_entity_name is not needed
            body_name: body-level name (e.g. "knob_rear_left"); converted internally to the corresponding site_name

        Returns: observations, waypoints, stage_success, task_success
        """

        explicit_direction = grasp_direction is not None
        if grasp_direction is None:
            grasp_direction = "vertical"

        env.update_pcd_generator()

        graspnet_pool = []
        if use_graspnet:
            from RMMBench.utils.graspnet_utils import graspnet_candidates
            graspnet_pool, is_cloud = graspnet_candidates(
                env, bbox=None, topk=topk_grasps, prepare_quat=None,
                grasp_seed=grasp_seed,
            )
            graspnet_viz_list = [c["viz"] for c in graspnet_pool]

        pure_body_name = None
        if body_name is not None and target_entity_name is None:
            if "/" in body_name:
                target_entity_name, pure_body_name = body_name.split("/", 1)
            else:
                pure_body_name = body_name
                for name, entity in env.task.entities.items():
                    if hasattr(entity, 'get_grasped_keypoints'):
                        keypoints = entity.get_grasped_keypoints(env.physics, body_name)
                        if len(keypoints) > 0:
                            target_entity_name = name
                            break

        if target_entity_name is not None:
            target_entity = env.task.entities[target_entity_name]
            entity_mjcf = target_entity.mjcf_model.worldbody
            entity_pos = env.physics.bind(entity_mjcf).xpos

            grasp_direction = getattr(target_entity, 'grasp_approach', grasp_direction)
            if hasattr(target_entity, 'grasp_direction'):
                grasp_direction = target_entity.grasp_direction

            if not explicit_direction:
                reach_containers = [
                    entity for entity in env.task.entities.values()
                    if type(entity).__name__ in SkillLib.HORIZONTAL_REACH_CONTAINERS
                ]
                if reach_containers:
                    for container in reach_containers:
                        if container is target_entity:
                            continue
                        if container.contain(entity_pos, env.physics):
                            grasp_direction = "horizontal"
                            break

            base_yaw = env.robot.get_link_base_info(env.physics)['euler'][2]

            if grasp_direction == "vertical":
                prior_eulers = [
                    [np.pi, 0, base_yaw - np.pi],
                ]
            else:
                prior_eulers = [
                    [base_yaw - np.pi, -np.pi / 2, 0],
                ]

            if target_pos is None or target_quat is None:
                key_pos, prepare_key_pos, key_quat = find_keypoint_and_prepare_grasp(
                    env, target_entity, prior_eulers,
                    specific_keypoint_id=specific_keypoint,
                    prepare_distance=prepare_distance,
                    base_yaw=base_yaw,
                    body_name=pure_body_name,
                    grasp_direction=grasp_direction,
                )
                if key_pos is None or prepare_key_pos is None:
                    return None
            else:
                key_pos, key_quat = target_pos, target_quat
                _, move_vec = env.robot.gripper_pcd(key_pos, key_quat)
                prepare_key_pos = key_pos + move_vec * prepare_distance
        else:
            raise ValueError("Either target_entity_name or body_name must be provided")

        start_pos  = np.array(env.robot.get_end_effector_pos(env.physics))
        start_quat = np.array(env.robot.get_end_effector_quat(env.physics))
        key_quat   = np.array(key_quat)
        prepare_pos = np.array(prepare_key_pos)
        key_pos     = np.array(key_pos)

        expert_viz_list = []
        if use_graspnet:
            expert_viz = SkillLib.visualize_grasp_poses(
                env, target_entity_name, base_yaw,
                grasp_direction=grasp_direction,
                specific_keypoint=specific_keypoint,
            )
            expert_viz_list = [expert_viz]

        if use_graspnet:
            candidates = list(graspnet_pool) + [{
                "source": "keypoint",
                "key_pos": key_pos, "prepare_pos": prepare_pos, "key_quat": key_quat,
            }]
            viz_list = graspnet_viz_list + expert_viz_list

            grasp_choice = None
            if vlm_handler is not None and len(candidates) > 0:
                grasp_choice, grasp_content = vlm_handler.request_grasp(
                    viz_list, extract_base_name(target_entity_name))
                if grasp_choice is not None and not (1 <= grasp_choice <= len(candidates)):
                    grasp_choice = None

            chosen = candidates[grasp_choice - 1] if grasp_choice is not None else candidates[-1]

            key_pos     = np.array(chosen["key_pos"])
            prepare_pos = np.array(chosen["prepare_pos"])
            key_quat    = np.array(chosen["key_quat"])


        obstacle_pcd = np.asarray(env.get_obstacle_pcd().points)


        current_robot_info = env.robot.get_link_base_info(env.physics)
        curr_robot_pos = current_robot_info["position"]
        init2prepare_path = rrt_motion_planning(
            tuple(start_pos), tuple(prepare_pos),
            obstacle_pcd, object_pos=None,
            robot_pos=curr_robot_pos,
            **motion_planning_kwargs,
        )
        if init2prepare_path is None:
            init2prepare_path = [start_pos, prepare_pos]

        n = len(init2prepare_path)
        quats_in_path = [
            qauternion_slerp(start_quat, key_quat, t)
            for t in np.linspace(0, 1, n, endpoint=True)
        ]

        init2prepare_path.append(tuple(key_pos))
        quats_in_path.append(key_quat)

        path = np.array(init2prepare_path)
        interplate_path, interplate_quat = interpolate_path(path, quats_in_path, target_velocity)

        observations = [env.get_observation()]
        waypoints    = []
        stage_success = False
        task_success  = False

        gripper_open = np.ones(2) * 0.08
        new_obs, new_waypoints, _, task_success = SkillLib.step_trajectory(
            env, interplate_path, interplate_quat, gripper_open, **kwargs
        )
        observations.extend(new_obs)
        waypoints.extend(new_waypoints)

        new_obs, new_waypoints, _, task_success = SkillLib.close_gripper(env)
        observations.extend(new_obs)
        waypoints.extend(new_waypoints)


        observations.pop(-1)
        assert len(observations) == len(waypoints), (
            f"observations and waypoints length mismatch: {len(observations)} vs {len(waypoints)}"
        )


        return observations, waypoints, stage_success, task_success

    HORIZONTAL_PLACE_CONTAINERS = ["shelf_1", "shelf_2", "big_fridge", "microwave"]

    HORIZONTAL_REACH_CONTAINERS = ("Shelf", "Fridge", "Microwave")

    @staticmethod
    def place(env,
              target_container_name=None,
              bbox=None,
              target_pos=None,
              target_quat=None,
              placement_th=0,
              lift_height=0.1,
              open_bbox=1,
              motion_planning_kwargs=dict(),
              place_direction=None,
              prepare_distance=-0.15):
        """
        general place function for data generation
        param:
            env: LM4manipEnv object
            target_entity_name: str, target entity name
            target_pos: np.array, target position. If None, will propose a target position automatically
            target_quat: np.array, target quaternion. If None, will propose a target quaternion automatically
            place_direction: str, "vertical" (vertical placement) or "horizontal" (horizontal placement).
                            If None, determined automatically by whether the target container name is in
                            the HORIZONTAL_PLACE_CONTAINERS list
        return:
            observations: list of obs
            waypoints: list of actions
            key_frame: list of key action such as move to prepare point, grasp
        """
        env.update_pcd_generator()
        current_robot_info = env.robot.get_link_base_info(env.physics)
        curr_robot_pos = current_robot_info["position"]

        if isinstance(target_container_name, list):
            target_container_name = random.choice(target_container_name)

        if place_direction is None:
            if target_container_name is not None and any(
                    keyword in target_container_name for keyword in SkillLib.HORIZONTAL_PLACE_CONTAINERS):
                place_direction = "horizontal"
            else:
                place_direction = "vertical"

        base_yaw = env.robot.get_link_base_info(env.physics)['euler'][2]
        if place_direction == "horizontal":
            prior_eulers = [
                [base_yaw - np.pi , np.pi / 2, 0],
            ]
        else:
            prior_eulers = [
                [np.pi, 0, base_yaw - np.pi],
            ]
        target_euler = prior_eulers[0]
        target_quat_from_direction = np.array(euler_to_quaternion(*target_euler))

        if target_container_name is not None and target_container_name in env.task.entities:

            target_container = env.task.entities[target_container_name]
        start_pos, start_quat = env.robot.get_end_effector_pos(env.physics), env.robot.get_end_effector_quat(
            env.physics)


        if bbox is not None:
            if target_container_name is not None:
                for fixture in ["sink"]:
                    if fixture in target_container_name:
                        placement_th += 0.25
            place_point = get_placement_top_down_view(env, bbox=bbox, cam_id=5, placement_offset=placement_th,target_container_name=target_container_name)
        elif target_pos is None:
            place_points = target_container.get_place_point(env.physics)
            place_point = random.choice(place_points)+[0,0,placement_th]
        else:
            place_point = np.array(target_pos)+[0,0,placement_th]
        target_pos = place_point

        if target_quat is None:
            target_quat = target_quat_from_direction

        obstacle_pcd = np.asarray(env.get_obstacle_pcd().points)
        start_pos, start_quat, target_pos, target_quat = np.array(start_pos), np.array(start_quat), np.array(
            target_pos), np.array(target_quat)

        link7_pos = env.robot.get_link7_obstacle_point_pos(env.physics)
        if link7_pos is not None:
            offset_at_start = link7_pos - start_pos
            R_start = quaternion_to_matrix(start_quat)
            R_target = quaternion_to_matrix(target_quat)
            link7_offset = R_target @ R_start.T @ offset_at_start
        else:
            link7_offset = None

        init2target_path = rrt_motion_planning(tuple(start_pos),
                                               tuple(target_pos),
                                               obstacle_pcd,
                                               link7_offset=None,
                                               q=0.05,
                                               r=0.02,
                                               max_samples=1024,
                                               prc=0.1,
                                               robot_pos = curr_robot_pos,
                                               **motion_planning_kwargs)
        offset = env.robot.ee_offset(env.physics)
        if init2target_path is None:
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
    def open_door(env,
                  target_container_name="small_fridge"):
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
        start_pos, start_quat = env.robot.get_end_effector_pos(env.physics), env.robot.get_end_effector_quat(env.physics)

        trajectory = target_container.get_open_trajectory(env.physics)
        trajectory_quats = []
        door_joint = target_container.door_joint
        rotation_axis = env.physics.bind(door_joint).xaxis
        observations = [env.get_observation()]
        waypoints = []
        stage_success = False
        task_success = False
        for i in range(len(trajectory)):
            rot_quat = quaternion_from_axis_angle(rotation_axis, -0.1*(i+1))
            new_quat = quaternion_multiply(start_quat, rot_quat)
            trajectory_quats.append(new_quat)
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
            obs = env.get_observation()
            observations.append(obs)
            waypoints.append(np.concatenate([pos, quaternion_to_euler(quat), np.ones(2)*0.04]))
        observations.pop(-1)
        assert len(observations) == len(waypoints), f"observations and waypoints should have the same length, {len(observations)} and {len(waypoints)}"
        if env.task.entities[target_container_name].is_open(env.physics):
            stage_success = True
        return observations, waypoints, stage_success, task_success

    @staticmethod
    def close_door(env, target_container_name, gripper_state=np.zeros(2)):
        target_container = env.task.entities[target_container_name]
        start_pos, start_quat = env.robot.get_end_effector_pos(env.physics), env.robot.get_end_effector_quat(env.physics)

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
        assert len(observations) == len(waypoints), f"observations and waypoints should have the same length, {len(observations)} and {len(waypoints)}"
        if env.task.entities[target_container_name].is_closed(env.physics):
            stage_success = True
        return observations, waypoints, stage_success, task_success

    @staticmethod
    def open_drawer(env,
                    target_container_name,
                    pick_prior_eulers=[[-np.pi/2, 0, 0]],
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
        new_obs, new_waypoints, _, success_ = SkillLib.pick(env, target_container_name, prior_eulers=pick_prior_eulers, specific_keypoint=drawer_id)

        observations.extend(new_obs)
        waypoints.extend(new_waypoints)
        task_success = task_success or success_
        start_pos, start_quat = env.robot.get_end_effector_pos(env.physics), env.robot.get_end_effector_quat(env.physics)
        trajectory = target_container.get_drawer_open_trajectory(env.physics, drawer_id)
        trajectory_quats = [start_quat for _ in range(len(trajectory))]
        trajectory, trajectory_quats = interpolate_path(trajectory, trajectory_quats)
        new_obs, new_waypoints, _, success_ = SkillLib.step_trajectory(env, trajectory, trajectory_quats, np.zeros(2))

        observations.extend(new_obs)
        waypoints.extend(new_waypoints)
        task_success = task_success or success_
        observations.pop(-1)

        assert len(observations) == len(waypoints), f"observations and waypoints should have the same length, {len(observations)} and {len(waypoints)}"
        return observations, waypoints, True, task_success

    @staticmethod
    def press(env, target_pos, target_quat=None, move_vector=[0, 0, 0.1], max_n_substep=100):
        prepare_pos = target_pos + np.array(move_vector) if move_vector is not None else target_pos
        observations, waypoints, _, _ = SkillLib.moveto(env,
                                                     prepare_pos,
                                                     target_quat,
                                                     max_n_substep=max_n_substep)
        qpos = np.array(env.robot.get_qpos(env.physics)).reshape(-1)
        for i in range(10):
            gripper_state = np.ones(2) * (0.04 - i/10 * 0.04)
            action = np.concatenate([qpos, gripper_state])
            timestep = env.step(action)
            if timestep.last():
                break
            obs = env.get_observation()
            observations.append(obs)
            waypoints.append(np.concatenate([env.robot.get_end_effector_pos(env.physics),
                                             quaternion_to_euler(env.robot.get_end_effector_quat(env.physics)),
                                             gripper_state]))
        new_obs, new_waypoints, stage_success, task_success = SkillLib.moveto(env, target_pos, target_quat, max_n_substep=max_n_substep)
        observations.extend(new_obs)
        waypoints.extend(new_waypoints)
        assert len(observations) == len(waypoints), f"observations and waypoints should have the same length, {len(observations)} and {len(waypoints)}"
        return observations, waypoints, stage_success, task_success

    @staticmethod
    def pull(env, target_pos=None, target_quat=None, gripper_state=None, pull_distance=0.3):
        """
        Common pull function.
        """
        start_pos, start_quat = env.robot.get_end_effector_pos(env.physics), env.robot.get_end_effector_quat(env.physics)
        if target_pos is None:
            R_ee = quaternion_to_matrix(np.array(start_quat))
            ee_z_world = R_ee[:, 2]
            if abs(ee_z_world[2]) > 0.7:
                base_yaw = env.robot.get_link_base_info(env.physics)['euler'][2]
                forward_dir = np.array([np.cos(base_yaw), np.sin(base_yaw), 0])
            else:
                forward_dir = ee_z_world.copy()
                forward_dir[2] = 0
                forward_dir /= np.linalg.norm(forward_dir)
            target_pos = np.array(start_pos) - forward_dir * pull_distance
        if target_quat is None: target_quat = start_quat
        interplate_path, interplate_quat = interpolate_path([start_pos, target_pos], [np.array(start_quat), np.array(target_quat)])
        observations = [env.get_observation()]
        waypoints = []
        task_success = False
        if gripper_state is None:
            gripper_closed = env.robot.get_ee_open_state(env.physics)
            if gripper_closed: gripper_state = np.zeros(2)
            else: gripper_state = np.ones(2) * 0.04
        new_obs, new_waypoints, stage_success, task_success = SkillLib.step_trajectory(env,
                                                           interplate_path,
                                                           interplate_quat,
                                                           gripper_state)
        observations.extend(new_obs)
        waypoints.extend(new_waypoints)
        observations.pop(-1)
        assert len(observations) == len(waypoints), f"observations and waypoints should have the same length, {len(observations)} and {len(waypoints)}"
        return observations, waypoints, stage_success, task_success


    @staticmethod
    def push(env, target_pos=None, target_quat=None, gripper_state=None, push_distance=0.3):
        obs, waypoints, stage_success, task_success = SkillLib.pull(env, target_pos, target_quat, gripper_state, -push_distance)
        return obs, waypoints, stage_success, task_success


    @staticmethod
    def lift(env, target_pos=None, target_quat=None, gripper_state=None, lift_height=0.2):
        """
        Common lift function.
        """
        start_pos, start_quat = env.robot.get_end_effector_pos(env.physics), env.robot.get_end_effector_quat(env.physics)
        if target_pos is None:
            target_pos = np.array(start_pos) + np.array([0, 0, lift_height])
        if target_quat is None:
            target_quat = start_quat
        interplate_path, interplate_quat = interpolate_path([start_pos, target_pos], [np.array(start_quat), np.array(target_quat)])
        observations = [env.get_observation()]
        waypoints = []
        if gripper_state is None:
            gripper_closed = env.robot.get_ee_open_state(env.physics)
            if gripper_closed: gripper_state = np.zeros(2)
            else: gripper_state = np.ones(2) * 0.06
        obs, new_waypoints, stage_success, task_success = SkillLib.step_trajectory(env,
                                                       interplate_path,
                                                       interplate_quat,
                                                       gripper_state)
        observations.extend(obs)
        waypoints.extend(new_waypoints)
        observations.pop(-1)
        assert len(observations) == len(waypoints), f"observations and waypoints should have the same length, {len(observations)} and {len(waypoints)}"
        return observations, waypoints, stage_success, task_success

    @staticmethod
    def rotate_knob(env, angle=90, target_velocity=0.05, **kwargs):
        """
        Rotate a pinched knob by controlling the end effector to rotate about its local z axis.

        Prerequisite: the gripper is already pinched on the knob (post-pick, pre-lift state).

        Args:
            env: environment object
            angle: rotation angle in degrees; positive is clockwise, negative is counterclockwise
            target_velocity: trajectory execution speed

        Returns: observations, waypoints, False, False
        """
        current_pos = np.array(env.robot.get_end_effector_pos(env.physics))
        current_quat = np.array(env.robot.get_end_effector_quat(env.physics))

        rad = np.deg2rad(angle)
        z_axis = np.array([0, 0, 1])
        rot_quat = quaternion_from_axis_angle(z_axis, rad)

        target_quat = quaternion_multiply(current_quat, rot_quat)
        target_pos = current_pos

        steps = max(int(abs(angle) / 2), 10)
        quats = [
            qauternion_slerp(current_quat, target_quat, t)
            for t in np.linspace(0, 1, steps, endpoint=True)
        ]
        path = [current_pos.copy() for _ in range(len(quats))]

        interplate_path = np.array(path)
        interplate_quat = quats

        gripper_closed = np.zeros(2)
        observations = [env.get_observation()]
        waypoints = []
        obs, new_waypoints, _, _ = SkillLib.step_trajectory(
            env, interplate_path, interplate_quat, gripper_closed, **kwargs
        )
        observations.extend(obs)
        waypoints.extend(new_waypoints)

        observations.pop(-1)
        assert len(observations) == len(waypoints), (
            f"observations and waypoints length mismatch: {len(observations)} vs {len(waypoints)}"
        )

        return observations, waypoints, False, False

    @staticmethod
    def open_sink(env, target_entity_name, **kwargs):
        """
        Turn on the sink faucet.

        Prerequisite: the gripper is already pinched on the handle (post-pick, pre-lift state).

        Args:
            env: environment object
            target_entity_name: sink entity name, e.g. "sink_0"

        Returns: observations, waypoints, stage_success, task_success
        """
        observations = [env.get_observation()]
        waypoints = []
        stage_success = False
        task_success = False

        sink = env.task.entities.get(target_entity_name)
        if sink is None or not hasattr(sink, 'open_direction'):
            return observations, waypoints, stage_success, task_success

        open_direction = sink.open_direction

        if open_direction == "lift":
            new_obs, new_waypoints, _, _ = SkillLib.lift(
                env,
                gripper_state=np.zeros(2),
                lift_height=0.3
            )
            observations.extend(new_obs)
            waypoints.extend(new_waypoints)
        elif open_direction == "arc_down":
            return observations, waypoints, stage_success, task_success
        else:
            return observations, waypoints, stage_success, task_success

        if sink.is_open(env.physics):
            stage_success = True
        else:
            pass
        observations.pop(-1)
        assert len(observations) == len(waypoints), (
            f"observations and waypoints length mismatch: {len(observations)} vs {len(waypoints)}"
        )

        return observations, waypoints, stage_success, task_success

    @staticmethod
    def close_sink(env, target_entity_name, **kwargs):
        """
        Turn off the sink faucet.

        Prerequisite: the gripper is already pinched on the handle (post-pick, pre-lift state).

        Args:
            env: environment object
            target_entity_name: sink entity name, e.g. "sink_0"

        Returns: observations, waypoints, stage_success, task_success
        """
        observations = [env.get_observation()]
        waypoints = []
        stage_success = False
        task_success = False

        sink = env.task.entities.get(target_entity_name)
        if sink is None or not hasattr(sink, 'open_direction'):
            return observations, waypoints, stage_success, task_success

        open_direction = sink.open_direction

        if open_direction == "lift":
            new_obs, new_waypoints, _, _ = SkillLib.lift(
                env,
                gripper_state=np.zeros(2),
                lift_height=-0.06
            )
            observations.extend(new_obs)
            waypoints.extend(new_waypoints)
        elif open_direction == "arc_down":
            return observations, waypoints, stage_success, task_success
        else:
            return observations, waypoints, stage_success, task_success

        if not sink.is_open(env.physics):
            stage_success = True
        else:
            pass

        observations.pop(-1)
        assert len(observations) == len(waypoints), (
            f"observations and waypoints length mismatch: {len(observations)} vs {len(waypoints)}"
        )
        return observations, waypoints, stage_success, task_success

    @staticmethod
    def observe(env,

                tolerance=0.15,
                motion_planning_kwargs=dict()):
        """
        Return to the initial TCP
        """

        curr_qpos = env.robot.get_qpos(env.physics)[:7]
        init_qpos = env.robot.init_ee_qpos[:7]
        diff = np.abs(curr_qpos - init_qpos)
        all_close = np.all(diff < tolerance)
        if all_close:
            return [], [], True, True


        target_pos=env.robot.init_ee_pos
        target_quat=env.robot.init_ee_quat


        start_pos, start_quat = env.robot.get_end_effector_pos(env.physics), env.robot.get_end_effector_quat(
            env.physics)

        gripper_state = np.ones(2)*0.06
        ee_state = env.robot.get_ee_open_state(env.physics)
        if ee_state:
            gripper_state = np.zeros(2)

        obstacle_pcd = np.asarray(env.get_obstacle_pcd().points)

        start_pos, start_quat, target_pos, target_quat = np.array(start_pos), np.array(start_quat), np.array(
            target_pos), np.array(target_quat)
        current_robot_info = env.robot.get_link_base_info(env.physics)
        curr_robot_pos = current_robot_info["position"]

        init2target_path = rrt_motion_planning(tuple(start_pos),
                                               tuple(target_pos),
                                               obstacle_pcd,
                                               robot_pos=curr_robot_pos,
                                               **motion_planning_kwargs)
        offset = env.robot.ee_offset(env.physics)

        if init2target_path is None:
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

        curr_c_pos = np.array(env.robot.get_end_effector_pos(env.physics)).copy()

        new_obs, new_waypoints, _, task_success = SkillLib.step_trajectory(env,
                                                                           interplate_path,
                                                                           interplate_quat,
                                                                           gripper_state,
                                                                           end_dpos = [0, -1.04, 0, -2.56, 0, 1.55, -2.32])
        observations.extend(new_obs)
        waypoints.extend(new_waypoints)


        curr_c_pos = np.array(env.robot.get_end_effector_pos(env.physics)).copy()

        if task_success:
            observations.pop(-1)
            assert len(observations) == len(
                waypoints), f"observations and waypoints should have the same length, {len(observations)} and {len(waypoints)}"
            return observations, waypoints, True, task_success


        for entity in env.task.entities.values():
            if hasattr(entity, "is_grasped") and entity.is_grasped(env.physics, env.robot):
                stage_success = False
        return observations, waypoints, stage_success, task_success

    @staticmethod
    def close_gripper(env, repeat=1):
        qpos = np.array(env.robot.get_qpos(env.physics)).reshape(-1)

        observations = [env.get_observation()]
        waypoints = []
        success = False
        for i in range(15):
            gripper_state = np.ones(2) * (0.04 - i * 0.04 / 10)
            action = np.concatenate([qpos[:7], gripper_state, qpos[7:]])
            for _ in range(repeat):
                timestep = env.step(action)
                if timestep.last():
                    success = True
                    obs = env.get_observation()
                    observations.append(obs)
                    waypoints.append(np.concatenate([env.robot.get_end_effector_pos(env.physics),
                                                     quaternion_to_euler(env.robot.get_end_effector_quat(env.physics)),
                                                     gripper_state]))
                    break

            obs = env.get_observation()
            observations.append(obs)
            waypoints.append(np.concatenate([env.robot.get_end_effector_pos(env.physics),
                                             quaternion_to_euler(env.robot.get_end_effector_quat(env.physics)),
                                             gripper_state]))

        observations.pop(-1)
        assert len(observations) == len(
            waypoints), f"observations and waypoints should have the same length, {len(observations)} and {len(waypoints)}"
        return observations, waypoints, True, success

    @staticmethod
    def open_gripper(env, repeat=1):
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
        for i in range(10):
            gripper_state = np.ones(2) * (i+1)/10 * 0.06
            if len(qpos) > 8:
                action = np.concatenate([qpos[:7], gripper_state,qpos[7:]])
            else:
                action = np.concatenate([qpos, np.ones(2) * (0.04 / 10) * (i + 1)])

            for _ in range(repeat):
                timestep = env.step(action)
                if timestep.last():
                    task_success = True
                    obs = env.get_observation()
                    observations.append(obs)
                    waypoints.append(np.concatenate([env.robot.get_end_effector_pos(env.physics),
                                             quaternion_to_euler(env.robot.get_end_effector_quat(env.physics)),
                                             gripper_state]))
            obs = env.get_observation()
            observations.append(obs)
            waypoints.append(np.concatenate([env.robot.get_end_effector_pos(env.physics),
                                             quaternion_to_euler(env.robot.get_end_effector_quat(env.physics)),
                                             gripper_state]))
            if timestep.last():
                task_success = True
                break
        observations.pop(-1)

        assert len(observations) == len(waypoints), f"observations and waypoints should have the same length, {len(observations)} and {len(waypoints)}"
        if env.robot.get_ee_open_state(env.physics):
            stage_success = True
        return observations, waypoints, stage_success, task_success


    @staticmethod
    def wait(env, wait_time=20, gripper_state=None):
        current_qpos = np.array(env.robot.get_qpos(env.physics)).reshape(-1)
        if gripper_state is None:
            gripper_closed = env.robot.get_ee_open_state(env.physics)
            if gripper_closed: gripper_state = np.zeros(2)
            else: gripper_state = np.ones(2) * 0.04
        observations = [env.get_observation()]
        waypoints = []
        task_success = False
        start_time = time.time()
        for _ in range(wait_time):
            if len(current_qpos) == 7:
                action = np.concatenate([current_qpos[:7], gripper_state])

            elif len(current_qpos) == 11:
                action = np.concatenate([current_qpos[:7], gripper_state,current_qpos[7:]])
            timestep = env.step(action)
            if timestep.last():
                task_success = True
                break
            obs = env.get_observation()
            observations.append(obs)
            waypoints.append(np.concatenate([env.robot.get_end_effector_pos(env.physics),
                                             quaternion_to_euler(env.robot.get_end_effector_quat(env.physics)),
                                             gripper_state]))
        end_time = time.time()
        observations.pop(-1)
        assert len(observations) == len(waypoints), f"observations and waypoints should have the same length, {len(observations)} and {len(waypoints)}"
        return observations, waypoints, True, task_success

    @staticmethod
    def end(env, wait_time=1, gripper_state=None):
        current_qpos = np.array(env.robot.get_qpos(env.physics)).reshape(-1)
        env.task.skill_end = True
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
            action = np.concatenate([current_qpos[:7], gripper_state, current_qpos[7:]])

            timestep = env.step(action)
            if timestep.last():
                task_success = True
                break
            obs = env.get_observation()
            observations.append(obs)
            waypoints.append(np.concatenate([env.robot.get_end_effector_pos(env.physics),
                                             quaternion_to_euler(env.robot.get_end_effector_quat(env.physics)),
                                             gripper_state]))
        end_time = time.time()


        observations.pop(-1)
        assert len(observations) == len(
            waypoints), f"observations and waypoints should have the same length, {len(observations)} and {len(waypoints)}"
        return observations, waypoints, True, task_success


    @staticmethod
    def recall(handler, total_observations, step_idx, n_frames=16, keep_first_last=True):
        """
        [RECALL] Replay the full trajectory observations of a specified historical
        step (step_idx), uniformly sample frames, and call the VLM to extract keyframes.

        Args:
            handler: VLAMessageHandler instance, used to call the VLM (request_keyframe method).
            total_observations: dict, like {"step_0": [obs0, obs1, ...], "step_1": [...], ...}
                                 maintained by codelab_eval.py after each non-recall step is executed.
            step_idx: str, the VLM output target, e.g. "step_0". The historical step to replay.
            n_frames: int, target number of frames for uniform sampling (first/last kept),
                      default 16, adjustable.
            keep_first_last: bool, whether to force keeping the first and last frames
                             when sampling, default True.

        Returns:
            observations: [None]  -- placeholder, recall produces no real observations
            waypoints: [None]     -- placeholder, recall produces no real waypoints
            stage_success: bool   -- whether recall itself executed successfully (whether the
                                     trajectory could be fetched and the VLM called normally)
            task_success: bool    -- always False, recall does not mean the task is complete
            keyframes_info: dict or None -- keyframe information, for codelab_eval.py to store into
                            handler.recall_results[step_idx]; see handler.request_keyframe for the
                            structure. Returns None if step_idx does not exist or the trajectory is empty.
        """
        observations = [None]
        waypoints = [None]
        task_success = False

        if step_idx not in total_observations or not total_observations[step_idx]:
            return observations, waypoints, False, task_success, None

        obs_list = total_observations[step_idx]

        N = len(obs_list)
        if N <= n_frames:
            sample_indices = list(range(N))
        elif keep_first_last and n_frames >= 2:
            mid_count = n_frames - 2
            if mid_count > 0:
                mid_indices = np.linspace(1, N - 2, mid_count).astype(int).tolist()
            else:
                mid_indices = []
            sample_indices = sorted(set([0] + mid_indices + [N - 1]))
        else:
            sample_indices = np.linspace(0, N - 1, n_frames).astype(int).tolist()

        sampled_obs = [obs_list[i] for i in sample_indices]

        concat_frames = []
        wrist_frames = []
        head_frames = []
        for o in sampled_obs:
            if o is None:
                continue
            wrist_img = cv2.cvtColor(o["rgb"][3], cv2.COLOR_BGR2RGB)
            head_img = cv2.cvtColor(o["rgb"][5], cv2.COLOR_BGR2RGB)
            concat_img = np.hstack([wrist_img, head_img])
            concat_frames.append(concat_img)
            wrist_frames.append(wrist_img)
            head_frames.append(head_img)

        if len(concat_frames) == 0:
            return observations, waypoints, False, task_success, None

        try:
            keyframe_indices, reasoning = handler.request_keyframe(
                frames=concat_frames,
                step_idx=step_idx,
            )
        except Exception as e:
            return observations, waypoints, False, task_success, None

        if not keyframe_indices:
            keyframe_indices = [len(concat_frames) // 2]

        display_idx = keyframe_indices[0]
        if not (0 <= display_idx < len(concat_frames)):
            display_idx = len(concat_frames) // 2

        keyframes_info = {
            "step_idx": step_idx,
            "keyframe_indices": keyframe_indices,
            "reasoning": reasoning,
            "n_sampled_frames": len(concat_frames),
            "display_cam_wrist": wrist_frames[display_idx],
            "display_cam_head": head_frames[display_idx],
        }

        return observations, waypoints, True, task_success, keyframes_info
