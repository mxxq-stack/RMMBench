from RMMBench.utils.paths import PROJECT_ROOT
import heapq
import math
import re
import sys
import os

import numpy as np
import torch
import copy
from PIL import Image

from RMMBench.utils.depth2cloud import posRotMat2Mat
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import random
import copy
from scipy.spatial import cKDTree
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import logging
import colorlog

CAMERA_NAME_MAP = {
    "right": 0,
    "left": 1,
    "forward": 2,
    "wrist": 3,
    "base_opposite": 4,
    "head": 5,
}
CAMERA_ID_MAP = {v: k for k, v in CAMERA_NAME_MAP.items()}


def degrees_to_radians_builtin(degrees):
    """
    Convert degrees to radians using Python's built-in math library
    """
    return math.radians(degrees)

def is_target_out_of_reach(env, target_entity_name: str, robot_r: float = 0.9, body_name=None) -> bool:
    """
    Determine whether the target object is beyond the arm's horizontal operating range.

    Args:
        env: MuJoCo simulation environment object
        target_entity_name (str): entity name of the target object (e.g. "apple_0", "cheese_corn")
        robot_r (float): maximum horizontal operating radius of the arm (arm-length limit), default 0.8 m
        body_name (str, optional): if specified, look up the position of this body under the entity
            corresponding to target_entity_name

    Returns:
        bool: if xy_distance > robot_r, return True (out of operating range, navigation needed);
              otherwise return False (within operating range, grasping can be attempted).
    """
    robot_pos_raw = env.robot.get_link_base_info(env.physics)["position"]
    r_pos = np.array(robot_pos_raw)

    if body_name is not None:
        entity = env.task.entities.get(target_entity_name)
        if entity is not None:
            body = entity.mjcf_model.find("body", body_name)
            if body is not None:
                target_pos_raw = env.physics.bind(body).xpos
            else:
                return False
        else:
            return False
    elif target_entity_name in env.task.entities:
        target_pos_raw = env.physics.bind(env.task.entities[target_entity_name].mjcf_model.worldbody).xpos
    else:
        return False

    t_pos = np.array(target_pos_raw)

    xy_distance = np.linalg.norm(r_pos[:2] - t_pos[:2])

    return bool(xy_distance > robot_r)


def extract_base_name(name):
    if name is None:
        return ""
    name_str = str(name)
    return re.sub(r'_\d+$', '', name_str)

def xml_path_completion(xml_path, root=None):
    """
    Takes in a local xml path and returns a full path.
        if @xml_path is absolute, do nothing
        if @xml_path is not absolute, load xml that is shipped by the package

    Args:
        xml_path (str): local xml path
        root (str): root folder for xml path. If not specified defaults to robosuite.models.assets_root

    Returns:
        str: Full (absolute) xml path
    """
    if xml_path.startswith("/"):
        full_path = xml_path
    else:
        if root is None:
            root = os.path.join(PROJECT_ROOT, "assets")
        full_path = os.path.join(root, xml_path)
    return full_path

def normalize(v):
    return v / np.linalg.norm(v)

def compute_rotation_quaternion(camera_pos, target_pos, forward_axis=[1, 0, 0]):
    """
        Compute the ratation quaternion from camera position to target position
    """
    target_direction = np.array(target_pos) - np.array(camera_pos)
    target_direction = normalize(target_direction)

    base_forward = normalize(np.array(forward_axis))

    if np.allclose(target_direction, base_forward):
        return R.from_quat([0, 0, 0, 1])
    elif np.allclose(target_direction, -base_forward):
        orthogonal_axis = np.array([base_forward[1], -base_forward[0], 0])
        orthogonal_axis = normalize(orthogonal_axis)
        return R.from_rotvec(np.pi * orthogonal_axis).as_quat()
    else:
        axis = np.cross(base_forward, target_direction)
        axis = normalize(axis)
        angle = np.arccos(np.clip(np.dot(base_forward, target_direction), -1.0, 1.0))
        return R.from_rotvec(angle * axis).as_quat()

def euler_to_quaternion(roll, pitch, yaw):
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return (qw, qx, qy, qz)

def quaternion_to_euler(quat, is_degree=False):
    r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
    euler_angles = r.as_euler('xyz', degrees=is_degree)  
    return euler_angles

def matrix_to_quaternion(matrix):
    if matrix.shape == (9,):
        matrix = matrix.reshape(3, 3)
    r = R.from_matrix(matrix)
    quaternion = r.as_quat()
    quaternion = [quaternion[3], quaternion[0], quaternion[1], quaternion[2]]
    return quaternion

def quaternion_to_matrix(quat):
    r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
    matrix = r.as_matrix()
    return matrix

def move_long_quaternion(position, quaternion, distance):
    """
        Move along the quaternion direction
    """
    roation = R.from_quat(quaternion)
    direction = roation.as_rotvec()
    direction = direction / np.linalg.norm(direction)
    new_position = position + direction * distance
    return new_position

def create_mesh_box(width, height, depth, dx=0, dy=0, dz=0):
    ''' Author: chenxi-wang
    Create box instance with mesh representation.
    '''
    box = o3d.geometry.TriangleMesh()
    vertices = np.array([[0,0,0],
                         [width,0,0],
                         [0,0,depth],
                         [width,0,depth],
                         [0,height,0],
                         [width,height,0],
                         [0,height,depth],
                         [width,height,depth]])
    vertices[:,0] += dx
    vertices[:,1] += dy
    vertices[:,2] += dz
    triangles = np.array([[4,7,5],[4,6,7],[0,2,4],[2,6,4],
                          [0,1,2],[1,3,2],[1,5,7],[1,7,3],
                          [2,3,7],[2,7,6],[0,4,1],[1,4,5]])
    box.vertices = o3d.utility.Vector3dVector(vertices)
    box.triangles = o3d.utility.Vector3iVector(triangles)
    return box

def pcd_has_overlap(pcd1, pcd2, voxel_size=0.05):
    voxel_grid1 = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd1, voxel_size)
    voxel_grid2 = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd2, voxel_size)
    
    voxels1 = set(map(lambda v: (v.grid_index[0], v.grid_index[1], v.grid_index[2]), voxel_grid1.get_voxels()))
    voxels2 = set(map(lambda v: (v.grid_index[0], v.grid_index[1], v.grid_index[2]), voxel_grid2.get_voxels()))
    
    intersection = voxels1.intersection(voxels2)
    union = voxels1.union(voxels2)
    
    iou = len(intersection) / len(union) if len(union) > 0 else 0
    
    return iou


def adjust_grasp_euler(key_euler, entity, physics, grasp_direction="vertical"):
    """
    Correct key_euler based on the object's current orientation and grasp_type, then apply grasp_roll_offset on top.

    The two modes correct different physical quantities:
    - vertical:  modifies key_euler[2] (yaw), i.e. the gripper's spin about its center axis (aligning the opening direction with the long axis)
    - horizontal: modifies key_euler[0] (roll), i.e. the approach direction (toward the object's long axis/handle face)

    The second layer (grasp_roll_offset) modifies the center-spin component in both modes:
    - vertical: key_euler[2] += offset
    - horizontal: key_euler[1] += offset  (pitch controls the wide-face orientation, i.e. the center spin)

    Args:
        key_euler: list/np.array, length-3 euler angles [roll, pitch, yaw]
        entity: Entity object
        physics: physics engine object
        grasp_direction: str, "vertical" or "horizontal"

    Returns:
        np.array: corrected key_euler (length 3)
    """
    key_euler = np.array(key_euler, dtype=float)
    grasp_type = getattr(entity, 'grasp_type', 1)

    if grasp_type != 1:
        entity_xquat = np.array(physics.bind(entity.mjcf_model.worldbody).xquat)
        R = quaternion_to_matrix(entity_xquat)
        entity_z_world = R @ np.array([0, 0, 1])

        if grasp_type == 2:
            long_axis_local = getattr(entity, 'long_axis', np.array([0, 1, 0]))
            long_axis_world = R @ np.array(long_axis_local, dtype=float)
            long_axis_yaw = np.arctan2(long_axis_world[1], long_axis_world[0])

            if grasp_direction == "vertical":
                key_euler[2] = long_axis_yaw + np.pi
            else:
                key_euler[0] = long_axis_yaw + np.pi

        elif grasp_type == 3:
            if abs(entity_z_world[2]) < 0.9:
                long_axis_yaw = np.arctan2(entity_z_world[1], entity_z_world[0])

                if grasp_direction == "vertical":
                    key_euler[2] = long_axis_yaw + np.pi
                else:
                    key_euler[0] = long_axis_yaw + np.pi

    return key_euler


def find_keypoint_and_prepare_grasp(env, entity, prior_euler, std=0, max_retry=100, specific_keypoint_id=0, move_vector=None, prepare_distance=-0.15, base_yaw=None, body_name=None, grasp_direction="vertical"):
    """
    sample a keypoint and confirm the validation of it and its prepare point
    return the valid grasp keypoint, prepare point and the quaternion of the gripper
    
    Args:
        body_name: if specified, only get grasp points under this body; otherwise get all grasp points of the entity
    """
    if body_name is not None:
        keypoints = entity.get_grasped_keypoints(env.physics, body_name)
    else:
        keypoints = entity.get_grasped_keypoints(env.physics)

    valid = False
    env_pcd = env.get_observation()["masked_point_cloud"]
    retry = 0
    while not valid:
        if specific_keypoint_id is not None:
            keypoint = keypoints[specific_keypoint_id]
        else:
            keypoint = random.choice(keypoints)
        key_euler = prior_euler[retry % len(prior_euler)]
        key_euler = adjust_grasp_euler(key_euler, entity, env.physics, grasp_direction)
        key_euler += np.random.normal(0, std, 3)
        key_quat = euler_to_quaternion(*key_euler)
        roll_offset = getattr(entity, 'grasp_roll_offset', 0)
        if roll_offset != 0:
            offset_rad = np.radians(roll_offset)
            rot_z = np.array([np.cos(offset_rad/2), 0, 0, np.sin(offset_rad/2)])
            key_quat = quaternion_multiply(key_quat, rot_z)
        gripper_pcd, gripper_forward_vector = env.robot.gripper_pcd(keypoint, key_quat)
        if move_vector is None:
            move_vector = gripper_forward_vector

        prepare_point = keypoint + move_vector * prepare_distance
        grasp_prepare_gripper_pcd = copy.deepcopy(gripper_pcd).translate(move_vector * prepare_distance)
        grasp_prepare_gripper_pcd.paint_uniform_color([0, 1, 0])

        collision1 = gripper_collision_check(gripper_pcd, env_pcd)
        collision2 = gripper_collision_check(grasp_prepare_gripper_pcd, env_pcd)
        if not collision1 and not collision2:
            valid = True
        retry += 1
        if retry > max_retry:
            fallback_euler = adjust_grasp_euler(prior_euler[0], entity, env.physics, grasp_direction)
            key_quat_default = euler_to_quaternion(*fallback_euler)
            roll_offset = getattr(entity, 'grasp_roll_offset', 0)
            if roll_offset != 0:
                offset_rad = np.radians(roll_offset)
                rot_z = np.array([np.cos(offset_rad/2), 0, 0, np.sin(offset_rad/2)])
                key_quat_default = quaternion_multiply(key_quat_default, rot_z)
            _, move_vec_default = env.robot.gripper_pcd(keypoint, key_quat_default)
            fallback_prepare = keypoint + move_vec_default * prepare_distance
            return keypoint, fallback_prepare, key_quat_default
        if retry % 10 == 0:
            std += np.pi / 100

    return keypoint, prepare_point, key_quat

def find_interactive_site_and_prepare(env, site_name, prior_euler, prepare_distance=-0.15, base_yaw=None):
    """
    Get the grasp position and prepare point for an interactive manipulation site
    (e.g. knob, button) by site_name.

    Keeps the same return style as find_keypoint_and_prepare_grasp:
    return keypoint, prepare_point, key_quat

    Args:
        env: environment object
        site_name: str, full MJCF site name (e.g. "grasp_knob_front_right")
        prior_euler: list, candidate euler angles for the grasp pose
        prepare_distance: float, backward offset distance of the prepare point along the approach direction
        base_yaw: float, robot base heading (radians), used to compute the grasp pose

    Returns:
        (keypoint, prepare_point, key_quat): grasp point, prepare point, grasp pose quaternion
    """
    from RMMBench.tasks.components.robocasa_scene import RobocasaScene
    from RMMBench.tasks.components.specific_entities.interactive_containers import Stove

    for entity in env.task.entities.values():
        if isinstance(entity, (Stove, RobocasaScene)):
            site, site_xpos = entity.get_site_by_name(site_name, env.physics)
            if site is not None:
                keypoint = np.array(site_xpos)
                if base_yaw is not None:
                    key_euler = [np.pi, 0, base_yaw - np.pi]
                else:
                    key_euler = prior_euler[0] if isinstance(prior_euler, list) else prior_euler
                key_quat = euler_to_quaternion(*key_euler)
                _, move_vector = env.robot.gripper_pcd(keypoint, key_quat)
                prepare_point = keypoint + move_vector * prepare_distance
                return keypoint, prepare_point, key_quat

    raise ValueError(f"Site {site_name} not found in any entity")

def gripper_collision_check(gripper_pcd, env_pcd, threshold=0.1):
    iou = pcd_has_overlap(env_pcd, gripper_pcd)
    return iou > threshold


def get_geom_ids_by_prefix(env, entity_name, prefix=None):
    """
    Get the geom IDs in an entity whose names match the given prefix

    Args:
        env: environment object
        entity_name: entity name
        prefix: geom name prefix, e.g. "sink_island_group"

    Returns:
        list: matching geom IDs
    """
    entity = env.task.entities[entity_name]
    geom_ids = []
    for geom in entity.geoms:
        geom_name = geom.name
        if prefix is None or geom_name.startswith(prefix):
            geom_id = env.physics.bind(geom).element_id
            geom_ids.append(geom_id)
    return geom_ids

def get_entity_mask_from_seg(seg, env, entity_name):
    """
    Get the mask of a specific object or body from the segmentation image

    Args:
        seg: segmentation image (H, W, 2) or (H, W, C)
        env: environment object
        entity_name: object name (e.g. 'apple_0', 'carrot_2') or body name (e.g. 'knob_front_right')

    Returns:
        list: list of matching geom_ids
    """
    geom_ids = []

    entity_name_list = flatten_list(entity_name)

    for name in entity_name_list:
        if name in env.task.entities:
            entity = env.task.entities[name]
            ids = [env.physics.bind(geom).element_id for geom in entity.geoms]
            geom_ids.extend(ids)
        else:
            try:
                body = env.get_element_by_name(name, type="body")
                if body is not None:
                    body_geoms = body.find_all("geom")
                    ids = [env.physics.bind(geom).element_id for geom in body_geoms]
                    geom_ids.extend(ids)
            except (ValueError, AttributeError):
                pass

    return geom_ids


def extract_containers(data):
    """
    Recursively extract all container and container_name entries from a dict or list.
    Returns a list whose elements are (container_key, container_value) tuples
    """
    containers = []
    entities = []

    if isinstance(data, dict):
        for target_key in ["container", "container_name"]:
            if target_key in data:
                containers.append(data[target_key])
        for target_key in ["entities"]:
            if target_key in data:
                entities.append(data[target_key])

        for value in data.values():
            containers.extend(extract_containers(value))
            entities.extend(extract_containers(value))


    elif isinstance(data, list):
        for item in data:
            containers.extend(extract_containers(item))

    return flatten_list(containers),

def calculate_mask_bbox_iou(env, entity_name, bbox, cam_id, body_name=None):
    """
    Compute the IoU between the segmentation mask and a bounding box
    full_mask: np.array (H, W), values 0 or 1
    bbox: [y_min, x_min, y_max, x_max]
    """
    _, _, seg, _, _ = env.get_camera_parse_physics(cam_id)

    if body_name is not None:
        entity = env.task.entities.get(entity_name)
        if entity is not None:
            body = entity.mjcf_model.find("body", body_name)
            if body is not None:
                geom_ids = [env.physics.bind(geom).element_id for geom in body.find_all("geom")]
            else:
                geom_ids = []
        else:
            geom_ids = []
    else:
        geom_ids = get_entity_mask_from_seg(seg, env, entity_name)

    full_mask = np.isin(seg[..., 0], geom_ids).astype(np.uint8)

    h, w = full_mask.shape
    y_min, x_min, y_max, x_max = bbox

    bbox_mask = np.zeros((h, w), dtype=np.uint8)
    bbox_mask[int(y_min):int(y_max), int(x_min):int(x_max)] = 1

    intersection = np.logical_and(full_mask, bbox_mask).sum()
    union = np.logical_or(full_mask, bbox_mask).sum()

    if union == 0:
        return 0.0

    iou = intersection / union
    return iou,full_mask


def get_placement_top_down_view(env, bbox, placement_offset=0.02, cam_id=5,target_container_name=None):
    rgb, depth_image, seg, camera_matrix, camera_pose = env.get_camera_parse_physics(cam_id)
    h, w = depth_image.shape
    y_min, x_min, y_max, x_max = [int(v) for v in bbox]

    y_min, y_max = max(0, y_min), min(h, y_max)
    x_min, x_max = max(0, x_min), min(w, x_max)
    center_y, center_x = (y_min + y_max) // 2, (x_min + x_max) // 2

    geom_ids = []
    container = env.task.config_manager.target_container
    container = [container] if isinstance(container, str) else container
    for name in container:
        if target_container_name is not None:
            if target_container_name in name:
                ids = get_entity_mask_from_seg(seg, env, name)
                if len(ids) > 0:
                    geom_ids.extend(ids)
                break

    else:
        if target_container_name is not None and len(geom_ids) == 0:
            for entity_name in env.task.entities:
                if target_container_name in entity_name:
                    ids = get_entity_mask_from_seg(seg, env, entity_name)
                    if len(ids) > 0:
                        geom_ids.extend(ids)
                        break

        if len(geom_ids) == 0 and env.task.config_manager.robocasa_scene is not None:
            robocasa_scene = env.task.config_manager.robocasa_scene
            robocasa_container = None

            if "scene_contain" in env.task.config["task"]["conditions"]:
                robocasa_container = env.task.config["task"]["conditions"]["scene_contain"]["container_name"]

            if robocasa_container is not None:
                geom_ids_robocasa = get_geom_ids_by_prefix(env, robocasa_scene, robocasa_container)
                geom_ids.extend(geom_ids_robocasa)

    full_mask = np.isin(seg[..., 0], geom_ids).astype(np.uint8)

    local_mask = full_mask[y_min:y_max, x_min:x_max]


    def collect_samples(m, use_depth_filter=False):
        res = []
        y_idxs, x_idxs = np.where(m == 1)
        for y_l, x_l in zip(y_idxs, x_idxs):
            gy, gx = y_min + y_l, x_min + x_l
            d = depth_image[gy, gx]
            if d > 0.1:
                dist = np.sqrt((gx - center_x) ** 2 + (gy - center_y) ** 2)
                res.append({'uv': (gx, gy), 'd': d, 'dist': dist})

        if not res: return None

        if use_depth_filter:
            res.sort(key=lambda x: x['d'], reverse=True)
            return res[:max(1, int(len(res) * 0.3))]
        else:
            res.sort(key=lambda x: x['dist'])
            return res[:max(1, int(len(res) * 0.1))]

    kernel = np.ones((5, 5), np.uint8)
    eroded_mask = cv2.erode(local_mask, kernel, iterations=1)
    final_selection = collect_samples(eroded_mask, use_depth_filter=False)

    if not final_selection:
        final_selection = collect_samples(local_mask, use_depth_filter=False)

    if not final_selection:
        bbox_mask = np.ones_like(local_mask)
        final_selection = collect_samples(bbox_mask, use_depth_filter=True)

    if final_selection:
        sample_world_points = []
        inv_camera_matrix = np.linalg.inv(camera_matrix)
        for item in final_selection:
            u, v = item['uv']
            depth = item['d']
            pixel_point = np.array([u, v, 1.0])
            camera_point = inv_camera_matrix @ pixel_point * depth
            world_point_homo = camera_pose @ np.append(camera_point, 1.0)
            sample_world_points.append(world_point_homo[:3] / world_point_homo[3])

        res_center = np.median(sample_world_points, axis=0)
        return res_center + np.array([0, 0, placement_offset])

    return None


def pcd_has_overlap(pcd1, pcd2, distance_threshold=0.05):
    points1 = np.asarray(pcd1.points)
    points2 = np.asarray(pcd2.points)

    tree1 = cKDTree(points1)
    tree2 = cKDTree(points2)

    distances1, _ = tree2.query(points1, k=1)
    matches1 = distances1 < distance_threshold

    distances2, _ = tree1.query(points2, k=1)
    matches2 = distances2 < distance_threshold

    intersection_count = np.sum(matches1) + np.sum(matches2)
    union_count = len(points1) + len(points2) - intersection_count
    
    iou = intersection_count / union_count if union_count > 0 else 0
    
    return iou

def distance(p1, p2):
    if not isinstance(p1, np.ndarray):
        p1 = np.array(p1)
    if not isinstance(p2, np.ndarray):
        p2 = np.array(p2)
    return np.linalg.norm(p1 - p2)

def farthest_first_sampling(points, k):
    sampled_points = [points[np.random.randint(len(points))]]
    
    for _ in range(1, k):
        min_distances = [min(distance(p, sp) for sp in sampled_points) for p in points]
        
        farthest_point = points[np.argmax(min_distances)]
        sampled_points.append(farthest_point)
    
    return sampled_points

def _sample_from_pool(pool, n, farthest=True):
    """Sample n points from pool using farthest-first or random sampling."""
    if n <= 0 or len(pool) == 0:
        return []
    n = min(n, len(pool))
    if farthest:
        return farthest_first_sampling(pool, n)
    else:
        return random.sample(pool, n)


_SPATIAL_DIRECTION_MAP = {
    ("bottom", "back"):  ("y", "high"),
    ("bottom", "front"): ("y", "low"),
    ("bottom", "left"):  ("x", "low"),
    ("bottom", "right"): ("x", "high"),
    ("top", "back"):     ("y", "low"),
    ("top", "front"):    ("y", "high"),
    ("top", "left"):     ("x", "high"),
    ("top", "right"):    ("x", "low"),
    ("left", "back"):    ("x", "high"),
    ("left", "front"):   ("x", "low"),
    ("left", "left"):    ("y", "high"),
    ("left", "right"):   ("y", "low"),
    ("right", "back"):   ("x", "low"),
    ("right", "front"):  ("x", "high"),
    ("right", "left"):   ("y", "low"),
    ("right", "right"):  ("y", "high"),
}

_SPATIAL_OPPOSITE = {"back": "front", "front": "back", "left": "right", "right": "left"}


def grid_sample(workspace, grid_size, n_samples, farthest_sample=True,
                mid_container_sample=None, n_mid=0, destination_position=None,
                split_ratio=0.5):
    """
    workspace: [min_x, max_x, min_y, max_y, min_z, max_z]
    grid_size: [n_row, n_col]

    When mid_container_sample is not None and n_mid > 0:
      - Grid points are split into two halves based on destination_position and mid_container_sample.
      - mid_container parents are sampled from one half; regular objects from the opposite half.
      - split_ratio controls the fraction of the axis allocated to the mid_container side.
      - Returns [regular_points..., mid_points...] to match load_objects consumption order.
    Otherwise, falls back to original farthest_sample / random behavior.
    """
    min_x, max_x, min_y, max_y, _, _ = workspace
    n_row, n_col = grid_size
    x_step = (max_x - min_x) / n_col
    y_step = (max_y - min_y) / n_row

    grid_points = []
    for i in range(n_row):
        for j in range(n_col):
            center_x = min_x + (j + 0.5) * x_step
            center_y = min_y + (i + 0.5) * y_step
            grid_points.append((center_x, center_y))

    if mid_container_sample is not None and n_mid > 0 and destination_position is not None:
        mid_axis, mid_dir = _SPATIAL_DIRECTION_MAP[(destination_position, mid_container_sample)]
        reg_key = _SPATIAL_OPPOSITE[mid_container_sample]
        _, reg_dir = _SPATIAL_DIRECTION_MAP[(destination_position, reg_key)]

        if mid_axis == "x":
            axis_min, axis_max = min_x, max_x
            axis_idx = 0
        else:
            axis_min, axis_max = min_y, max_y
            axis_idx = 1

        if mid_dir == "high":
            threshold = axis_min + (axis_max - axis_min) * (1 - split_ratio)
            mid_pool = [p for p in grid_points if p[axis_idx] >= threshold]
            reg_pool = [p for p in grid_points if p[axis_idx] < threshold]
        else:
            threshold = axis_min + (axis_max - axis_min) * split_ratio
            mid_pool = [p for p in grid_points if p[axis_idx] < threshold]
            reg_pool = [p for p in grid_points if p[axis_idx] >= threshold]

        n_regular = n_samples - n_mid

        if mid_axis == "x":
            edge_center = (axis_max if mid_dir == "high" else axis_min,
                           (min_y + max_y) / 2)
        else:
            edge_center = ((min_x + max_x) / 2,
                            axis_max if mid_dir == "high" else axis_min)

        mid_pool_sorted = sorted(mid_pool, key=lambda p: distance(p, edge_center))
        mid_sampled = mid_pool_sorted[:n_mid]

        reg_sampled = _sample_from_pool(reg_pool, n_regular, farthest_sample)

        if len(reg_sampled) < n_regular:
            borrowed = [p for p in mid_pool if p not in mid_sampled and p not in reg_sampled]
            reg_sampled.extend(_sample_from_pool(borrowed, n_regular - len(reg_sampled), farthest_sample))
        if len(mid_sampled) < n_mid:
            borrowed = [p for p in reg_pool if p not in reg_sampled and p not in mid_sampled]
            mid_sampled.extend(_sample_from_pool(borrowed, n_mid - len(mid_sampled), farthest_sample))
        return reg_sampled + mid_sampled

    if farthest_sample:
        sampled_points = farthest_first_sampling(grid_points, n_samples)
    else:
        sampled_points = random.sample(grid_points, n_samples)

    return sampled_points


def multi_grid_sample(workregion, grid_size, entity_n_samples,anchor="bottom",sample_type="long_distance"):
    sampled_points_info = None

    min_x, max_x, min_y, max_y, min_z, max_z = workregion
    n_row, n_col = grid_size
    x_step = (max_x - min_x) / n_col
    y_step = (max_y - min_y) / n_row
    grid_points = []
    for i in range(n_row):
        for j in range(n_col):
            center_x = min_x + (j + 0.5) * x_step
            center_y = min_y + (i + 0.5) * y_step
            grid_points.append((center_x, center_y))

    if sample_type == "long_distance":
        sampled_points_info = long_distance_task(entity_n_samples, sample_type,anchor)

    return sampled_points_info


def long_distance_task(points, k,anchor):
    pass

def point_to_line_distance(anchor, axis, point):
    """
    compute the distance from a point to a line
    
    param:
    - anchor: the anchor point of rotation axis (3D vector) [x, y, z]
    - axis: the direction vector of rotation axis [vx, vy, vz]
    - point: (3D vector) [x, y, z]
    
    return:
    - the distance
    """
    A = np.array(anchor)  
    V = np.array(axis)    
    Q = np.array(point)

    AQ = Q - A

    cross_product = np.cross(AQ, V)

    distance = np.linalg.norm(cross_product)

    return distance

def rotate_point_around_axis(point, anchor, axis, angle):
    """
    compute the point after rotation around the axis with Rodrigues' rotation formula

    params:
    - point: (3D vector) [x, y, z]
    - anchor:(3D vector) [x, y, z]
    - axis: (3D vector) [vx, vy, vz]
    - angle: rotation angle (radian)

    return:
    - the vector point after (3D vector)
    """
    P = np.array(point)
    A = np.array(anchor)
    V = np.array(axis) / np.linalg.norm(axis)  

    PA = P - A

    part1 = np.cos(angle) * PA
    part2 = np.sin(angle) * np.cross(V, PA)
    part3 = (1 - np.cos(angle)) * V * np.dot(V, PA)

    P_prime = A + part1 + part2 + part3

    return P_prime

def slide_point_along_axis(point, axis, distance):
    """
    compute the point after sliding along the axis 

    params:
    - point: (3D vector) [x, y, z]
    - axis: (3D vector) [vx, vy, vz]
    - angle: rotation angle (radian)

    return:
    - the vector point after (3D vector)
    """
    point = np.array(point)
    axis = np.array(axis)
    
    xaxis_normalized = axis / np.linalg.norm(axis)
    
    new_point = point + distance * xaxis_normalized
    
    return new_point

def quaternion_from_axis_angle(axis, angle):
    """
    param:
     - angle: radian
    """
    half_angle = angle / 2
    w = np.cos(half_angle)
    sin_half_angle = np.sin(half_angle)
    
    v = np.array(axis) / np.linalg.norm(axis)
    
    x = v[0] * sin_half_angle
    y = v[1] * sin_half_angle
    z = v[2] * sin_half_angle
    
    return np.array([w, x, y, z])

def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2

    return np.array([w, x, y, z])

def flatten_list(ls):
    if isinstance(ls, str):
        return [ls]
    if ls is not None:
        new_list = []

        for item in ls:
            if isinstance(item, list):
                new_list.extend(item)
            elif isinstance(item, str):
                new_list.append(item)
        return new_list
    else:
        return []

def quaternion_conjugate(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z])

def rotate_point_by_quaternion(point, quat):
    p = np.array([0] + list(point))
    q_conj = quaternion_conjugate(quat)
    p_prime = quaternion_multiply(quaternion_multiply(quat, p), q_conj)
    
    return p_prime[1:]

def expand_mask(masks, kernel_size=3, iterations=1):
    """
    Expands a batch of binary masks (0 and 1 values) using morphological dilation.
    
    Parameters:
    - masks: np.ndarray, shape (n, h, w), batch of binary masks (0 and 1 values).
    - kernel_size: int, size of the kernel for dilation, default is 3x3.
    - iterations: int, number of times to apply dilation, default is 1.

    Returns:
    - expanded_masks: np.ndarray, shape (n, h, w), batch of masks with dilated edges.
    """
    if len(masks.shape) == 2:
        masks = masks.reshape(1, masks.shape[0], masks.shape[1])
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    expanded_masks = np.zeros_like(masks, dtype=np.uint8)
    for i in range(masks.shape[0]):
        inverted_mask = 1 - masks[i]
        mask_uint8 = (inverted_mask * 255).astype(np.uint8)
        expanded_mask = cv2.dilate(mask_uint8, kernel, iterations=iterations)
        expanded_masks[i] = 1 - (expanded_mask > 0).astype(np.uint8)
    return expanded_masks
    

def pcd_filtering(pcd, eps=0.05, min_samples=10):
    """
    DBSCAN clustering
    params:
        eps: The maximum distance two points can be considered neighbors.
        min_samples: The minimum number of points required to form a dense cluster.
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(pcd)
    unique_labels, counts = np.unique(labels, return_counts=True)

    if -1 in unique_labels:
        noise_index = np.where(unique_labels == -1)[0][0]
        unique_labels = np.delete(unique_labels, noise_index)
        counts = np.delete(counts, noise_index)

    max_cluster_label = unique_labels[np.argmax(counts)]
    most_dense_cluster = pcd[labels == max_cluster_label]
    return most_dense_cluster

def find_key_by_value(dictionary, target_value):
    """
    Given a dictionary and the corresponding value, find the key that contains the target value
    """
    for key, value in dictionary.items():
        if isinstance(value, list) and target_value in value:
            return key
        elif not isinstance(value, list) and value == target_value:
            return key
    return target_value

def get_logger(level=logging.INFO):
    logger = logging.getLogger()
    logger.setLevel(level)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    color_formatter = colorlog.ColoredFormatter(
        '%(log_color)s%(levelname)s: %(message)s',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )
    console_handler.setFormatter(color_formatter)
    for handler in logger.handlers:
        logger.removeHandler(handler)
        
    logger.addHandler(console_handler)
    return logger

def visulize_grid_point(points):
    import matplotlib.pyplot as plt
    x_coords = [point[0] for point in points]
    y_coords = [point[1] for point in points]

    plt.figure(figsize=(8, 8))
    plt.scatter(x_coords, y_coords, c='blue', marker='o')

    for i, point in enumerate(points):
        plt.text(point[0], point[1], f'({point[0]:.1f}, {point[1]:.1f})', fontsize=9, ha='right')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('2D Points Visualization')
    plt.grid(True)
    plt.axis('equal')

    plt.show()


def get_direction_from_euler(yaw, tolerance=0.1):
    direction_map = {
        0.0: "right",
        1.57: "top",
        -1.57: "bottom",
        3.14: "left",
        -3.14: "left"
    }
    for angle, direction in direction_map.items():
        if abs(yaw - angle) <= tolerance:
            return direction
    return "unknown"


def compute_lmax_hanan_grid(current_pos, move_direction, obstacle_bboxes, stay_points,
                            robot_radius=0.2, boundary=None, max_search_dist=10.0,
                            boundary_tol=0.05):
    """
    Compute the maximum safe movement distance Lmax on the Hanan grid.
    Inflate only once (by robot_radius); the inflated boundary is the hard boundary (impassable).

    Tolerance-zone logic (boundary_tol):
    - Robot within inflated boundary ±boundary_tol is considered "on the boundary"
    - On the boundary and facing inward (perpendicular to the boundary, toward the obstacle interior) → stop (return 0)
    - On the boundary and facing outward or parallel → navigate normally
    - Strictly inside (away from the boundary) → stop (return 0)

    Layered logic:
    1. Parameter preprocessing
    2. Obstacle clipping and inflation (only once)
    3. Boundary detection: robot inside/on/outside the inflated boundary (including the tolerance zone)
    4. Physical wall distance computation (distance to the nearest inflated boundary ahead)
    5. Collect Hanan-aligned lines (reuse the inflated boundaries)
    6. Extract and filter candidate points (stay-point ray check + Hanan-line extraction)
    7. Priority-based decision (stay points > Hanan lines > physical walls)

    :param boundary: [xmin, xmax, ymin, ymax] or None
    :param boundary_tol: boundary tolerance depth, default 0.05
    :return: float (positive = distance, 0 = stop, inf = default distance)
    """


    curr_x, curr_y = current_pos
    direct = move_direction.strip().lower()
    tol = 0.08
    T = boundary_tol

    b_xmin, b_xmax, b_ymin, b_ymax = (None, None, None, None) if boundary is None else boundary


    effective_boxes = []
    inflated_boxes = []

    for box in obstacle_bboxes:
        x1 = max(box[0], b_xmin) if b_xmin is not None else box[0]
        y1 = max(box[1], b_ymin) if b_ymin is not None else box[1]
        x2 = min(box[2], b_xmax) if b_xmax is not None else box[2]
        y2 = min(box[3], b_ymax) if b_ymax is not None else box[3]

        if x1 < x2 and y1 < y2:
            effective_boxes.append([x1, y1, x2, y2])
            ib = [x1 - robot_radius, y1 - robot_radius,
                  x2 + robot_radius, y2 + robot_radius]
            inflated_boxes.append(ib)

    for ib in inflated_boxes:
        bx1, by1, bx2, by2 = ib

        near_bottom = abs(curr_y - by1) < T and (bx1 - T) <= curr_x <= (bx2 + T)
        near_top    = abs(curr_y - by2) < T and (bx1 - T) <= curr_x <= (bx2 + T)
        near_left   = abs(curr_x - bx1) < T and (by1 - T) <= curr_y <= (by2 + T)
        near_right  = abs(curr_x - bx2) < T and (by1 - T) <= curr_y <= (by2 + T)
        on_boundary = near_bottom or near_top or near_left or near_right

        if on_boundary:
            can_slide = False
            if near_bottom and direct in ['+x', '-x']:
                can_slide = True
            elif near_top and direct in ['+x', '-x']:
                can_slide = True
            elif near_left and direct in ['+y', '-y']:
                can_slide = True
            elif near_right and direct in ['+y', '-y']:
                can_slide = True
            
            if can_slide:
                continue
            
            if near_bottom:
                if direct == '+y':
                    return 0.0
                elif direct == '-y':
                    continue
            if near_top:
                if direct == '-y':
                    return 0.0
                elif direct == '+y':
                    continue
            if near_left:
                if direct == '+x':
                    return 0.0
                elif direct == '-x':
                    continue
            if near_right:
                if direct == '-x':
                    return 0.0
                elif direct == '+x':
                    continue

        if bx1 < curr_x < bx2 and by1 < curr_y < by2:
            return 0.0

    if boundary is not None:
        if b_ymin is not None and abs(curr_y - (b_ymin + robot_radius)) < T:
            if direct == '-y':
                return 0.0
        if b_ymax is not None and abs(curr_y - (b_ymax - robot_radius)) < T:
            if direct == '+y':
                return 0.0
        if b_xmin is not None and abs(curr_x - (b_xmin + robot_radius)) < T:
            if direct == '-x':
                return 0.0
        if b_xmax is not None and abs(curr_x - (b_xmax - robot_radius)) < T:
            if direct == '+x':
                return 0.0

    dist_to_wall = float('inf')

    if direct == '+x' and b_xmax is not None:
        dist_to_wall = min(dist_to_wall, b_xmax - robot_radius - curr_x)
    elif direct == '-x' and b_xmin is not None:
        dist_to_wall = min(dist_to_wall, curr_x - (b_xmin + robot_radius))
    elif direct == '+y' and b_ymax is not None:
        dist_to_wall = min(dist_to_wall, b_ymax - robot_radius - curr_y)
    elif direct == '-y' and b_ymin is not None:
        dist_to_wall = min(dist_to_wall, curr_y - (b_ymin + robot_radius))

    for ib in inflated_boxes:
        bx1, by1, bx2, by2 = ib

        d = float('inf')
        if direct in ['+x', '-x']:
            if (by1 + tol) < curr_y < (by2 - tol):
                if direct == '+x' and bx1 > curr_x:
                    d = bx1 - curr_x
                elif direct == '-x' and bx2 < curr_x:
                    d = curr_x - bx2
        elif direct in ['+y', '-y']:
            if (bx1 + tol) < curr_x < (bx2 - tol):
                if direct == '+y' and by1 > curr_y:
                    d = by1 - curr_y
                elif direct == '-y' and by2 < curr_y:
                    d = curr_y - by2
        dist_to_wall = min(dist_to_wall, d)


    obs_xs, obs_ys = set(), set()
    for ib in inflated_boxes:
        obs_xs.update([ib[0], ib[2]])
        obs_ys.update([ib[1], ib[3]])

    if b_xmin is not None: obs_xs.add(b_xmin + robot_radius)
    if b_xmax is not None: obs_xs.add(b_xmax - robot_radius)
    if b_ymin is not None: obs_ys.add(b_ymin + robot_radius)
    if b_ymax is not None: obs_ys.add(b_ymax - robot_radius)


    def get_valid_stay_points(stay_pts, current_pos, direction, wall_limit, boxes):
        valid_dists = []
        cx, cy = current_pos
        for sp in stay_pts:
            sp_x, sp_y = sp[0], sp[1]
            if direction in ['+y', '-y']:
                d = (sp_y - cy) if direction == '+y' else (cy - sp_y)
                check_path = (sp_x, cx, sp_y, True)
            else:
                d = (sp_x - cx) if direction == '+x' else (cx - sp_x)
                check_path = (sp_y, cy, sp_x, False)

            if tol < d <= wall_limit:
                is_blocked = False
                start, end, fixed, is_horiz = check_path
                for ib in boxes:
                    bx1, by1, bx2, by2 = ib
                    if is_horiz:
                        if (by1 + 0.01) < fixed < (by2 - 0.01):
                            if not (bx2 < min(start, end) or bx1 > max(start, end)):
                                is_blocked = True
                                break
                    else:
                        if (bx1 + 0.01) < fixed < (bx2 - 0.01):
                            if not (by2 < min(start, end) or by1 > max(start, end)):
                                is_blocked = True
                                break
                if not is_blocked:
                    valid_dists.append(d)
        return sorted(valid_dists)

    def get_valid_hanan(target_set, current_val, direction, wall_limit):
        return sorted([d for d in [(v - current_val if direction in ['+x', '+y'] else current_val - v)
                                   for v in target_set] if tol < d <= wall_limit])

    if direct in ['+x', '-x']:
        s_dists = get_valid_stay_points(stay_points, current_pos, direct, dist_to_wall, inflated_boxes)
        o_dists = get_valid_hanan(obs_xs, curr_x, direct, dist_to_wall)
    else:
        s_dists = get_valid_stay_points(stay_points, current_pos, direct, dist_to_wall, inflated_boxes)
        o_dists = get_valid_hanan(obs_ys, curr_y, direct, dist_to_wall)


    if s_dists:
        res = s_dists[0]
        return res if res < max_search_dist else float('inf')

    if o_dists:
        res = o_dists[-1]
        return res if res < max_search_dist else float('inf')

    if tol < dist_to_wall < max_search_dist:
        return dist_to_wall

    return float('inf')


def get_hanan_l1_distance(start, end, bbox_list, boundary=None, robot_radius=0.2):
    """
    Compute the shortest Manhattan distance with obstacle avoidance (L1 geodesic distance)
    :param start: [x, y]
    :param end: [x, y]
    :param bbox_list: [[min_x, min_y, max_x, max_y], ...]
    """
    inflated_boxes = []
    xs = {start[0], end[0]}
    ys = {start[1], end[1]}

    for box in bbox_list:
        ibox = [box[0] - robot_radius, box[1] - robot_radius,
                box[2] + robot_radius, box[3] + robot_radius]
        inflated_boxes.append(ibox)
        xs.update([ibox[0], ibox[2]])
        ys.update([ibox[1], ibox[3]])

    if boundary:
        for i, val in enumerate(boundary):
            if val is not None:
                if i < 2:
                    xs.add(val)
                else:
                    ys.add(val)

    sorted_xs = sorted(list(xs))
    sorted_ys = sorted(list(ys))

    x_map = {val: i for i, val in enumerate(sorted_xs)}
    y_map = {val: j for j, val in enumerate(sorted_ys)}

    start_node = (x_map[start[0]], y_map[start[1]])
    end_node = (x_map[end[0]], y_map[end[1]])

    pq = [(0, start_node[0], start_node[1])]
    visited_dist = {start_node: 0}

    while pq:
        dist, ix, iy = heapq.heappop(pq)

        if (ix, iy) == end_node:
            return dist

        if dist > visited_dist.get((ix, iy), float('inf')):
            continue

        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = ix + dx, iy + dy

            if 0 <= nx < len(sorted_xs) and 0 <= ny < len(sorted_ys):
                p1 = (sorted_xs[ix], sorted_ys[iy])
                p2 = (sorted_xs[nx], sorted_ys[ny])

                mid = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
                is_collision = False
                for box in inflated_boxes:
                    if (box[0] + 1e-6) < mid[0] < (box[2] - 1e-6) and \
                            (box[1] + 1e-6) < mid[1] < (box[3] - 1e-6):
                        is_collision = True
                        break

                if not is_collision:
                    weight = abs(p2[0] - p1[0]) + abs(p2[1] - p1[1])
                    new_dist = dist + weight
                    if new_dist < visited_dist.get((nx, ny), float('inf')):
                        visited_dist[(nx, ny)] = new_dist
                        heapq.heappush(pq, (new_dist, nx, ny))

    return float('inf')

def euler_to_rotation_matrix(euler, order='xyz'):
    """
    Convert euler angles [roll, pitch, yaw] (radians) to a 3x3 rotation matrix.

    Parameters:
        euler: [roll, pitch, yaw] euler angles in radians
        order: rotation order, default 'xyz' (extrinsic X-Y-Z, matching scipy's as_euler('xyz'))

    Returns:
        3x3 rotation matrix
    """
    r = R.from_euler(order, euler, degrees=False)
    return r.as_matrix()


def compute_camera_positions_and_bounds(robot_pos, robot_euler,
                                        left_offset=np.array([0.1, 0.225, 1.34]),
                                        right_offset=np.array([0.1, -0.4, 1.34]),
                                        bound_center_offset=np.array([0.95, 0.0, 0.75]),
                                        bound_extents=np.array([1.0, 1.0, 0.65])):
    """
    Compute left/right camera world positions and point cloud bounds based on robot pose.

    Parameters:
        robot_pos: [x, y, z] robot base position
        robot_euler: [roll, pitch, yaw] robot orientation in radians
        left_offset: local offset of left camera relative to robot base [dx, dy, dz]
        right_offset: local offset of right camera relative to robot base [dx, dy, dz]
        bound_center_offset: local offset of bound center relative to robot base [dx, dy, dz]
        bound_extents: half-sizes of bounds [ex, ey, ez], final bound size is 2*extents

    Returns:
        left_pos, right_pos, min_bound, max_bound (all numpy arrays)
    """
    robot_pos = np.array(robot_pos)
    robot_euler = np.array(robot_euler)

    R_mat = euler_to_rotation_matrix(robot_euler, order='xyz')

    left_pos = robot_pos + R_mat @ left_offset
    right_pos = robot_pos + R_mat @ right_offset

    bound_center = robot_pos + R_mat @ bound_center_offset

    rotated_extents = np.abs(R_mat) @ bound_extents
    min_bound = bound_center - rotated_extents
    max_bound = bound_center + rotated_extents

    return left_pos, right_pos, min_bound, max_bound


if __name__ == "__main__":
    workspace = [-1, 1, -1, 1, -1, 1]
    sampled_points = grid_sample(workspace, [5, 5], n_samples=10)
    visulize_grid_point(sampled_points)
