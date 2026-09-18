import glob
import os
import sys

import numpy as np
import open3d as o3d
import cv2
from dm_control import viewer

from RMMBench.envs import load_env
from RMMBench.tasks import *
from utils.utils import save_bev,smooth_and_rectangularize
import json

from RMMBench.utils.utils import extract_containers,get_geom_ids_by_prefix,get_entity_mask_from_seg,get_placement_top_down_view

from pathlib import Path

from RMMBench.robots.single_arm.franka import Franka
def all_cam_views(env):
    # Test camera views
    rgb_data = env.get_observation()
    # Cameras 0-2: fixed camera views, corresponding to right, left, and forward respectively; cameras 3 and 4 correspond to wrist and hand
    n = 1
    vlm_img_input = [cv2.cvtColor(rgb_data["rgb"][i], cv2.COLOR_BGR2RGB) for i in range(6)]
    img_path = "/Users/lh/work/RMMBench/RMMBench/utils/map/test_data/test_image/all_cam_view"
    cv2.imwrite(os.path.join(img_path, f"wrist_img_{n}.png"), vlm_img_input[3])
    cv2.imwrite(os.path.join(img_path, f"right_img_{n}.png"), vlm_img_input[0])
    cv2.imwrite(os.path.join(img_path, f"left_img_{n}.png"), vlm_img_input[1])
    cv2.imwrite(os.path.join(img_path, f"opposite_img_{n}.png"), vlm_img_input[5])
    cv2.imwrite(os.path.join(img_path, f"head_img_{n}.png"), vlm_img_input[4])
    cv2.imwrite(os.path.join(img_path, f"top_img_{n}.png"), vlm_img_input[2])



def print_env_info(env):
    instruction = env.task.get_instruction()
    print("instruction:", instruction)

def rgb_2_pcd_mask_robot(env,cam_id=2):
    rgb, depth, seg, intrinsic, extrinsic = env.get_camera_parse(cam_id)
    h, w = depth.shape

    segmentation = np.array(seg)
    robot_mask = np.where((segmentation[..., 0] <= 72) & (segmentation[..., 0] > 0), 0, 1).astype(np.uint8)
    depth_img = np.ascontiguousarray(depth)
    color_img = np.ascontiguousarray(rgb)
    depth = np.ascontiguousarray(depth_img * robot_mask)
    rgb = np.ascontiguousarray(color_img * np.repeat(robot_mask[:, :, np.newaxis], 3, axis=2))

    # Generate a camera-frame point cloud with Open3D
    od_cammat = o3d.camera.PinholeCameraIntrinsic(w, h, intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2])
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(np.ascontiguousarray(rgb)),
        o3d.geometry.Image(np.ascontiguousarray(depth)),
        convert_rgb_to_intensity=False,
        depth_scale=1.0,
        depth_trunc=10.0,
    )
    cam_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, od_cammat)
    # Transform to the world frame: extrinsic is camera->world, so transform directly
    world_pcd = cam_pcd.transform(extrinsic)
    return world_pcd

def rgb_2_pcd(env,cam_id=2):
    rgb, depth, seg, intrinsic, extrinsic = env.get_camera_parse(cam_id)
    h, w = depth.shape

    # Generate a camera-frame point cloud with Open3D
    od_cammat = o3d.camera.PinholeCameraIntrinsic(w, h, intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2])
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(np.ascontiguousarray(rgb)),
        o3d.geometry.Image(np.ascontiguousarray(depth)),
        convert_rgb_to_intensity=False,
        depth_scale=1.0,
        depth_trunc=10.0,
    )
    cam_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, od_cammat)
    # Transform to the world frame: extrinsic is camera->world, so transform directly
    world_pcd = cam_pcd.transform(extrinsic)
    return world_pcd

def pcd_filter(pcd, z_min=None, z_max=None, x_min=None, x_max=None, y_min=None, y_max=None):
    """
        Filter the point cloud by axial ranges; each axis is judged independently,
        and a parameter of None skips that condition.

        Parameters:
            pcd: Open3D point cloud object
            x_min, x_max: x-axis range; None means no limit
            y_min, y_max: y-axis range; None means no limit
            z_min, z_max: z-axis range; None means no limit
        Returns:
            The filtered Open3D point cloud
    """
    points = np.asarray(pcd.points)
    mask = np.ones(len(points), dtype=bool)

    if x_min is not None:
        mask &= (points[:, 0] >= x_min)
    if x_max is not None:
        mask &= (points[:, 0] <= x_max)
    if y_min is not None:
        mask &= (points[:, 1] >= y_min)
    if y_max is not None:
        mask &= (points[:, 1] <= y_max)
    if z_min is not None:
        mask &= (points[:, 2] >= z_min)
    if z_max is not None:
        mask &= (points[:, 2] <= z_max)

    filtered_pcd = pcd.select_by_index(np.where(mask)[0])
    return filtered_pcd



def pcd_to_bev(pcd,x_range=None,y_range=None, resolution=0.05, min_points=7):
    """Project a point cloud onto a BEV occupancy grid map
    resolution: physical size of each grid cell, in meters
    min_points: minimum number of points a cell must contain to be considered an obstacle.
    """
    points = np.asarray(pcd.points)[:, :2]  # only take x, y

    if x_range is None:
        x_range = (points[:, 0].min(), points[:, 0].max())
    if y_range is None:
        y_range = (points[:, 1].min(), points[:, 1].max())

    width = int((x_range[1] - x_range[0]) / resolution) + 1
    height = int((y_range[1] - y_range[0]) / resolution) + 1

    bev_count = np.zeros((height, width), dtype=np.int32)

    px = ((points[:, 0] - x_range[0]) / resolution).astype(int)
    py = ((points[:, 1] - y_range[0]) / resolution).astype(int)

    valid = (px >= 0) & (px < width) & (py >= 0) & (py < height)
    np.add.at(bev_count, (py[valid], px[valid]), 1)

    bev = (bev_count >= min_points).astype(np.uint8)
    # save_bev(bev, resolution, min_points,path = "/Users/lh/work/RMMBench/RMMBench/utils/map/test_data/test_image/bev")
    return bev, x_range, y_range, resolution,min_points

def find_max_bev(bev):
    # bev: 1 = obstacle, 0 = free space
    # Invert: 1 = free space, 0 = obstacle
    free = (1 - bev).astype(np.uint8)
    # Connected component analysis
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(free, connectivity=4)
    # Find the largest free connected component (skip label 0, which is the background)
    # Label 0 is the largest black region (obstacle); we want the largest white region
    max_area = 0
    max_label = 0
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > max_area:
            max_area = area
            max_label = i
    # Extract the mask of the largest connected component
    free_mask = (labels == max_label).astype(np.uint8)
    return free_mask

task = 'select_bread_to_plate'
robot = "pandaomron"
env = load_env(task,robot=robot,time_limit=1000)
# env = load_env(task,robot=robot,episode_config=config,time_limit=1000)
env.reset()
all_cam_views(env)
print_env_info(env)

# Point cloud visualization:
pcd_mask_robot = rgb_2_pcd_mask_robot(env,cam_id=2)
pcd_mask_robot = pcd_filter(pcd_mask_robot,z_min=0.5,z_max=2)

bev_filter_ground,_,_,r,m = pcd_to_bev(pcd_mask_robot)
bev_filter_ground = find_max_bev(bev_filter_ground)
bev_filter_ground = smooth_and_rectangularize(bev_filter_ground)
save_bev(bev_filter_ground, r, m,path="/Users/lh/work/RMMBench/RMMBench/utils/map/test_data/test_image/bev")

exit()
o3d.visualization.draw_geometries([pcd])


#




viewer.launch(env)# automatically performs one reset

env.close()
