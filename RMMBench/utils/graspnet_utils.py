"""
GraspNet utility functions (new version migrated from graspnet/utils.py).

Includes: checkpoint loading (module-level singleton), point cloud retrieval,
graspnet inference, collision/position filtering, top-k selection,
visualization, coordinate conversion, prepare-point collision checking,
and the end-to-end pipeline graspnet_candidates.
"""
import sys
import os
import copy
import random

import numpy as np
import torch
import cv2
import open3d as o3d
from scipy.spatial.transform import Rotation as R

from graspnetAPI import GraspGroup
from graspnetAPI.utils.utils import CameraInfo, create_point_cloud_from_depth_image
from graspnet import GraspNet, pred_decode
from collision_detector import ModelFreeCollisionDetector

from RMMBench.utils.depth2cloud import posRotMat2Mat
from RMMBench.utils.utils import matrix_to_quaternion, gripper_collision_check
from RMMBench.utils.paths import REPO_ROOT

# graspnet-baseline under the repository root (RMMBench)
sys.path.append(os.path.join(str(REPO_ROOT), 'graspnet-baseline', 'models'))
sys.path.append(os.path.join(str(REPO_ROOT), 'graspnet-baseline', 'dataset'))
sys.path.append(os.path.join(str(REPO_ROOT), 'graspnet-baseline', 'utils'))


# ── Checkpoint loading (module-level singleton to avoid reloading on every inference) ──────────────────────────
_NET = None


def get_net(force_reload=False):
    """Load the GraspNet checkpoint; uses a module-level singleton by default, force-reloads when force_reload=True."""
    global _NET
    if _NET is not None and not force_reload:
        return _NET

    checkpoint_path = os.path.join(str(REPO_ROOT), 'graspnet-baseline', 'logs/log_rs/checkpoint-rs.tar')
    net = GraspNet(input_feature_dim=0, num_view=300, num_angle=12, num_depth=4,
                   cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01, 0.02, 0.03, 0.04], is_training=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    print("-> loaded checkpoint %s (epoch: %d)" % (checkpoint_path, start_epoch))
    net.eval()
    _NET = net
    return net


def get_graspnet_pcd(env, bbox=[0, 0, 480, 480], cam_id=3):
    """
    Capture RGB-D from camera in VLABench, obtain target_mask from bounding box, and process it into GraspNet input format.
    Returns:
        end_points: dict containing 'point_clouds' (torch tensor, 1×N×3)
        cloud: open3d.geometry.PointCloud for visualization
        is_cloud: False means the point cloud is empty after masking
    """
    # --- 1. Capture from camera ---
    rgb, depth, seg, intr, extr = env.get_rgbd(cam_id)
    rgb = np.ascontiguousarray(rgb.astype(np.float32) / 255.0)  # (H, W, 3)
    depth = np.ascontiguousarray(depth.astype(np.float32))  # (H, W)
    assert rgb.shape[:2] == depth.shape, "RGB and depth must have same shape"

    # --- 2. Retrieve intrinsics from VLABench camera model ---
    fx, fy = intr[0, 0], intr[1, 1]
    cx, cy = intr[0, 2], intr[1, 2]
    h, w = depth.shape
    factor_depth = 1.0  # assume depth already in meters

    camera = CameraInfo(width=w, height=h, fx=fx, fy=fy, cx=cx, cy=cy, scale=factor_depth)

    # --- 3. Generate point cloud from depth image (same as GraspNet) ---
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
    if cloud is None or cloud.size == 0:
        print("点云为空")

    # --- 4. Apply mask (degenerate bbox -> whole image; otherwise expanded by 50 pixels) ---
    ymin, xmin, ymax, xmax = bbox
    if ymin == ymax or xmin == xmax:
        target_mask = np.ones((h, w), dtype=bool)
    else:
        target_mask = np.zeros((h, w), dtype=bool)
        ymin = max([0, ymin - 50])
        xmin = max([0, xmin - 50])
        ymax = min([480, ymax + 50])
        xmax = min([480, xmax + 50])
        target_mask[ymin:ymax, xmin:xmax] = True
    robot_mask = np.where((seg[..., 0] <= 72) & (seg[..., 0] > 0), 0, 1).astype(np.uint8)
    mask = target_mask & (depth > 0.3) & (depth < 2) & (robot_mask > 0)
    cloud_masked = cloud[mask]
    color_masked = rgb[mask]

    if len(cloud_masked) == 0:
        print("Warning: cloud_masked is empty!")
        # The last False means the point cloud is invalid
        return None, None, intr, extr, False

    # --- 5. Randomly sample num_point points (GraspNet expects fixed number) ---
    num_point = 20000
    if len(cloud_masked) >= num_point:
        idxs = np.random.choice(len(cloud_masked), num_point, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), num_point - len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    cloud_sampled = cloud_masked[idxs]
    color_sampled = color_masked[idxs]

    # --- 6. Convert to Open3D point cloud for visualization ---
    cloud_o3d = o3d.geometry.PointCloud()
    cloud_o3d.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
    cloud_o3d.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))

    # --- 7. Convert to PyTorch tensor for GraspNet input ---
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32)).to(device)

    end_points = dict()
    end_points['point_clouds'] = cloud_sampled
    end_points['cloud_colors'] = color_sampled

    return end_points, cloud_o3d, intr, extr, True


def filter_grasps_bbox(gg, bbox, intr):
    """
    Filters a GraspGroup to only include grasps whose centers project into the 2D bounding box on the image plane
    """
    if len(gg) == 0:
        return gg

    centers = gg.translations

    fx, fy = intr[0, 0], intr[1, 1]
    cx, cy = intr[0, 2], intr[1, 2]
    x, y, z = centers[:, 0], centers[:, 1], centers[:, 2]
    u = (x / z) * fx + cx
    v = (y / z) * fy + cy

    ymin, xmin, ymax, xmax = bbox
    mask_u = (u >= xmin) & (u <= xmax)
    mask_v = (v >= ymin) & (v <= ymax)
    bbox_mask = mask_u & mask_v

    return gg[bbox_mask]


def run_graspnet_on_pcd(end_points, cloud, bbox=None, intr=None):
    net = get_net()

    with torch.no_grad():
        end_points = net(end_points)                 # model forward
        grasp_preds = pred_decode(end_points)        # list, [batch_size]
    gg_array = grasp_preds[0].detach().cpu().numpy()
    gg = GraspGroup(gg_array)

    # Filtering by 2D bounding box
    if bbox is not None and intr is not None:
        gg = filter_grasps_bbox(gg, bbox, intr)
    return gg


def filter_grasps_collision(gg, cloud, voxel_size=0.01, approach_dist=0.05, collision_thresh=0.01):
    mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=voxel_size)
    collision_mask = mfcdetector.detect(gg, approach_dist=approach_dist, collision_thresh=collision_thresh)
    gg = gg[~collision_mask]
    return gg


def filter_grasps_position(gg, x_p=None, y_p1=35, y_p2=65, z_p=10):
    """Camera-frame window filtering: use the [y_p1, y_p2] percentile window for the y axis, the z_p percentile as the z-axis upper limit, and an optional window for the x axis."""
    t = gg.translations  # (N, 3) numpy array

    # 1. Compute the z-axis and y-axis thresholds (always present)
    z_th = np.percentile(t[:, 2], z_p)
    y_low = np.percentile(t[:, 1], y_p1)
    y_high = np.percentile(t[:, 1], y_p2)

    # 2. Initialize the mask, starting with the Y and Z conditions
    mask = (t[:, 1] >= y_low) & (t[:, 1] <= y_high) & (t[:, 2] <= z_th)

    # 3. If x_p is provided, compute the x-axis thresholds and update the mask
    if x_p is not None:
        x_p1, x_p2 = x_p
        x_low = np.percentile(t[:, 0], x_p1)
        x_high = np.percentile(t[:, 0], x_p2)
        mask = mask & (t[:, 0] >= x_low) & (t[:, 0] <= x_high)

    return gg[mask]


def select_topk_grasps(gg, topk=5):
    """Take the top-k grasps by score in descending order."""
    return gg.sort_by_score()[:topk]


def vis_grasps(gg, cloud):
    grippers = gg.to_open3d_geometry_list()
    o3d.visualization.draw_plotly([cloud, *grippers])


def vis_gripper(rgb_img, p_grasp_cam, r_grasp_cam, intrinsic, gripper_w=0.08, gripper_d=0.06):
    '''Visualization of graspnet grasps on camera rgb_img'''
    img_viz = rgb_img.copy()

    # Define Local Gripper Points: Z+ is forward, X is the width of the gripper
    hw = gripper_w / 2
    depth_base = 0.02
    pts_local = np.array([
        [0, 0, 0],       # Base center
        [hw, 0, -depth_base],      # Right base
        [-hw, 0, -depth_base],     # Left base
        [hw, 0, -depth_base+gripper_d],  # Right tip
        [-hw, 0, -depth_base+gripper_d]  # Left tip
    ])

    # Transform Gripper to Camera Space
    pts_cam = (r_grasp_cam @ pts_local.T).T + p_grasp_cam

    # Project 3D Camera Space to 2D Pixels
    pts_2d_homo = (intrinsic @ pts_cam.T).T
    pts_2d = (pts_2d_homo[:, :2] / pts_2d_homo[:, 2:3]).astype(int)

    color = (0, 255, 0)  # Green
    cv2.line(img_viz, tuple(pts_2d[1]), tuple(pts_2d[2]), color, 2)  # Base
    cv2.line(img_viz, tuple(pts_2d[1]), tuple(pts_2d[3]), color, 2)  # Right finger
    cv2.line(img_viz, tuple(pts_2d[2]), tuple(pts_2d[4]), color, 2)  # Left finger

    return img_viz


def graspnet_rot_correction(rot_matrix):
    R = rot_matrix
    z = copy.deepcopy(R[:, 2])
    y = copy.deepcopy(R[:, 1])
    x = copy.deepcopy(R[:, 0])

    R_new = np.zeros((3, 3), dtype=R.dtype)
    R_new[:, 0] = y
    R_new[:, 1] = z
    R_new[:, 2] = x
    return R_new


def graspnet_grasp_to_pose(p_grasp, r_grasp, T_W_C):
    """
    - p_grasp: position of graspnet grasp from camera view (camera frame: Z+ forward, X+ right, Y+ up)
    - r_grasp: rotation of graspnet grasp
    - T_W_C: 4x4 Transformation matrix from Camera (C) to World (W) frame
    Returns: world_position, world_quaternion
    """
    T_C_G = posRotMat2Mat(p_grasp, r_grasp)

    Rx = np.eye(4)
    Rx[:3, :3] = o3d.geometry.get_rotation_matrix_from_xyz([np.pi, 0.0, 0.0])
    T_C_G_flip = Rx @ T_C_G  # rotate 180° around x-axis for tranforming graspnet camera pose to mujoco's real camera pose

    T_W_G = T_W_C @ T_C_G_flip
    Rz = np.eye(4)
    Rz[:3, :3] = o3d.geometry.get_rotation_matrix_from_xyz([0.0, 0.0, -np.pi * 3 / 2])
    T_W_G = T_W_G @ Rz

    world_position = T_W_G[:3, 3]  # Position of the gripper origin in World frame
    world_rotation_matrix = T_W_G[:3, :3]  # Gripper orientation

    world_quaternion = matrix_to_quaternion(world_rotation_matrix)

    return world_position, world_quaternion


def prepare_grasp_graspnet(env, key_pos, key_quat, grasp_depth, std=0, max_retry=100, move_vector=None, grasp_choice=None):
    """Move the key point forward by grasp_depth, compute the prepare point, and run gripper point-cloud collision checks at both locations (perturb and retry on failure)."""
    valid = False
    retry = 0
    # Note: grasp_choice==3/4 overrides key_quat with an expert pose; do not pass this argument when calling from the graspnet pipeline
    if grasp_choice == 3:
        key_quat = R.from_euler('xyz', [np.pi/2, -np.pi, 0]).as_quat()
    elif grasp_choice == 4:
        key_quat = R.from_euler('xyz', [np.pi/2, -(5*np.pi)/6, 0]).as_quat()

    while not valid and retry < max_retry:
        # Optionally perturb quaternion (small random rotation)
        if retry > 0:
            euler = R.from_quat(key_quat).as_euler('xyz')
            print("euler angle is", euler)
            euler += np.random.normal(0, std, 3)
            pert_quat = R.from_euler('xyz', euler).as_quat()
        else:
            euler = R.from_quat(key_quat).as_euler('xyz')
            print("euler angle is", euler)
            pert_quat = key_quat

        gripper_pcd, gripper_forward_vector = env.robot.gripper_pcd(key_pos, key_quat)
        if move_vector is None:
            move_vector = gripper_forward_vector

        key_pos = key_pos + move_vector * grasp_depth
        distance = -0.1
        prepare_pos = key_pos + move_vector * distance

        # Get environment and gripper point clouds for collision checking
        env_pcd = env.get_observation()["masked_point_cloud"]
        prepare_gripper_pcd = copy.deepcopy(gripper_pcd).translate(move_vector * distance)
        prepare_gripper_pcd.paint_uniform_color([0, 1, 0])

        collision_key = gripper_collision_check(gripper_pcd, env_pcd)
        collision_prepare = gripper_collision_check(prepare_gripper_pcd, env_pcd)

        if not collision_key and not collision_prepare:
            valid = True
            key_quat = pert_quat
            break
        retry += 1

    if not valid:
        print("No valid grasp found for this candidate after", retry, "tries.")
        return None
    else:
        return key_pos, prepare_pos, key_quat


def graspnet_candidates(env, bbox=None, topk=2, prepare_quat=None,
                        voxel_size=0.01, approach_dist=0.05, collision_thresh=0.01,
                        grasp_seed=None):
    """
    End-to-end graspnet pipeline: point cloud retrieval → graspnet inference →
    coarse collision filtering → progressively relaxed position filtering → top-k
    → per-candidate coordinate conversion (camera frame → world frame) + prepare-point collision check.
    Produces candidate dicts isomorphic to the expert keypoint candidates
    (key_pos/prepare_pos/key_quat/viz).

    Returns:
        candidates: list of {"source": "graspnet", "key_pos", "prepare_pos", "key_quat", "viz"}
                    Candidates whose prepare check fails are dropped;
                    prepare_grasp_graspnet is called without grasp_choice, to avoid expert poses overriding the graspnet pose
        flag: False only when the point cloud is empty, True otherwise (including when no usable candidates remain)
    """
    # Fix the random seed to ensure consistent GraspNet output
    if grasp_seed is not None:
        np.random.seed(grasp_seed)
        random.seed(grasp_seed)
        torch.manual_seed(grasp_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(grasp_seed)

    end_points, cloud, cam_intr, cam_extr, is_cloud = get_graspnet_pcd(env, bbox=bbox, cam_id=3)
    if is_cloud is False:
        print("The point cloud is empty")
        return [], False

    gg = run_graspnet_on_pcd(end_points, cloud, bbox=bbox, intr=cam_intr)
    print("len(gg):", len(gg))
    if len(gg) == 0:
        return [], True

    gg_col = filter_grasps_collision(gg, np.array(cloud.points), voxel_size=voxel_size,
                                     approach_dist=approach_dist, collision_thresh=collision_thresh)
    if len(gg_col) == 0:
        print("Warning: All grasps filtered out by collision detection")
        return [], True

    # Position filtering with progressive relaxation; fallback order: relaxed result → collision-only result → empty
    gg_pos = None
    y_p1, y_p2, z_p, x_p = 60, 95, 10, [30, 70]
    LIMIT_MIN, LIMIT_MAX, max_expand_iter = 0, 100, 50
    expand_iter = 0
    while gg_pos is None:
        current_grasps = filter_grasps_position(gg_col, x_p=x_p, y_p1=y_p1, y_p2=y_p2, z_p=z_p)
        if len(current_grasps) < topk:
            y_p1 = max(LIMIT_MIN, y_p1 - 1)
            z_p = min(LIMIT_MAX, z_p + 1)
            x_p = [max(LIMIT_MIN, x_p[0] - 1), min(LIMIT_MAX, x_p[1] + 1)]
            expand_iter += 1
            is_at_limit = (y_p1 == LIMIT_MIN and z_p == LIMIT_MAX and
                           x_p[0] == LIMIT_MIN and x_p[1] == LIMIT_MAX)
            if expand_iter >= max_expand_iter or is_at_limit:
                if len(current_grasps) > 0:
                    gg_pos = current_grasps
                    print(f"Warning: Using relaxed grasps after {expand_iter} iterations, got {len(current_grasps)} grasps")
                elif len(gg_col) > 0:
                    gg_pos = gg_col
                    print(f"Warning: Fallback to collision-only grasps, got {len(gg_col)} grasps")
                else:
                    return [], True
                break
        else:
            gg_pos = current_grasps
    print(f"y_p1:{y_p1},z_p:{z_p},x_p:{x_p}")

    # Per candidate: visualization + world-frame pose conversion + prepare-point collision check (reusing the same rendered image)
    rgb_img = cv2.cvtColor(env.render(camera_id=3, height=480, width=480), cv2.COLOR_BGR2RGB)
    candidates = []
    for g in select_topk_grasps(gg_pos, topk=topk):
        grasp_rotation = graspnet_rot_correction(g.rotation_matrix)
        viz = vis_gripper(rgb_img, g.translation, grasp_rotation, cam_intr, g.width, g.depth)
        key_pos, key_quat = graspnet_grasp_to_pose(g.translation, grasp_rotation, cam_extr)
        res = prepare_grasp_graspnet(env, key_pos, key_quat, g.depth, move_vector=prepare_quat)
        if res is None:
            print("Warning: graspnet candidate failed prepare collision check, dropped")
            continue
        key_pos, prepare_pos, key_quat = res
        candidates.append({"source": "graspnet", "key_pos": key_pos,
                           "prepare_pos": prepare_pos, "key_quat": key_quat, "viz": viz})
    return candidates, True
