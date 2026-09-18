import math
import numpy as np
import cv2
import open3d as o3d

# =============================================================================
# Navigation ray visualization configuration
# =============================================================================
RAY_ANGLES_DEG = [-90, -60, -30, 0, 30, 60, 90]
RAY_ANGLES = np.array(RAY_ANGLES_DEG) * math.pi / 180

RAY_LENGTH_DEFAULT = 0.95
RAY_LENGTH_MIN = 0.4
SCALE_LENGTH = 0.05  # Horizontal length of the scale tick
SCALE_POSITIONS = [0.9]  # Scale tick positions along the ray (meters)

# Collision detection parameters (tunable)
COLLISION_Z_TH = 0.9       # Collision detection height (rays are drawn at z=0, detection is done at z=z_th)
COLLISION_RADIUS = 0.1     # Collision detection radius in the XY plane
COLLISION_Z_TOLERANCE = 0.2  # Tolerance range along the z direction

ROBOT_RADIUS = 0.3

# =============================================================================
# Navigation grid visualization configuration
# =============================================================================
GRID_CELL_SIZE_DEFAULT = 0.4       # Default grid cell size (meters)
GRID_FRONT_DEFAULT = 1.0           # Default range in front of the robot (meters)
GRID_BACK_DEFAULT = 1            # Default range behind the robot (meters)
GRID_SIDE_DEFAULT = 1.0            # Default range to the robot's left/right (meters, per side)
GRID_Z_TH_DEFAULT = 0.3            # Default obstacle height threshold (meters)
GRID_MIN_PIXEL_SIZE = 20           # Minimum grid pixel size; labels are not drawn below this value
GRID_TEXT_SCALE = 0.35             # Grid label font size
GRID_TEXT_THICKNESS = 1            # Grid label font thickness
GRID_LINE_COLOR = (0, 0, 0)        # Grid line color (BGR black)
GRID_FILL_COLOR = (200, 200, 255)  # Grid fill color (BGR light blue)
GRID_TEXT_COLOR = (255, 255, 255)  # Grid label color (BGR white)
GRID_ALPHA = 0.15                  # Grid fill transparency
GRID_LINE_THICKNESS = 1            # Grid line width
GRID_OBSTACLE_RADIUS = 0.15        # Obstacle detection radius (meters)


def _draw_dashed_line(img, pt1, pt2, color, thickness, dash_length=10):
    """Draw a dashed line on the image."""
    x1, y1 = pt1
    x2, y2 = pt2
    dx = x2 - x1
    dy = y2 - y1
    dist = math.hypot(dx, dy)
    if dist < 1e-6:
        return
    steps = int(dist / dash_length)
    for i in range(0, steps, 2):
        start_t = i / steps
        end_t = min((i + 1) / steps, 1.0)
        sx = int(x1 + dx * start_t)
        sy = int(y1 + dy * start_t)
        ex = int(x1 + dx * end_t)
        ey = int(y1 + dy * end_t)
        cv2.line(img, (sx, sy), (ex, ey), color, thickness)


def project_world_to_image(world_pts, intrinsic, extrinsic_inv, img_w, img_h,
                           return_in_front_only=False):
    """
    Project 3D points in world coordinates onto the image plane, returning
    pixel coordinates and a visibility mask.

    Parameters
    ----------
    world_pts : np.ndarray, shape (N, 3)
        Points in world coordinates.
    intrinsic : np.ndarray, shape (3, 3)
        Camera intrinsic matrix.
    extrinsic_inv : np.ndarray, shape (4, 4)
        Inverse of the extrinsic matrix (i.e., the world -> camera transform).
        Note: in RMMBench, extrinsic is camera -> world, so projection requires
        np.linalg.inv(extrinsic).
    img_w, img_h : int
        Image width and height.
    return_in_front_only : bool
        If True, valid_mask only checks that points are in front of the camera
        (z > 0), without checking image bounds.
        Used when out-of-bounds projected coordinates are needed.

    Returns
    -------
    uv : np.ndarray, shape (N, 2)
        Pixel coordinates (u, v).
    valid_mask : np.ndarray, shape (N,), dtype bool
        Mask of points in front of the camera (and optionally within image bounds).
    """
    world_pts = np.asarray(world_pts)
    if world_pts.ndim == 1:
        world_pts = world_pts.reshape(1, -1)

    # Convert to homogeneous coordinates
    ones = np.ones((world_pts.shape[0], 1))
    world_homo = np.concatenate([world_pts, ones], axis=1)  # (N, 4)

    # World -> camera
    cam_homo = (extrinsic_inv @ world_homo.T).T  # (N, 4)
    cam_pts = cam_homo[:, :3]

    # Keep only points in front of the camera (z > 0)
    in_front = cam_pts[:, 2] > 1e-6

    # Camera coordinates -> image coordinates
    uv_homo = (intrinsic @ cam_pts.T).T  # (N, 3)
    z = uv_homo[:, 2:3]
    z = np.where(np.abs(z) < 1e-6, 1e-6, z)
    uv = uv_homo[:, :2] / z  # (N, 2)

    if return_in_front_only:
        valid_mask = in_front
    else:
        # Check whether the point is within image bounds
        in_bounds = (uv[:, 0] >= 0) & (uv[:, 0] < img_w) & (uv[:, 1] >= 0) & (uv[:, 1] < img_h)
        valid_mask = in_front & in_bounds
    return uv, valid_mask


def _clip_line_to_image_bounds(uv1, uv2, img_w, img_h):
    """
    Clip a 2D line segment to the image bounds using the Liang-Barsky algorithm.

    Parameters
    ----------
    uv1, uv2 : tuple or np.ndarray
        Pixel coordinates (u, v) of the two segment endpoints.
    img_w, img_h : int
        Image width and height.

    Returns
    -------
    clipped1, clipped2 : tuple or None
        The two clipped endpoints; returns (None, None) if the segment lies
        entirely outside the image.
    """
    x1, y1 = float(uv1[0]), float(uv1[1])
    x2, y2 = float(uv2[0]), float(uv2[1])

    dx = x2 - x1
    dy = y2 - y1

    p = [-dx, dx, -dy, dy]
    q = [x1 - 0, img_w - 1 - x1, y1 - 0, img_h - 1 - y1]

    u1, u2 = 0.0, 1.0

    for i in range(4):
        if p[i] == 0:
            if q[i] < 0:
                return None, None
        else:
            t = q[i] / p[i]
            if p[i] < 0:
                u1 = max(u1, t)
            else:
                u2 = min(u2, t)

    if u1 > u2:
        return None, None

    cx1 = x1 + dx * u1
    cy1 = y1 + dy * u1
    cx2 = x1 + dx * u2
    cy2 = y1 + dy * u2

    return (int(round(cx1)), int(round(cy1))), (int(round(cx2)), int(round(cy2)))


def compute_ray_lengths(ray_origin, robot_yaw, obstacle_pcd=None,
                        ray_angles=RAY_ANGLES,
                        default_length=RAY_LENGTH_DEFAULT,
                        min_length=RAY_LENGTH_MIN,
                        robot_radius=ROBOT_RADIUS,
                        collision_z_th=COLLISION_Z_TH,
                        collision_radius=COLLISION_RADIUS,
                        collision_z_tolerance=COLLISION_Z_TOLERANCE):
    """
    Compute the actual length of each ray: if an obstacle point cloud is provided,
    detect collisions along the ray and truncate it just before the obstacle;
    otherwise use the default length.

    Parameters
    ----------
    ray_origin : np.ndarray, shape (3,)
        Ray origin (world coordinates), usually the chassis center [x, y, 0.0].
    robot_yaw : float
        Current robot heading (radians, around the Z axis).
    obstacle_pcd : open3d.geometry.PointCloud or np.ndarray or None
        Obstacle point cloud. If None, all rays use default_length.
    ray_angles : np.ndarray
        Ray angle offsets relative to robot_yaw (radians).
    default_length : float
        Default ray length when no obstacle is present.
    min_length : float
        Minimum ray length (avoids being drawn right up against the view).
    robot_radius : float
        Robot radius, used to keep a distance from obstacles during collision detection.

    Returns
    -------
    ray_lengths : np.ndarray, shape (len(ray_angles),)
        Actual length of each ray.
    """
    n_rays = len(ray_angles)
    ray_lengths = np.full(n_rays, default_length)

    if obstacle_pcd is None:
        return np.maximum(ray_lengths, min_length)

    # Convert to a numpy array (M, 3)
    if hasattr(obstacle_pcd, 'points'):
        obs_pts = np.asarray(obstacle_pcd.points)
    else:
        obs_pts = np.asarray(obstacle_pcd)


    if obs_pts.shape[0] == 0:
        return np.maximum(ray_lengths, min_length)

    for i, angle_offset in enumerate(ray_angles):
        total_angle = robot_yaw + angle_offset
        direction = np.array([math.cos(total_angle), math.sin(total_angle), 0.0])

        vec_to_obs = obs_pts - ray_origin  # (M, 3)
        proj_len = np.dot(vec_to_obs, direction)  # (M,)

        # Collision detection: ignore points within robot_radius of ray_origin (the robot itself)
        valid = (proj_len > 0) & (np.abs(obs_pts[:, 2] - collision_z_th) < collision_z_tolerance)
        n_valid = np.sum(valid)

        if n_valid > 0:
            proj_pts = ray_origin + np.outer(proj_len, direction)
            perp_dists_xy = np.linalg.norm(obs_pts[:, :2] - proj_pts[:, :2], axis=1)
            collision_mask = valid & (perp_dists_xy < collision_radius)
            n_collision = np.sum(collision_mask)

            if n_collision > 0:
                collision_proj = proj_len[collision_mask]
                min_collision_dist = np.min(collision_proj)
                ray_lengths[i] = max(min_length, min(min_collision_dist, default_length))
    #             print(f"  [debug] ray {i}: min_collision_dist={min_collision_dist:.6f}, min_length={min_length}, ray_length={ray_lengths[i]:.6f}")
    #         else:
    #             print(f"[compute_ray_lengths] ray {i} (angle={math.degrees(angle_offset):.0f}°): {n_valid} valid points but no collision (radius={collision_radius})")
    #     else:
    #         print(f"[compute_ray_lengths] ray {i} (angle={math.degrees(angle_offset):.0f}°): no valid points in z range")
    #
    # for i in range(len(ray_angles)):
    #     print(f"[compute_ray_lengths] ray {i} (angle={math.degrees(ray_angles[i]):.0f}°): length={ray_lengths[i]:.3f}m")

    return ray_lengths


def draw_navigation_overlay(rgb, robot_pos, robot_yaw, intrinsic, extrinsic,
                            obstacle_pcd=None,
                            ray_angles_deg=RAY_ANGLES_DEG,
                            default_length=RAY_LENGTH_DEFAULT,
                            min_length=RAY_LENGTH_MIN,
                            robot_radius=ROBOT_RADIUS,
                            scale_length=SCALE_LENGTH,
                            # scale_start_offset=SCALE_START_OFFSET,
                            ray_color=(0, 255, 0),
                            scale_color=(0, 0, 255),
                            text_color=(255, 255, 255),
                            ray_thickness=2,
                            scale_thickness=2):
    """
    Draw navigation rays and 0.5m reference scale ticks on the head camera RGB image.

    Parameters
    ----------
    rgb : np.ndarray, shape (H, W, 3)
        Head camera RGB image (uint8).
    robot_pos : np.ndarray or list, shape (3,)
        Robot chassis center in world coordinates [x, y, z]. The ray origin is [x, y, 0.0].
    robot_yaw : float
        Current robot heading (radians).
    intrinsic : np.ndarray, shape (3, 3)
        Camera intrinsic matrix.
    extrinsic : np.ndarray, shape (4, 4)
        Camera extrinsic matrix (camera -> world).
    obstacle_pcd : open3d.geometry.PointCloud or np.ndarray or None
        Obstacle point cloud, used to dynamically truncate ray lengths.
    ray_angles_deg : list of float
        Ray angle offsets in degrees, relative to robot_yaw.
    default_length, min_length, robot_radius : float
        See compute_ray_lengths.
    scale_length : float
        Reference scale tick length (meters).
    scale_start_offset : float
        Offset of the scale tick start point along the ray (meters).
    ray_color, scale_color, text_color : tuple
        BGR colors.
    ray_thickness, scale_thickness : int
        Line width used for drawing.

    Returns
    -------
    overlay_img : np.ndarray, shape (H, W, 3)
        Image with rays and scale ticks drawn (uint8).
    ray_info : list of dict
        Info for each ray, containing angle_deg, length, uv_start, uv_end, for use by callers.
    """
    img = rgb.copy()
    img_h, img_w = img.shape[:2]

    # Extrinsic: camera -> world; take its inverse for projection
    extrinsic_inv = np.linalg.inv(extrinsic)

    # Ray origin: chassis center, z=0
    ray_origin = np.array([robot_pos[0], robot_pos[1], 0])

    # Compute ray lengths
    ray_angles = np.array(ray_angles_deg) * math.pi / 180
    ray_lengths = compute_ray_lengths(
        ray_origin, robot_yaw, obstacle_pcd,
        ray_angles=ray_angles,
        default_length=default_length,
        min_length=min_length,
        robot_radius=robot_radius
    )

    # First collect all 3D points that need to be projected
    world_points = []
    point_metadata = []  # Record which ray each point belongs to and its type (start/end/scale_start/scale_end)

    for i, angle_offset in enumerate(ray_angles):
        total_angle = robot_yaw + angle_offset
        direction = np.array([math.cos(total_angle), math.sin(total_angle), 0.0])

        # Ray origin
        world_points.append(ray_origin)
        point_metadata.append({'ray_idx': i, 'type': 'start'})

        # Ray endpoint
        end_pt = ray_origin + direction * ray_lengths[i]
        world_points.append(end_pt)
        point_metadata.append({'ray_idx': i, 'type': 'end'})

        # Scale ticks: drawn at SCALE_POSITIONS, perpendicular to the ray direction
        for sp in SCALE_POSITIONS:
            if ray_lengths[i] > sp + 0.01:  # the tick must be before the ray endpoint
                scale_center = ray_origin + direction * sp
                perp_direction = np.array([-direction[1], direction[0], 0.0])
                scale_s = scale_center - perp_direction * (scale_length / 2)
                scale_e = scale_center + perp_direction * (scale_length / 2)
                world_points.extend([scale_s, scale_e])
                point_metadata.extend([
                    {'ray_idx': i, 'type': f'scale_{sp}_start'},
                    {'ray_idx': i, 'type': f'scale_{sp}_end'}
                ])

    # Batch-project onto the image (out-of-bounds allowed; only requires being in front of the camera)
    world_points = np.array(world_points)
    uv, valid_mask = project_world_to_image(
        world_points, intrinsic, extrinsic_inv, img_w, img_h,
        return_in_front_only=True
    )

    # Organize drawing by ray
    ray_info = []
    for i in range(len(ray_angles)):
        # Collect all points for this ray
        ray_pts = {}
        for j, meta in enumerate(point_metadata):
            if meta['ray_idx'] == i:
                ray_pts[meta['type']] = (uv[j], valid_mask[j])

        if 'start' not in ray_pts or 'end' not in ray_pts:
            continue

        uv_s, valid_s = ray_pts['start']
        uv_e, valid_e = ray_pts['end']

        if not valid_s or not valid_e:
            continue

        # Clip the segment to the image bounds with Liang-Barsky to ensure correct direction
        pt_s, pt_e = _clip_line_to_image_bounds(uv_s, uv_e, img_w, img_h)
        if pt_s is None or pt_e is None:
            continue

        # Check whether this is a world-frame X/Y direction (+X/-X or +Y/-Y)
        total_angle = robot_yaw + ray_angles[i]
        is_xy = (abs(total_angle) < 0.1 or
                 abs(abs(total_angle) - math.pi) < 0.1 or
                 abs(abs(total_angle) - math.pi / 2) < 0.1)
        this_ray_color = (128, 0, 128) if is_xy else ray_color  # RGB dark purple

        # Draw the dashed ray
        _draw_dashed_line(img, pt_s, pt_e, this_ray_color, ray_thickness, dash_length=10)

        # Draw scale ticks (at 0.5m and 1.0m): clip first, then draw
        for sp in SCALE_POSITIONS:
            start_key = f'scale_{sp}_start'
            end_key = f'scale_{sp}_end'
            if start_key not in ray_pts or end_key not in ray_pts:
                continue
            uv_ss, valid_ss = ray_pts[start_key]
            uv_se, valid_se = ray_pts[end_key]
            if not valid_ss or not valid_se:
                continue
            pt_ss, pt_se = _clip_line_to_image_bounds(uv_ss, uv_se, img_w, img_h)
            if pt_ss is None or pt_se is None:
                continue
            cv2.line(img, pt_ss, pt_se, scale_color, ray_thickness)

        # Record ray info (including tick coordinates, for draw_rays_on_image to reuse)
        info = {
            'angle_deg': ray_angles_deg[i],
            'total_angle': total_angle,            # Absolute heading angle in world coordinates (radians), used by draw_rays_on_image for color selection
            'robot_yaw': robot_yaw,                # Robot yaw (radians), kept so draw_rays_on_image can fall back on it
            'length': float(ray_lengths[i]),
            'uv_start': pt_s,
            'uv_end': pt_e,
            'valid': bool(valid_s and valid_e)
        }
        for sp in SCALE_POSITIONS:
            start_key = f'scale_{sp}_start'
            end_key = f'scale_{sp}_end'
            if start_key not in ray_pts or end_key not in ray_pts:
                continue
            uv_ss, valid_ss = ray_pts[start_key]
            uv_se, valid_se = ray_pts[end_key]
            if not valid_ss or not valid_se:
                continue
            pt_ss, pt_se = _clip_line_to_image_bounds(uv_ss, uv_se, img_w, img_h)
            if pt_ss is None or pt_se is None:
                continue
            info[start_key] = pt_ss
            info[end_key] = pt_se
        ray_info.append(info)

    return img, ray_info


def draw_rays_on_image(rgb, ray_info, ray_color=(0, 255, 0), scale_color=(0, 0, 255),
                        ray_thickness=2, dash_length=10):
    """
    Draw precomputed ray info (uv coordinates) onto the given image.
    Used to copy rays computed from a chassis-free image onto a chassis-included image.

    Parameters
    ----------
    rgb : np.ndarray
        Target image (head camera image with the chassis visible).
    ray_info : list of dict
        List of ray info dicts containing uv_start, uv_end, angle_deg, etc.
    ray_color, scale_color : tuple
        BGR colors.
    ray_thickness : int
        Ray line width.
    dash_length : int
        Dash segment length (pixels).

    Returns
    -------
    img : np.ndarray
        Image with rays drawn.
    """
    img = rgb.copy()
    for idx, info in enumerate(ray_info):
        pt_s = info['uv_start']
        pt_e = info['uv_end']

        # Check whether this is a world-frame X/Y direction (+X/-X or +Y/-Y)
        # Prefer total_angle (which includes robot_yaw), consistent with draw_navigation_overlay
        total_angle = info.get('total_angle', None)
        if total_angle is not None:
            check_angle = total_angle
        else:
            # Backward compatibility: old ray_info lacks total_angle; approximate with angle_deg
            check_angle = info.get('angle_deg', 0) * math.pi / 180
        is_xy = (abs(check_angle) < 0.1 or
                 abs(abs(check_angle) - math.pi) < 0.1 or
                 abs(abs(check_angle) - math.pi / 2) < 0.1)
        this_ray_color = (128, 0, 128) if is_xy else ray_color  # dark purple

        _draw_dashed_line(img, pt_s, pt_e, this_ray_color, ray_thickness, dash_length)

        # Draw scale ticks if their info is present
        scale_count = 0
        for key in info:
            if 'scale_' in key and '_start' in key:
                sp = key.replace('_start', '')
                end_key = sp + '_end'
                if end_key in info:
                    pt_ss = info[key]
                    pt_se = info[end_key]
                    cv2.line(img, pt_ss, pt_se, scale_color, ray_thickness)
    return img


def visualize_pcd(pcd, robot_pos=None, ray_origin=None, window_name="PointCloud"):
    """
    Visualize a point cloud, optionally marking the robot position and ray origin.

    Parameters
    ----------
    pcd : open3d.geometry.PointCloud or np.ndarray
        Point cloud data.
    robot_pos : np.ndarray or list, optional
        Robot position [x, y, z], marked in red.
    ray_origin : np.ndarray or list, optional
        Ray origin [x, y, z], marked in green.
    window_name : str
        Window title.
    """
    import open3d as o3d

    if hasattr(pcd, 'points'):
        vis_pcd = pcd
    else:
        vis_pcd = o3d.geometry.PointCloud()
        vis_pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd))

    geometries = [vis_pcd]

    if robot_pos is not None:
        robot_pt = o3d.geometry.PointCloud()
        robot_pt.points = o3d.utility.Vector3dVector([np.asarray(robot_pos)])
        robot_pt.colors = o3d.utility.Vector3dVector([[1.0, 0.0, 0.0]])
        geometries.append(robot_pt)

    if ray_origin is not None:
        origin_pt = o3d.geometry.PointCloud()
        origin_pt.points = o3d.utility.Vector3dVector([np.asarray(ray_origin)])
        origin_pt.colors = o3d.utility.Vector3dVector([[0.0, 1.0, 0.0]])
        geometries.append(origin_pt)

    o3d.visualization.draw_geometries(geometries, window_name=window_name)


def generate_head_camera_pcd(env, cam_id=5, depth_min=0.3, depth_max=2.0,
                              mask_floor=True, floor_z_thresh=0.05):
    """
    Generate an obstacle point cloud in world coordinates from the RGB-D data of
    the head camera (or a specified camera).
    Follows the wrist camera point cloud generation logic, using env.get_rgbd /
    get_camera_parse to obtain the correct extrinsic.

    Parameters
    ----------
    env : RMMBench environment object
    cam_id : int
        Camera id, default 5 (head camera).
    depth_min, depth_max : float
        Depth filtering range (meters).
    mask_floor : bool
        Whether to mask out the floor (z < floor_z_thresh).
    floor_z_thresh : float
        Floor height threshold.

    Returns
    -------
    pcd : open3d.geometry.PointCloud
        Obstacle point cloud in world coordinates.
    """
    import open3d as o3d

    rgb, depth, seg, intrinsic, extrinsic = env.get_camera_parse_physics(cam_id)
    h, w = depth.shape

    # Build the mask: only filter by depth range, keep all points (including the robot itself)
    valid_mask = (depth > depth_min) & (depth < depth_max)

    # Generate the camera-frame point cloud with Open3D
    od_cammat = o3d.camera.PinholeCameraIntrinsic(
        w, h, intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
    )
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(np.ascontiguousarray(rgb)),
        o3d.geometry.Image(np.ascontiguousarray(depth)),
        convert_rgb_to_intensity=False,
        depth_scale=1.0,
        depth_trunc=10.0,
    )
    cam_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, od_cammat)

    # Transform to world coordinates: extrinsic is camera->world, so apply it directly
    world_pcd = cam_pcd.transform(extrinsic)

    # Filter out the floor
    if mask_floor:
        pts = np.asarray(world_pcd.points)
        colors = np.asarray(world_pcd.colors)
        keep = pts[:, 2] > floor_z_thresh
        world_pcd = o3d.geometry.PointCloud()
        world_pcd.points = o3d.utility.Vector3dVector(pts[keep])
        if len(colors) > 0:
            world_pcd.colors = o3d.utility.Vector3dVector(colors[keep])

    return world_pcd


def get_head_camera_navigation_view(env, cam_id=5, robot_radius=ROBOT_RADIUS,
                                     use_head_pcd=True):
    """
    Convenience function: fetch the specified camera image from the environment
    and draw the navigation ray overlay on top of it.

    Parameters
    ----------
    env : RMMBench environment object
        Must support env.get_camera_parse(cam_id) returning (rgb, depth, seg, intrinsic, extrinsic).
    cam_id : int
        Camera id, default 5 (head camera).
    robot_radius : float
        Robot radius.
    use_head_pcd : bool
        Whether to use a point cloud generated separately by the specified camera (True recommended).
        If False, fall back to env.get_obstacle_pcd().

    Returns
    -------
    overlay_img : np.ndarray
        RGB image with rays drawn.
    ray_info : list of dict
        List of ray info dicts.
    """
    rgb, depth, seg, intrinsic, extrinsic = env.get_camera_parse_physics(cam_id)
    robot_info = env.robot.get_link_base_info(env.physics)
    robot_pos = robot_info["position"]
    robot_yaw = robot_info["euler"][2]

    # Get the obstacle point cloud
    obstacle_pcd = None
    if use_head_pcd:
        try:
            obstacle_pcd = generate_head_camera_pcd(env, cam_id=cam_id)
        except Exception as e:
            print(f"[get_head_camera_navigation_view] head pcd failed: {e}")
            obstacle_pcd = None

    if obstacle_pcd is None and hasattr(env, 'get_obstacle_pcd'):
        try:
            obstacle_pcd = env.get_obstacle_pcd()
        except Exception:
            pass

    # dm_control render returns RGB, but OpenCV drawing uses BGR; convert first
    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    overlay_img, ray_info = draw_navigation_overlay(
        rgb=rgb_bgr,
        robot_pos=robot_pos,
        robot_yaw=robot_yaw,
        intrinsic=intrinsic,
        extrinsic=extrinsic,
        obstacle_pcd=obstacle_pcd,
        robot_radius=robot_radius
    )
    return overlay_img, ray_info


# =============================================================================
# Navigation grid drawing (scene level)
# =============================================================================

def _world_to_local(world_pt, robot_pos, robot_yaw):
    """Convert a world-coordinate point to the robot's local frame (body frame)."""
    dx = world_pt[0] - robot_pos[0]
    dy = world_pt[1] - robot_pos[1]
    cos_yaw = math.cos(-robot_yaw)
    sin_yaw = math.sin(-robot_yaw)
    local_x = dx * cos_yaw - dy * sin_yaw
    local_y = dx * sin_yaw + dy * cos_yaw
    return np.array([local_x, local_y])


def _local_to_world(local_pt, robot_pos, robot_yaw):
    """Convert a point in the robot's local frame to world coordinates."""
    cos_yaw = math.cos(robot_yaw)
    sin_yaw = math.sin(robot_yaw)
    world_x = robot_pos[0] + local_pt[0] * cos_yaw - local_pt[1] * sin_yaw
    world_y = robot_pos[1] + local_pt[0] * sin_yaw + local_pt[1] * cos_yaw
    return np.array([world_x, world_y])


def draw_navigation_grid(rgb, env, cam_id=5,
                         obstacle_pcd=None,
                         z_th=GRID_Z_TH_DEFAULT,
                         grid_cell_size=GRID_CELL_SIZE_DEFAULT,# side length of a single grid cell, in meters
                         grid_front=GRID_FRONT_DEFAULT,# grid_front (default GRID_FRONT_DEFAULT = 1.0) is the maximum distance to draw in front of the robot
                         grid_back=GRID_BACK_DEFAULT,# grid_back (default GRID_BACK_DEFAULT = 0.5) is the maximum distance to draw behind the robot; likewise determines the initial value of x_min = -bounds['back']
                         grid_side=GRID_SIDE_DEFAULT,
                         grid_color=GRID_LINE_COLOR,
                         fill_color=GRID_FILL_COLOR,
                         line_thickness=GRID_LINE_THICKNESS,
                         alpha=GRID_ALPHA,
                         text_scale=GRID_TEXT_SCALE,
                         text_thickness=GRID_TEXT_THICKNESS,
                         text_color=GRID_TEXT_COLOR,
                         obstacle_radius=GRID_OBSTACLE_RADIUS):
    """
    Draw a navigation grid around the robot on the head camera RGB image.

    The grid is drawn within a rectangle in the robot's local frame:
    - x ∈ [-grid_back, grid_front] (front/back)
    - y ∈ [-grid_side, grid_side] (left/right)

    For each grid cell, use its center as the circle center and obstacle_radius as the
    radius to check whether nearby ground obstacle points exist (z < z_th):
    if they do, the cell is considered occupied by an obstacle and is neither drawn nor numbered;
    if the cell's projection falls outside the image or behind the camera, it is likewise
    not drawn and not numbered.
    Only finally-visible cells (not occupied and successfully projected into the frame)
    receive row-column labels, numbered consecutively "top to bottom, left to right"
    as 1-1, 1-2, ..., 2-1, ... (top-left is 1-1).
    Numbering has no gaps caused by intermediate cells being filtered out
    (e.g., no case of 1-1, 1-3 without 1-2).

    Parameters
    ----------
    rgb : np.ndarray, shape (H, W, 3)
        Head camera RGB image (uint8, BGR format).
    env : RMMBench environment object
        Must support env.get_camera_parse(cam_id) and env.robot.get_link_base_info(env.physics).
    cam_id : int
        Camera id, default 5 (head camera).
    obstacle_pcd : open3d.geometry.PointCloud or np.ndarray or None
        Obstacle point cloud. If None, generated automatically from the head camera.
    z_th : float
        Obstacle height threshold. Only points with z < z_th are considered ground obstacles.
    grid_cell_size : float
        Grid cell size (meters).
    grid_front : float
        Drawing range in front of the robot (meters).
    grid_back : float
        Drawing range behind the robot (meters).
    grid_side : float
        Drawing range to the robot's left/right per side (meters); total width is 2*grid_side.
    grid_color : tuple
        Grid line color (BGR).
    fill_color : tuple
        Grid fill color (BGR).
    line_thickness : int
        Grid line width.
    alpha : float
        Grid fill transparency (0-1); 0 means no fill.
    text_scale : float
        Label font size.
    text_thickness : int
        Label font thickness.
    text_color : tuple
        Label text color (BGR).
    obstacle_radius : float
        Obstacle detection radius (meters), default 0.15.

    Returns
    -------
    overlay_img : np.ndarray, shape (H, W, 3)
        Image with the grid drawn (uint8, BGR).
    grid_info : list of dict
        Info for each grid cell, containing:
        - idx: (i, j) original grid indices, i along the front/back direction and j along the left/right direction
        - label: label string (e.g. "1-1"); only cells with visible=True have this field
        - center_world: center point in world coordinates [x, y, z]
        - corners_uv: pixel coordinates of the four corners [(u1,v1), ...]
        - projectable: whether the cell projects successfully into the image (ignoring obstacles)
        - occupied: whether the cell is occupied by an obstacle
        - visible: whether the cell is finally visible and numbered (projectable and not occupied)
    table_bounds : dict
        Table bounds and intermediate values needed for projection, for direct reuse by
        functions such as draw_stop_points_on_grid without recomputation, containing:
        - x_min, x_max, y_min, y_max: reachable rectangle bounds in the robot's local frame
        - robot_pos, robot_yaw: robot position and heading
        - intrinsic, extrinsic_inv: camera intrinsics and inverse extrinsics (for projecting world coordinates to pixels)
        - img_w, img_h: image width and height
    """
    # Get camera parameters and robot pose from the environment
    _, _, _, intrinsic, extrinsic = env.get_camera_parse_physics(cam_id)
    robot_info = env.robot.get_link_base_info(env.physics)
    robot_pos = robot_info["position"]
    robot_yaw = robot_info["euler"][2]

    # Generate the obstacle point cloud automatically if not provided
    if obstacle_pcd is None:
        try:
            obstacle_pcd = generate_head_camera_pcd(env, cam_id=cam_id)
        except Exception as e:
            print(f"[draw_navigation_grid] auto-generate pcd failed: {e}")
            obstacle_pcd = None

    img = rgb.copy()
    img_h, img_w = img.shape[:2]
    extrinsic_inv = np.linalg.inv(extrinsic)

    # ---------- 1. Prepare the obstacle point cloud ----------
    obs_pts = None
    if obstacle_pcd is not None:
        if hasattr(obstacle_pcd, 'points'):
            obs_pts = np.asarray(obstacle_pcd.points)
        else:
            obs_pts = np.asarray(obstacle_pcd)
        if obs_pts.shape[0] > 0:
            # Keep only ground-level obstacles with z < z_th
            obs_pts = obs_pts[obs_pts[:, 2] < z_th]

    # ---------- 2. Compute ray bounds in the four directions (cf. compute_ray_lengths) ----------
    # Cast rays forward, backward, left, and right from the robot center, stopping at obstacles,
    # forming the actual reachable rectangle
    ray_origin = np.array([robot_pos[0], robot_pos[1], 0])

    # Four directions (world coordinates): front(+x_body), back(-x_body), left(+y_body), right(-y_body)
    directions = [
        ('front', robot_yaw, grid_front),
        ('back', robot_yaw + math.pi, grid_back),
        ('left', robot_yaw + math.pi / 2, grid_side),
        ('right', robot_yaw - math.pi / 2, grid_side),
    ]

    bounds = {}
    for name, angle, max_dist in directions:
        direction = np.array([math.cos(angle), math.sin(angle), 0.0])
        actual_dist = max_dist

        if obs_pts is not None and obs_pts.shape[0] > 0:
            vec_to_obs = obs_pts - ray_origin
            proj_len = np.dot(vec_to_obs, direction)

            # Collision detection (consistent with compute_ray_lengths)
            valid = (proj_len > 0) & (np.abs(obs_pts[:, 2] - z_th) < 0.2)
            if np.sum(valid) > 0:
                proj_pts = ray_origin + np.outer(proj_len, direction)
                perp_dists_xy = np.linalg.norm(obs_pts[:, :2] - proj_pts[:, :2], axis=1)
                collision_mask = valid & (perp_dists_xy < obstacle_radius)
                if np.sum(collision_mask) > 0:
                    collision_dist = np.min(proj_len[collision_mask])
                    actual_dist = min(actual_dist, max(0, collision_dist - obstacle_radius))

        bounds[name] = actual_dist

    # Reachable region (robot local coordinates)
    x_min = -bounds['back']
    x_max = bounds['front']
    y_min = -bounds['right']
    y_max = bounds['left']

    # ---------- 3. Generate grid cells ----------
    # Count the full cells (excluding edge portions smaller than one cell_size)
    n_cells_x = int((x_max - x_min) // grid_cell_size)
    n_cells_y = int((y_max - y_min) // grid_cell_size)

    # Compute the total extent of the full cells
    total_x = n_cells_x * grid_cell_size
    total_y = n_cells_y * grid_cell_size

    # Centering offset: center the full-cell region within the reachable region
    offset_x = (x_max - x_min - total_x) / 2
    offset_y = (y_max - y_min - total_y) / 2

    cells = []
    world_corners_all = []

    # Convention: i iterates along the robot's local x axis (front/back) over [0, n_cells_x-1],
    #       i=0 corresponds to the frontmost cell (largest local x, top of the image, smallest row number);
    #      j iterates along the robot's local y axis (left/right) over [0, n_cells_y-1],
    #       j=0 corresponds to the leftmost cell (largest local y, left of the image, smallest column number).
    # Thus row = i+1 and column = j+1, naturally satisfying "top-left is 1-1, column increases rightward, row increases downward".
    for i in range(n_cells_x):
        for j in range(n_cells_y):
            cx1 = x_max - offset_x - i * grid_cell_size
            cx0 = cx1 - grid_cell_size
            cy1 = y_max - offset_y - j * grid_cell_size
            cy0 = cy1 - grid_cell_size

            corners_local = [
                np.array([cx0, cy0]),
                np.array([cx1, cy0]),
                np.array([cx1, cy1]),
                np.array([cx0, cy1]),
            ]

            corners_world = [
                np.array([*_local_to_world(cl, robot_pos, robot_yaw), 0.0])
                for cl in corners_local
            ]

            center_local = np.array([(cx0 + cx1) / 2, (cy0 + cy1) / 2])
            center_world = np.array([*_local_to_world(center_local, robot_pos, robot_yaw), 0.0])

            cells.append({
                'idx': (i, j),
                'corners_world': corners_world,
                'center_world': center_world,
                'center_local': center_local,
            })
            world_corners_all.extend(corners_world)

    table_bounds = {
        'x_min': x_min,
        'x_max': x_max,
        'y_min': y_min,
        'y_max': y_max,
        'robot_pos': robot_pos,
        'robot_yaw': robot_yaw,
        'intrinsic': intrinsic,
        'extrinsic_inv': extrinsic_inv,
        'img_w': img_w,
        'img_h': img_h,
    }

    if len(cells) == 0:
        return img, [], table_bounds

    # ---------- 4. Batch-project onto the image ----------
    world_corners_all = np.array(world_corners_all)
    uv_all, valid_all = project_world_to_image(world_corners_all, intrinsic, extrinsic_inv, img_w, img_h)

    # ---------- 5. Per-cell processing: projection visibility + obstacle occupancy check ----------
    grid_info = []
    corner_offset = 0

    for cell in cells:
        n_corners = 4
        uv_corners = uv_all[corner_offset:corner_offset + n_corners]
        valid_corners = valid_all[corner_offset:corner_offset + n_corners]
        corner_offset += n_corners

        in_bounds = all(
            0 <= uv[0] < img_w and 0 <= uv[1] < img_h
            for uv in uv_corners
        )
        projectable = bool(np.all(valid_corners) and in_bounds)

        # Obstacle occupancy check: with the cell center as the circle center and obstacle_radius
        # as the radius, if ground obstacle points (z < z_th) exist within that range,
        # the cell is considered occupied and is neither drawn nor numbered.
        occupied = False
        if obs_pts is not None and obs_pts.shape[0] > 0:
            dists_xy = np.linalg.norm(obs_pts[:, :2] - cell['center_world'][:2], axis=1)
            occupied = bool(np.any(dists_xy < obstacle_radius))

        visible = projectable and not occupied

        info = {
            'idx': cell['idx'],
            'center_world': cell['center_world'],
            'corners_uv': [(int(uv[0]), int(uv[1])) for uv in uv_corners],
            'projectable': projectable,
            'occupied': occupied,
            'visible': visible,
        }
        grid_info.append(info)

    # ---------- 6. Assign labels (row-column format) ----------
    # Numbering rules:
    # - i iterates along the front/back direction with i=0 at the front; j iterates along the
    #   left/right direction with j=0 at the leftmost
    #   (both are guaranteed during grid generation: the top of the image corresponds to i=0,
    #   the left of the image corresponds to j=0)
    # - Only finally-visible cells (not occupied by obstacles and successfully projected into
    #   the frame) are numbered, renumbered consecutively as 1..N in "top to bottom, left to
    #   right" order; no gaps are kept for filtered-out cells.
    visible_cells = [info for info in grid_info if info['visible']]
    visible_cells.sort(key=lambda info: (info['idx'][0], info['idx'][1]))

    # First determine how many distinct "rows" there are (grouped by i); number columns consecutively by j within each row
    row_ids = sorted(set(info['idx'][0] for info in visible_cells))
    row_number = {i_val: r + 1 for r, i_val in enumerate(row_ids)}

    col_counter = {}
    for info in visible_cells:
        i_val = info['idx'][0]
        row = row_number[i_val]
        col = col_counter.get(i_val, 0) + 1
        col_counter[i_val] = col
        info['label'] = f"{row}-{col}"

    # ---------- 7. Draw the grid ----------
    if alpha > 0:
        overlay = img.copy()

    for info in grid_info:
        if not info['visible']:
            continue

        pts = np.array(info['corners_uv'], dtype=np.int32).reshape((-1, 1, 2))

        if alpha > 0:
            cv2.fillPoly(overlay, [pts], fill_color)

        cv2.polylines(img, [pts], isClosed=True, color=grid_color, thickness=line_thickness)

    if alpha > 0:
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # ---------- 8. Draw the labels ----------
    for info in grid_info:
        if not info['visible']:
            continue

        corners = info['corners_uv']
        center_u = int(sum(c[0] for c in corners) / 4)
        center_v = int(sum(c[1] for c in corners) / 4)

        uv_np = np.array(corners)
        cell_w = int(np.max(uv_np[:, 0]) - np.min(uv_np[:, 0]))
        cell_h = int(np.max(uv_np[:, 1]) - np.min(uv_np[:, 1]))
        min_cell_size = min(cell_w, cell_h)

        if min_cell_size >= GRID_MIN_PIXEL_SIZE:
            label = info['label']
            (text_w, text_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_thickness
            )
            text_x = center_u - text_w // 2
            text_y = center_v + text_h // 2
            cv2.putText(
                img, label, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_color, text_thickness,
                lineType=cv2.LINE_AA
            )

    return img, grid_info, table_bounds


def draw_stop_points_on_grid(rgb, env, stop_points, cam_id=5,
                              z_th=GRID_Z_TH_DEFAULT,
                              grid_cell_size=GRID_CELL_SIZE_DEFAULT,
                              grid_front=GRID_FRONT_DEFAULT,
                              grid_back=GRID_BACK_DEFAULT,
                              grid_side=GRID_SIDE_DEFAULT,
                              grid_color=GRID_LINE_COLOR,
                              fill_color=GRID_FILL_COLOR,
                              line_thickness=GRID_LINE_THICKNESS,
                              alpha=GRID_ALPHA,
                              text_scale=GRID_TEXT_SCALE,
                              text_thickness=GRID_TEXT_THICKNESS,
                              text_color=GRID_TEXT_COLOR,
                              obstacle_radius=GRID_OBSTACLE_RADIUS,
                              stop_point_front=0.2,
                              stop_point_side=0.2,
                              stop_point_color=(0, 0, 255),
                              stop_point_line_thickness=2):
    """
    First call draw_navigation_grid to draw the table grid, then draw the stop point
    cells on top of it (red borders, no fill), and compute how each stop point cell
    overlaps with the table range (reachable rectangle).

    The table range and obstacle point cloud are computed once inside draw_navigation_grid;
    this function directly reuses its returned table_bounds (containing robot_pos,
    robot_yaw, x_min/x_max/y_min/y_max, camera intrinsics/extrinsics, etc.) instead of
    recomputing them, and no separate obstacle_pcd needs to be passed in.

    How the stop point cells are constructed:
    - Only the xy coordinates of the stop point's position (world frame) are used;
      fields such as target_bbox and target_orientation are ignored.
    - The position is converted to the robot's local frame; using it as the center,
      the cell is inflated by stop_point_front along the local x axis (front/back)
      and by stop_point_side along the local y axis (left/right), forming an
      axis-aligned rectangle in the local frame oriented like the grid.
    - When drawing, the four corners of this local rectangle are converted back to
      world coordinates and projected onto the image, outlined in red, with no
      fill or text annotation inside.

    Parameters
    ----------
    rgb : np.ndarray, shape (H, W, 3)
        Head camera RGB image (uint8, BGR format).
    env : RMMBench environment object
        Must support env.get_camera_parse(cam_id) and env.robot.get_link_base_info(env.physics).
    stop_points : list of dict
        Each element must contain at least 'position': [x, y, z] (world frame).
        Other fields (e.g. target_bbox, target_orientation) are ignored.
    cam_id, z_th, grid_cell_size, grid_front, grid_back, grid_side,
    grid_color, fill_color, line_thickness, alpha, text_scale,
    text_thickness, text_color, obstacle_radius :
        Exactly the same meaning as in draw_navigation_grid, passed through unchanged
        to draw the table grid.
    stop_point_front : float
        One-sided inflation radius of the stop point cell along the robot's local
        front/back direction (meters).
    stop_point_side : float
        One-sided inflation radius of the stop point cell along the robot's local
        left/right direction (meters).
    stop_point_color : tuple
        Stop point cell border color (BGR), default red (0, 0, 255).
    stop_point_line_thickness : int
        Stop point cell border line width.

    Returns
    -------
    overlay_img : np.ndarray, shape (H, W, 3)
        Image with the table grid + stop point cells drawn (uint8, BGR).
    grid_info : list of dict
        Same as the return value of draw_navigation_grid; see its docstring.
    stop_point_results : list of dict
        One-to-one with stop_points; each element contains:
        - position: the original input position
        - local_xy: the stop point's (x, y) in the robot's local frame
        - stop_bbox_local: the stop point cell's (x_min, x_max, y_min, y_max) in the local frame
        - in_table_range: whether the stop point cell intersects the table range
        - overlap_ratio: intersection area / stop point cell area (0.0 if no intersection)
        - overlap_area: intersection area (square meters)
        - stop_bbox_area: stop point cell area (square meters)
        - drawn: whether it was actually drawn onto the image. Requires all of: the four
          corner projections are valid and within the image, and in_table_range is True
          (i.e. it intersects the table range);
          stop points entirely outside the table range are not drawn, even if their
          projection itself is visible.
    """
    img, grid_info, table_bounds = draw_navigation_grid(
        rgb, env, cam_id=cam_id,
        z_th=z_th,
        grid_cell_size=grid_cell_size,
        grid_front=grid_front,
        grid_back=grid_back,
        grid_side=grid_side,
        grid_color=grid_color,
        fill_color=fill_color,
        line_thickness=line_thickness,
        alpha=alpha,
        text_scale=text_scale,
        text_thickness=text_thickness,
        text_color=text_color,
        obstacle_radius=obstacle_radius,
    )

    robot_pos = table_bounds['robot_pos']
    robot_yaw = table_bounds['robot_yaw']
    table_x_min = table_bounds['x_min']
    table_x_max = table_bounds['x_max']
    table_y_min = table_bounds['y_min']
    table_y_max = table_bounds['y_max']
    intrinsic = table_bounds['intrinsic']
    extrinsic_inv = table_bounds['extrinsic_inv']
    img_w = table_bounds['img_w']
    img_h = table_bounds['img_h']

    stop_point_results = []
    for sp in stop_points:
        position = sp['position']
        local_xy = _world_to_local(position, robot_pos, robot_yaw)
        lx, ly = float(local_xy[0]), float(local_xy[1])

        stop_x_min = lx - stop_point_front
        stop_x_max = lx + stop_point_front
        stop_y_min = ly - stop_point_side
        stop_y_max = ly + stop_point_side

        inter_x_min = max(stop_x_min, table_x_min)
        inter_x_max = min(stop_x_max, table_x_max)
        inter_y_min = max(stop_y_min, table_y_min)
        inter_y_max = min(stop_y_max, table_y_max)

        inter_w = max(0.0, inter_x_max - inter_x_min)
        inter_h = max(0.0, inter_y_max - inter_y_min)
        overlap_area = inter_w * inter_h

        stop_bbox_area = (stop_x_max - stop_x_min) * (stop_y_max - stop_y_min)
        overlap_ratio = overlap_area / stop_bbox_area if stop_bbox_area > 0 else 0.0
        in_table_range = overlap_area > 0

        # Four corners in local coordinates -> world coordinates -> project onto the image
        corners_local = [
            np.array([stop_x_min, stop_y_min]),
            np.array([stop_x_max, stop_y_min]),
            np.array([stop_x_max, stop_y_max]),
            np.array([stop_x_min, stop_y_max]),
        ]
        corners_world = np.array([
            [*_local_to_world(cl, robot_pos, robot_yaw), 0.0]
            for cl in corners_local
        ])
        uv_corners, valid_corners = project_world_to_image(
            corners_world, intrinsic, extrinsic_inv, img_w, img_h
        )

        in_bounds = all(
            0 <= uv[0] < img_w and 0 <= uv[1] < img_h
            for uv in uv_corners
        )
        # A stop point cell is drawn only if it intersects the table range (reachable rectangle);
        # stop points entirely outside the table range are not drawn, even if their projection is visible.
        drawn = bool(np.all(valid_corners) and in_bounds and in_table_range)

        if drawn:
            pts = np.array(
                [(int(uv[0]), int(uv[1])) for uv in uv_corners],
                dtype=np.int32
            ).reshape((-1, 1, 2))
            cv2.polylines(
                img, [pts], isClosed=True,
                color=stop_point_color, thickness=stop_point_line_thickness
            )

        stop_point_results.append({
            'position': position,
            'local_xy': (lx, ly),
            'stop_bbox_local': (stop_x_min, stop_x_max, stop_y_min, stop_y_max),
            'in_table_range': in_table_range,
            'overlap_ratio': overlap_ratio,
            'overlap_area': overlap_area,
            'stop_bbox_area': stop_bbox_area,
            'drawn': drawn,
        })

    return img, grid_info, stop_point_results
