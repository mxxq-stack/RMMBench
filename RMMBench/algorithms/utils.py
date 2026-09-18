import numpy as np
from scipy.spatial.transform import Rotation as R

def qauternion_slerp(start_quat, end_quat, t):
    """ 
    Performs Spherical Linear Interpolation (SLERP) between two quaternions.

    Given two quaternions, `start_quat` and `end_quat`, this function interpolates between them
    based on the parameter `t`, which typically ranges from 0 to 1. The function performs the 
    interpolation along the shortest path on the 4D hypersphere.

    Parameters:
        -start_quat: The starting quaternion, a 4-dimensional vector representing the initial orientation.
        -end_quat: The ending quaternion, a 4-dimensional vector representing the final orientation.
        -t : The interpolation parameter, typically between 0 and 1. t = 0 returns `start_quat`, and t = 1 returns `end_quat`.
        
    Returns:
        -interpolated quaternion: a smooth transition between `start_quat` and `end_quat`. The result is normalized.
    """
    start_quat = start_quat / np.linalg.norm(start_quat)
    end_quat = end_quat / np.linalg.norm(end_quat)
    
    dot_product = np.dot(start_quat, end_quat)
    
    if dot_product < 0:
        end_quat = -end_quat
        dot_product = -dot_product
    
    if dot_product > 0.9995:
        result = start_quat + t * (end_quat - start_quat)
        return result / np.linalg.norm(result)
    
    theta_0 = np.arccos(dot_product)
    theta = theta_0 * t
    
    theta_quat = end_quat - start_quat * dot_product
    theta_quat = theta_quat / np.linalg.norm(theta_quat)
    
    return start_quat * np.cos(theta) + theta_quat * np.sin(theta)

def interpolate_path(positions, quaternions, target_velocity=0.05):
    """
        Perform smooth interpolation on an RRT-generated path, including both positions and quaternions.

        Parameters:
        - positions: A list of positions in the path.
        - quaternions: A list of quaternions corresponding to each position in the path.
        - target_velocity: The target velocity for the move, used to compute interpolation number.

        Returns:
        - interpolated_positions: A list of interpolated positions.
        - interpolated_quaternions: A list of interpolated quaternions.
    """
    assert len(positions) == len(quaternions), "positions and quaternions must have the same length"
    interpolated_positions = []
    interpolated_quaternions = []
    
    for i in range(len(positions) - 1):
        start_pos = positions[i]
        end_pos = positions[i + 1]
        start_quat = quaternions[i]
        end_quat = quaternions[i + 1]
        
        distance = np.linalg.norm(end_pos - start_pos)
        num_interpolations = int(distance / (target_velocity * 0.2)) # no interpolation; going from start to goal takes 10 integration steps, interpolation means 0.05 m per step (0.2 s); default simulation timestep is 0.02 with 10 substeps
        for t in np.linspace(0, 1, num_interpolations, endpoint=False):
            interp_pos = (1 - t) * start_pos + t * end_pos
            interp_quat = qauternion_slerp(start_quat, end_quat, t)
            
            interpolated_positions.append(interp_pos)
            interpolated_quaternions.append(interp_quat)
    
    interpolated_positions.append(positions[-1])
    interpolated_quaternions.append(quaternions[-1])
    
    return interpolated_positions, interpolated_quaternions


def interpolate_pos(positions, target_velocity=0.05):
    """
        Perform smooth interpolation on an RRT-generated path, including both positions and quaternions.

        Parameters:
        - positions: A list of positions in the path.
        - quaternions: A list of quaternions corresponding to each position in the path.
        - target_velocity: The target velocity for the move, used to compute interpolation number.

        Returns:
        - interpolated_positions: A list of interpolated positions.
        - interpolated_quaternions: A list of interpolated quaternions.
    """
    interpolated_positions = []

    # for i in range(len(positions) - 1):
    start_pos = np.array(positions[0])
    end_pos = np.array(positions[-1])

    # print("start_pos:", start_pos)
    # print("end_pos:", end_pos)

    distance = np.linalg.norm(end_pos - start_pos)
    num_interpolations = int(distance / (
                target_velocity * 0.2))  # no interpolation; going from start to goal takes 10 integration steps, interpolation means 0.05 m per step (0.2 s); default simulation timestep is 0.02 with 10 substeps
    for t in np.linspace(0, 1, num_interpolations, endpoint=False):
        interp_pos = (1 - t) * start_pos + t * end_pos

        interpolated_positions.append(list(interp_pos))

    interpolated_positions.append(positions[-1])

    return interpolated_positions


def interpolate_radians(radians, target_velocity=0.157):
    interpolated_radians = []
    dt = 0.2  # time per step

    for i in range(len(radians) - 1):
        start = np.array(radians[i])
        end = np.array(radians[i + 1])

        # 1. Compute the Euclidean distance (magnitude of the radian difference)
        dist = np.linalg.norm(end - start)

        # 2. Compute the number of steps: distance / (velocity * time)
        num_interpolations = int(dist / (target_velocity * dt))

        if num_interpolations <= 1:
            interpolated_radians.append(start)
            continue

        # 3. Directly use simple linear interpolation (LERP)
        for t in np.linspace(0, 1, num_interpolations, endpoint=False):
            interp_radian = start + t * (end - start)
            interpolated_radians.append(interp_radian)

    interpolated_radians.append(radians[-1])
    return interpolated_radians

def remove_pcd_near_point(pcd, center, bbox_size=[0.05, 0.05, 0.05]):
    """
    Remove points near a specified bbox from the point cloud.
    
    Args:
    - point_cloud: numpy array or Open3D PointCloud object, input point cloud.
    - center: tuple or list, center coordinates of the bbox (x, y, z).
    - bbox_size: tuple or list, size of the bbox in each dimension (size_x, size_y, size_z).
    
    Returns:
    - filtered_point_cloud: numpy array, point cloud with points near the bbox removed.
    """
    min_bound = np.array(center) - np.array(bbox_size) / 2.0
    max_bound = np.array(center) + np.array(bbox_size) / 2.0
    
    mask = np.logical_or.reduce((pcd[:, 0] < min_bound[0], 
                                pcd[:, 0] > max_bound[0],
                                pcd[:, 1] < min_bound[1],
                                pcd[:, 1] > max_bound[1],
                                pcd[:, 2] < min_bound[2],
                                pcd[:, 2] > max_bound[2]))
    filtered_pcd = pcd[mask]
    return filtered_pcd
    