import numpy as np
import open3d as o3d
from rrt_algorithms.rrt.rrt import RRT
from rrt_algorithms.rrt.rrt_star import RRTStar
from rrt_algorithms.search_space.search_space import SearchSpace
from rrt_algorithms.utilities.plotting import Plot
from RMMBench.algorithms.path_smoothing import bezier_smoothing, polynomial_smoothing
from RMMBench.algorithms.utils import remove_pcd_near_point
from rrt_algorithms.utilities.geometry import steer


SEARCH_DIMENSIONS = np.array([(-1, 1), (-1, 1), (0.8, 1.3)])


def rrt_motion_planning(start_pos,
                        end_pos,
                        obstacle_pcd,
                        object_pos=None,
                        search_dimensions=SEARCH_DIMENSIONS,
                        q=0.05,
                        r=0.02,
                        max_samples=1024,
                        prc=0.1,
                        margin=2e-2,
                        z_threshold=0.8,
                        smooth_method=None,
                        retry_time=0,
                        # [New] link7 obstacle-avoidance parameters
                        link7_offset=None,
                        link7_radius=0.02,
                        robot_pos=None,
                        ):
    """
    3d rrt motion planning for robot arm.
    Input:
        start_pos: tuple, (3,)
        end_pos: ttple, (3,)
        obstacle_pcd: np.ndarray, (n, 3)
        search_dimensions: np.ndarray, (3, 2), range of x-y-z space
        q: length of tree edges
        r: length of smallest edge to check for intersection with obstacles
        max_samples:  max number of samples to take before timing out
        prc: probability of checking for a connection to goal
        smooth_method: str, method to smooth the path, None or one of ["bezier", "polynomial"]
    """
    if isinstance(obstacle_pcd, o3d.geometry.PointCloud):
        obstacle_pcd = np.asarray(obstacle_pcd.points)

    obstacle_pcd = obstacle_pcd[obstacle_pcd[:, 2] >= z_threshold]
    # Remove obstacle point cloud near the start point to avoid being unable to start
    obstacle_pcd = remove_pcd_near_point(obstacle_pcd, start_pos)
    # Obstacle inflation
    if obstacle_pcd is not None and obstacle_pcd.shape[1] != 6:
        obstacle_pcd = np.concatenate([obstacle_pcd - margin * np.ones((obstacle_pcd.shape[0], 3)),
                                       obstacle_pcd + margin * np.ones((obstacle_pcd.shape[0], 3))],
                                      axis=1)
    if robot_pos is not None:
        robot_x, robot_y = robot_pos[0], robot_pos[1]

        x_min = robot_x - 1.0
        x_max = robot_x + 1.0
        y_min = robot_y - 1.0
        y_max = robot_y + 1.0

        search_dimensions = np.array([(x_min, x_max), (y_min, y_max), (0.8, 1.3)])
        print(f"[RRT] Dynamic search_dimensions: {search_dimensions}")



    # Initialize the search space
    # Inputs: search bounds + inflated obstacles

    search_space = SearchSpace(search_dimensions, obstacle_pcd)

    # Initialize the RRT object
    # Inputs: search space, step length, start point, end point, max samples,
    # collision-check resolution, goal-bias probability
    # [Modified] GraspingRRT is also used when link7_offset is not None
    if object_pos is not None:
        object_offset = start_pos - np.array(object_pos)
        rrt = GraspingRRT(search_space, q, start_pos, end_pos, max_samples, r, prc, object_offset+0.05,
                          link7_offset=link7_offset, link7_radius=link7_radius)
    elif link7_offset is not None:
        # [New] scenarios without object_pos (e.g. place) that still need link7 obstacle avoidance
        rrt = GraspingRRT(search_space, q, start_pos, end_pos, max_samples, r, prc,
                          link7_offset=link7_offset, link7_radius=link7_radius)
    else:
        rrt = RRT(search_space, q, start_pos, end_pos, max_samples, r, prc)
    # Run the RRT search
    path = rrt.rrt_search()
    retry = 0
    # Retry; no retries by default
    while retry < retry_time and path is None:
        print("retry:", retry)
        retry += 1
        start_pos_list = list(start_pos)
        start_pos_list[-1] += 0.1
        start_pos = tuple(start_pos_list)
        # [Modified] keep link7 obstacle avoidance during retry too
        if link7_offset is not None:
            rrt = GraspingRRT(search_space, q, start_pos, end_pos, max_samples, r, prc,
                              link7_offset=link7_offset, link7_radius=link7_radius)
        else:
            rrt = RRT(search_space, q, start_pos, end_pos, max_samples, r, prc)
        path = rrt.rrt_search()
    if smooth_method == "bezier":
        path_arr = bezier_smoothing(path)
        path = [tuple(p) for p in path_arr]
    elif smooth_method == "polynomial":
        path_arr = polynomial_smoothing(path)
        path = [tuple(p) for p in path_arr]
    return path


def rrt_star(start_pos,
             end_pos,
             obstacle_pcd,
             search_dimensions=SEARCH_DIMENSIONS,
             q=8,
             r=1,
             max_samples=1024,
             rewire_count=32,
             prc=0.1):
    """
    3d rrt* motion planning for robot arm.
    Input:
        start_pos: np.ndarray, (3,)
        end_pos: np.ndarray, (3,)
        obstacle_pcd: np.ndarray, (n, 3)
        search_dimensions: np.ndarray, (3, 2), range of x-y-z space
        q: length of tree edges
        r: length of smallest edge to check for intersection with obstacles
        max_samples:  max number of samples to take before timing out
        rewire_count: optional, number of nearby branches to rewire
        prc: probability of checking for a connection to goal
    """
    if isinstance(obstacle_pcd, o3d.geometry.PointCloud):
        obstacle_pcd = np.asarray(obstacle_pcd.points)
    if obstacle_pcd is not None and obstacle_pcd.shape[1] != 6:
        obstacle_pcd = np.concatenate([obstacle_pcd, obstacle_pcd +
                                       0.01 * np.ones((obstacle_pcd.shape[0], 3))], axis=1)
    search_space = SearchSpace(search_dimensions, obstacle_pcd)
    rrt = RRTStar(search_space, q, start_pos, end_pos, max_samples, r, prc, rewire_count)
    path = rrt.rrt_star()

    return path


def visulize_search_path(search_space, rrt, path, obstacles, start_pos, end_pos):
    """
    Visulize the search path of rrt algorithm.
    Input:
        search_space: SearchSpace, search space
        rrt: RRT, rrt object
    """
    plot = Plot("rrt_3d_with_random_obstacles")
    plot.plot_tree(search_space, rrt.trees)
    if path is not None:
        plot.plot_path(search_space, path)
    plot.plot_obstacles(search_space, obstacles)
    plot.plot_start(search_space, start_pos)
    plot.plot_goal(search_space, end_pos)
    plot.draw(auto_open=True)


class GraspingRRT(RRT):
    def __init__(self, X, q, x_init, x_goal, max_samples, r, prc, 
                 object_offset=None, object_radius=0.02,
                 # [New] link7 obstacle-avoidance parameters
                 link7_offset=None, link7_radius=0.02):
        super().__init__(X, q, x_init, x_goal, max_samples, r, prc)
        self.object_offset = object_offset
        self.object_radius = object_radius  # object inflation radius
        # [New] link7 obstacle avoidance: fixed offset vector corresponding to the target quat
        self.link7_offset = link7_offset
        self.link7_radius = link7_radius

    def connect_to_point(self, tree, x_a, x_b):
        if not (self.trees[tree].V.count(x_b) == 0 and self.X.collision_free(x_a, x_b, self.r)):
            return False

        # [Modified] moved the steps computation out here, shared by the object and link7 checks
        steps = int(np.linalg.norm(np.array(x_b) - np.array(x_a)) / self.r) + 1

        # [Original] gripper midpoint (object) collision check
        if self.object_offset is not None:
            obj_pos_a = np.array(x_a) + self.object_offset
            obj_pos_b = np.array(x_b) + self.object_offset

            # Approach 1: interpolate along the path and check the inflated sphere at each interpolated point
            # steps = int(np.linalg.norm(np.array(x_b) - np.array(x_a)) / self.r) + 1  # [moved above, computed once]
            for i in range(steps + 1):
                t = i / steps
                obj_interp = obj_pos_a * (1 - t) + obj_pos_b * t

                # Check the obstacles within the radius around this point
                if not self.is_sphere_free(obj_interp, self.object_radius):
                    return False

            # Approach 2: alternatively just check the inflated start and end points (faster but may miss collisions)
            # if not self.is_sphere_free(obj_pos_a, self.object_radius):
            #     return False
            # if not self.is_sphere_free(obj_pos_b, self.object_radius):
            #     return False

        # [New] link7 obstacle-avoidance check: fixed offset vector corresponding to the target quat
        if self.link7_offset is not None:
            link7_pos_a = np.array(x_a) + self.link7_offset
            link7_pos_b = np.array(x_b) + self.link7_offset

            for i in range(steps + 1):
                t = i / steps
                link7_interp = link7_pos_a * (1 - t) + link7_pos_b * t

                if not self.is_sphere_free(link7_interp, self.link7_radius):
                    return False

        self.add_vertex(tree, x_b)
        self.add_edge(tree, x_b, x_a)
        return True

    def is_sphere_free(self, center, radius):
        """Check whether the sphere centered at `center` with `radius` collides with any obstacle"""
        # Assume obstacles are stored in self.X.obstacles as (n, 6) AABBs
        if hasattr(self.X, 'obstacles') and self.X.obstacles is not None:
            for obs in self.X.obstacles:
                # obs format: [x_min, y_min, z_min, x_max, y_max, z_max]
                # Compute the closest distance from the sphere center to the AABB
                closest_point = np.clip(center, obs[:3], obs[3:])
                distance = np.linalg.norm(center - closest_point)
                if distance < radius:
                    return False
        return True


class test_rrt(RRT):
    def __init__(self, X, q, x_init, x_goal, max_samples, r, prc=0.01):
        """
        Template RRT planner
        :param X: Search Space
        :param q: list of lengths of edges added to tree
        :param x_init: tuple, initial location
        :param x_goal: tuple, goal location
        :param max_samples: max number of samples to take
        :param r: resolution of points to sample along edge when checking for collisions
        :param prc: probability of checking whether there is a solution
        """
        super().__init__(X, q, x_init, x_goal, max_samples, r, prc)

    def get_path(self):
        if self.can_connect_to_goal(0):
            print("Can connect to goal")
            self.connect_to_goal(0)

            print("RRT FAILED - Copy for visualization:")
            print("=" * 60)
            all_vertices = list(self.trees[0].V.nearest((0, 0, 0), num_results=self.trees[0].V_count, objects="raw"))
            # print(f"tree_vertices = {all_vertices}")
            # print(f"start_pos = {self.x_init}")
            # print(f"target_pos = {self.x_goal}")
            # x_nearest = self.get_nearest(0, self.x_goal)
            # print(f"nearest_to_goal = {x_nearest}")
            # print(f"distance_to_goal = {np.linalg.norm(np.array(x_nearest) - np.array(self.x_goal))}")
            # print("=" * 60)

            return self.reconstruct_path(0, self.x_init, self.x_goal)

        # New: print failure info
        # print("RRT FAILED - Copy for visualization:")
        # print("=" * 60)
        # all_vertices = list(self.trees[0].V.nearest((0, 0, 0), num_results=self.trees[0].V_count, objects="raw"))
        # print(f"tree_vertices = {all_vertices}")
        # print(f"start_pos = {self.x_init}")
        # print(f"target_pos = {self.x_goal}")
        # x_nearest = self.get_nearest(0, self.x_goal)
        # print(f"nearest_to_goal = {x_nearest}")
        # print(f"distance_to_goal = {np.linalg.norm(np.array(x_nearest) - np.array(self.x_goal))}")
        # print("=" * 60)
        #
        # print("Could not connect to goal")
        # print("=" * 60)

        return None

    def new_and_near(self, tree, q):
        x_rand = self.X.sample_free()
        x_nearest = self.get_nearest(tree, x_rand)
        x_new = self.bound_point(steer(x_nearest, x_rand, q))

        in_tree = not self.trees[0].V.count(x_new) == 0
        in_obstacle = not self.X.obstacle_free(x_new)

        if in_tree or in_obstacle:
            if self.samples_taken < 10:  # only print the first 10 times
                print(f"new_and_near failed: in_tree={in_tree}, in_obstacle={in_obstacle}, x_new={x_new}")
            return None, None

        self.samples_taken += 1
        return x_new, x_nearest