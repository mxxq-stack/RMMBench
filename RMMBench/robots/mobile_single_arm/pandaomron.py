import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from RMMBench.robots.mobile_single_arm.base import MobileSingleArm
from RMMBench.utils.register import register
from RMMBench.utils.utils import quaternion_to_euler, create_mesh_box, quaternion_to_matrix, \
    compute_rotation_quaternion, normalize


@register.add_robot("pandaomron")
class PandaOmron(MobileSingleArm):
    def __init__(self, **kwargs):
        super().__init__(name=kwargs.get("name", "pandaomron"),
                         # n_dof=kwargs.get("n_dof", 7),
                         n_dof=kwargs.get("n_dof", 12),
                         **kwargs)

    def _add_camera_to_wrist(self, **camera_params):
        name = camera_params.get("name", "wrist_cam")
        pos = camera_params.get("pos", [0, 0, 0])
        euler = camera_params.get("euler", [np.pi, 0, np.pi / 2])
        fovy = camera_params.get("fovy", 30)
        body_name = camera_params.get("body_name", "hand")
        # gripper_site = self.mjcf_model.find("body", "hand")
        gripper_site = self.mjcf_model.find("body", body_name)

        gripper_site.add("camera", name=name, pos=pos, euler=euler, fovy=fovy)

    def get_ee_open_distance(self, physics=None):
        left_finger_joint = self.mjcf_model.find("joint", "finger_joint1")
        right_finger_joint = self.mjcf_model.find("joint", "finger_joint2")
        left_finger_pos = physics.bind(left_finger_joint).qpos
        right_finger_pos = physics.bind(right_finger_joint).qpos
        distance = left_finger_pos + left_finger_pos
        return distance

    @property
    def link_base(self):
        return self._mjcf_model.find("body", "mobile_base")


    def get_body_info(self, body_name = None,physics=None):
        if body_name is not None:
            body_mjcf = self._mjcf_model.find("body", body_name)
            position = physics.bind(body_mjcf).xpos.copy()
            quat = physics.bind(body_mjcf).xquat.copy()
            euler = quaternion_to_euler(quat)
            info = {
                "position": position,
                "euler": euler,
            }
            return info
        pass


    @property
    def cam_head(self):
        return self._mjcf_model.find("body", "cam_head")

    def get_link_base_info(self, physics=None):
        position = physics.bind(self.link_base).xpos.copy()
        quat = physics.bind(self.link_base).xquat.copy()
        euler = quaternion_to_euler(quat)
        info = {
            "position": position,
            "euler": euler,
        }
        return info

    @property
    def end_effector_site(self):
        return self.mjcf_model.find("site", "end_effector")

    @property
    def torso_height_joint(self):
        return self.mjcf_model.find("joint", "joint_torso_height")

    @property
    def head_yaw_joint(self):
        return self.mjcf_model.find("joint", "head_yaw")

    @property
    def mobile_forward_joint(self):
        return self.mjcf_model.find("joint", "joint_mobile_forward")

    def get_mobile_forward_axis(self, physics):
        return physics.bind(self.mobile_forward_joint).axis

    # def set_mobile_forward_axis(self,physics):
    #     physics.bind(self.mobile_forward_joint).axis = [-1, 0, 0]

    @property
    def mobile_side_joint(self):
        return self.mjcf_model.find("joint", "joint_mobile_side")

    @property
    def mobile_yaw_joint(self):
        return self.mjcf_model.find("joint", "joint_mobile_yaw")

    @property
    def gripper_geoms(self):
        left_finger_geoms = self.mjcf_model.find("body", "left_finger").find_all("geom")
        right_finger_geoms = self.mjcf_model.find("body", "right_finger").find_all("geom")
        return left_finger_geoms + right_finger_geoms

    def gripper_move_orientation(self, physics):
        ee_site = self.end_effector_site
        ee_site_pos = physics.bind(ee_site).xpos
        ee_move_site = self._mjcf_model.find("site", "end_effector_move")
        ee_move_site_pos = physics.bind(ee_move_site).xpos
        move_quat = compute_rotation_quaternion(ee_site_pos, ee_move_site_pos)
        return move_quat

    @property
    def ee2move_transform(self):
        R_ee = R.from_euler("xyz", [0, 0, 0], degrees=True).as_matrix()
        R_move = R.from_euler("xyz", [0, 0, -90], degrees=True).as_matrix()
        ee2move_transform = np.dot(R_move, R_ee.T)
        return ee2move_transform

    def get_ee_open_state(self, physics=None):
        left_finger_joint = self.mjcf_model.find("joint", "finger_joint1")
        right_finger_joint = self.mjcf_model.find("joint", "finger_joint2")
        left_finger_pos = physics.bind(left_finger_joint).qpos
        right_finger_pos = physics.bind(right_finger_joint).qpos
        # print("(left_finger_pos + right_finger_pos):",(left_finger_pos + right_finger_pos))

        if (left_finger_pos + right_finger_pos) < 0.079:
            return True  # BUG: should be False
        else:
            return False

    # def initialize_episode(self, physics, random_state):
    #     super().initialize_episode(physics, random_state)
    #     for i, qpos in enumerate(self.default_qpos):
    #         # print("qpos:", qpos)
    #         if i < 7: physics.bind(self.mjcf_model.find("joint", f"joint{i + 1}")).qpos = qpos
    #         elif i < 9 : physics.bind(self.mjcf_model.find("joint", f"finger_joint{i - 6}")).qpos = qpos
    #         elif i == 9: physics.bind(self.mjcf_model.find("joint", "joint_torso_height")).qpos = qpos
    #         elif i == 10: physics.bind(self.mjcf_model.find("joint", "head_yaw")).qpos = qpos
    #         elif i == 11: physics.bind(self.mjcf_model.find("joint", "joint_mobile_forward")).qpos = qpos
    #         elif i == 12: physics.bind(self.mjcf_model.find("joint", "joint_mobile_side")).qpos = qpos
    #         elif i == 13: physics.bind(self.mjcf_model.find("joint", "joint_mobile_yaw")).qpos = qpos

    def initialize_episode(self, physics, random_state):
        super().initialize_episode(physics, random_state)
        if len(self.default_qpos) == 13:
            for i, qpos in enumerate(self.default_qpos):
                # print("qpos:", qpos)
                if i < 7:
                    physics.bind(self.mjcf_model.find("joint", f"joint{i + 1}")).qpos = qpos
                elif i < 9:
                    physics.bind(self.mjcf_model.find("joint", f"finger_joint{i - 6}")).qpos = qpos
                elif i == 9:
                    physics.bind(self.mjcf_model.find("joint", "joint_torso_height")).qpos = qpos
                elif i == 10:
                    physics.bind(self.mjcf_model.find("joint", "joint_mobile_forward")).qpos = qpos
                elif i == 11:
                    physics.bind(self.mjcf_model.find("joint", "joint_mobile_side")).qpos = qpos
                elif i == 12:
                    physics.bind(self.mjcf_model.find("joint", "joint_mobile_yaw")).qpos = qpos

            self.init_ee_qpos = self.get_qpos(physics)
            physics.data.ctrl = self.default_qpos
        elif len(self.default_qpos)==14:
            for i, qpos in enumerate(self.default_qpos):
                # print("qpos:", qpos)
                if i < 7: physics.bind(self.mjcf_model.find("joint", f"joint{i + 1}")).qpos = qpos
                elif i < 9 : physics.bind(self.mjcf_model.find("joint", f"finger_joint{i - 6}")).qpos = qpos
                elif i == 9: physics.bind(self.mjcf_model.find("joint", "joint_torso_height")).qpos = qpos
                elif i == 10: physics.bind(self.mjcf_model.find("joint", "head_yaw")).qpos = qpos
                elif i == 11: physics.bind(self.mjcf_model.find("joint", "joint_mobile_forward")).qpos = qpos
                elif i == 12: physics.bind(self.mjcf_model.find("joint", "joint_mobile_side")).qpos = qpos
                elif i == 13: physics.bind(self.mjcf_model.find("joint", "joint_mobile_yaw")).qpos = qpos

            self.init_ee_qpos = self.get_qpos(physics)
            physics.data.ctrl = self.default_qpos

    def gripper_pcd(self, target_pos=None, target_quat=None, color=None):
        """
        get abstract&simple gripper point cloud for collision detection
        """
        center = np.asarray(target_pos)
        quat = np.asarray(target_quat)
        rot_matrix = quaternion_to_matrix(quat)
        left_finger = create_mesh_box(width=0.02, height=0.02, depth=0.08, dx=-0.01, dy=-0.055, dz=-0.04)
        right_finger = create_mesh_box(width=0.02, height=0.02, depth=0.08, dx=-0.01, dy=0.04, dz=-0.04)
        gripper_link = create_mesh_box(width=0.05, height=0.18, depth=0.08, dx=-0.025, dy=-0.09, dz=-0.12)

        move_point = np.array([np.array([0, 0, 0]), np.array([0, 0, 0.1])])
        move_point = np.dot(rot_matrix, move_point.T).T + center
        move_vector = normalize(move_point[1] - move_point[0])
        move_point_pcd = o3d.geometry.PointCloud()
        move_point_pcd.points = o3d.utility.Vector3dVector(move_point)
        move_point_pcd.colors = o3d.utility.Vector3dVector([[1, 0, 0], [1, 0, 0]])

        left_points = np.array(left_finger.vertices)
        left_triangles = np.array(left_finger.triangles)

        right_points = np.array(right_finger.vertices)
        right_triangles = np.array(right_finger.triangles) + 8

        gripper_link_points = np.array(gripper_link.vertices)
        gripper_link_triangles = np.array(gripper_link.triangles) + 16

        vertices = np.concatenate([left_points, right_points, gripper_link_points], axis=0)
        vertices = np.dot(rot_matrix, vertices.T).T + center
        triangles = np.concatenate([left_triangles, right_triangles, gripper_link_triangles], axis=0)

        colors = np.array([[0, 0, 1] for _ in range(len(vertices))]) if color is None else [color for _ in
                                                                                            range(len(vertices))]

        gripper = o3d.geometry.TriangleMesh()
        gripper.vertices = o3d.utility.Vector3dVector(vertices)
        gripper.triangles = o3d.utility.Vector3iVector(triangles)
        gripper.vertex_colors = o3d.utility.Vector3dVector(colors)
        return gripper.sample_points_uniformly(number_of_points=1000), move_vector

    def get_ee_state(self, physics):
        pos = np.array(self.get_end_effector_pos(physics))
        quat = np.array(self.get_end_effector_quat(physics))
        open = np.array(self.get_ee_open_state(physics)).astype(np.float32).reshape((1,))
        return np.concatenate([pos, quat, open])

    def ee_offset(self, physics):
        grasp_pos = self.get_end_effector_pos(physics)
        ee_end_site = self.mjcf_model.find("site", "end_effector_move")
        ee_end_pos = physics.bind(ee_end_site).xpos
        return grasp_pos - ee_end_pos

    # [New] Get the world coordinates of the link7 obstacle marker point, used for link7 collision checking in RRT path planning
    def get_link7_obstacle_point_pos(self, physics):
        site = self.mjcf_model.find("site", "link7_obstacle_point")
        if site is None:
            return None
        return physics.bind(site).xpos.copy()

    def find_wrist_camera(self, physics=None):
        return self.mjcf_model.find("camera", "Franka_wrist_cam")

    def get_wrist_camera_qpose(self, physics=None):
        pos = physics.bind(self.find_wrist_camera(physics)).xpos
        quat = physics.bind(self.find_wrist_camera(physics)).xmat
        # pos=np.array(pos)
        # quat=np.array(quat)
        qpos = [list(pos), list(quat)]
        return qpos

    # ==================== Transparency control ====================

    def _ensure_material_alpha_saved(self, physics):
        """Save the original alpha values of all materials on the first call."""
        if hasattr(self, "_material_alpha_records"):
            return
        self._material_alpha_records = []
        for material in self.mjcf_model.find_all("material"):
            orig_alpha = physics.bind(material).rgba[3]
            self._material_alpha_records.append((material, orig_alpha))

    def set_transparent(self, physics, alpha=0.0, exclude_bodies=None):
        """
        Set the robot's transparency (modifies material rgba[3] at runtime).

        Args:
            physics: dm_control Physics object
            alpha: transparency, float, default 0 (fully transparent)
            exclude_bodies: list[str], list of body names to exclude.
                            The materials referenced by these bodies and their child bodies
                            keep their original opacity.
                            Note: materials are globally shared, so if an excluded body shares
                            the same material with other bodies, that material will also
                            stay opaque.
        """
        if exclude_bodies is None:
            exclude_bodies = []

        self._ensure_material_alpha_saved(physics)

        # Collect the materials referenced by all geoms under the excluded bodies
        exclude_materials = set()
        for body_name in exclude_bodies:
            body = self.mjcf_model.find("body", body_name)
            if body is not None:
                for geom in body.find_all("geom"):
                    if geom.material is not None:
                        exclude_materials.add(geom.material)

        # Set the transparency
        for material, orig_alpha in self._material_alpha_records:
            if material in exclude_materials:
                physics.bind(material).rgba[3] = orig_alpha
            else:
                physics.bind(material).rgba[3] = alpha

    def restore_opacity(self, physics):
        """
        Restore the original transparency of all the robot's materials.
        Does nothing if set_transparent was never called.
        """
        if not hasattr(self, "_material_alpha_records"):
            return
        for material, orig_alpha in self._material_alpha_records:
            physics.bind(material).rgba[3] = orig_alpha

    # ----- geom-level transparency (precise per-body control) -----

    def _ensure_geom_alpha_saved(self, physics):
        """Save the original alpha values of all geoms under every body on the first call."""
        if hasattr(self, "_geom_alpha_records"):
            return
        self._geom_alpha_records = []
        for body in self.mjcf_model.find_all("body"):
            for geom in body.find_all("geom"):
                orig_alpha = physics.bind(geom).rgba[3]
                self._geom_alpha_records.append((geom, orig_alpha))

    def set_transparent_geom(self, physics, alpha=0.0, exclude_bodies=None):
        """
        Set the robot's transparency (modifies geom rgba[3] at runtime, precise to per-body).

        Unlike set_transparent (material level), this method operates on each geom
        individually, so exclude_bodies can precisely exclude the given bodies and
        their child bodies, unaffected by material sharing.

        Args:
            physics: dm_control Physics object
            alpha: transparency, float, default 0 (fully transparent)
            exclude_bodies: list[str], list of body names to exclude.
                            The geoms under these bodies and their child bodies keep
                            their original opacity.
        """
        if exclude_bodies is None:
            exclude_bodies = []

        self._ensure_geom_alpha_saved(physics)

        # Collect the geoms under the excluded bodies and their child bodies
        exclude_geoms = set()
        for body_name in exclude_bodies:
            body = self.mjcf_model.find("body", body_name)
            if body is not None:
                for geom in body.find_all("geom"):
                    exclude_geoms.add(id(geom))

        # Set the transparency
        for geom, orig_alpha in self._geom_alpha_records:
            if id(geom) in exclude_geoms:
                physics.bind(geom).rgba[3] = orig_alpha
            else:
                physics.bind(geom).rgba[3] = alpha

    def restore_opacity_geom(self, physics):
        """
        Restore the original transparency of all geoms (geom level).
        """
        if not hasattr(self, "_geom_alpha_records"):
            return
        for geom, orig_alpha in self._geom_alpha_records:
            physics.bind(geom).rgba[3] = orig_alpha

    # def after_substep(self, physics, random_state):
    #     for material in self.mjcf_model.find_all("material"):
    #         physics.bind(material).rgba[3] = 0
        # self.set_transparent_geom(physics, alpha=0.0, exclude_bodies=["link6","wheeled_base"])

