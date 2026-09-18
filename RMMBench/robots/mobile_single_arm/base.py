import os
import numpy as np
from dm_control.utils.inverse_kinematics import qpos_from_site_pose
from RMMBench.robots.base import Robot
from RMMBench.utils.utils import matrix_to_quaternion,euler_to_quaternion

class MobileSingleArm(Robot):
    """
    Base class for mobile_single_arm robots.
    Params:
        n_dof: number of degrees of freedom for the robot
    """
    def __init__(self, n_dof=11, **kwargs):
        super().__init__(**kwargs)
        self._n_dof = n_dof
        self.count = 10
    
    @property
    def link_base(self):
        return self.mjcf_model.find("body", "mobile_base")


    def set_init_pos(self,**robot_params):
        pos = robot_params.get("pos", [0, 0, 0])
        euler = robot_params.get("euler", [0,0,1.57])
        link_base = self.link_base
        link_base._set_attribute("pos", pos)
        link_base._set_attribute("euler", euler)

        # physics.bind(link_base).xpos = pos
        # physics.bind(link_base).xquat = xquat


    def get_link_base_quat(self,physics) :
        matrix = np.array(physics.bind(self.link_base).xmat)
        q1 = np.array([0.7073882053998232, 0.0, 0.0, 0.7068252449235367])
        q2=np.array(matrix_to_quaternion(matrix))
        dot_product = np.dot(q1, q2)

        # Calculate the angle between the quaternions
        angle = 2 * np.arccos(np.clip(dot_product, -1.0, 1.0))  # Ensure dot product is within valid range for arccos
        angle_deg = np.degrees(angle)
        return angle_deg
        # return matrix_to_quaternion(matrix)

    def get_base_xmat(self, physics):
        return np.array(physics.bind(self.link_base).xmat)

    def get_joints_name(self):
        joint_names = []
        for joint in self.joints[:7]:
            p = self.mjcf_model.find("joint", joint.name)
            name = p.full_identifier
            joint_names.append(name)
        print("self.joint_name:",joint_names)

    @property
    def end_effector_site(self):
        return self.mjcf_model.find("site", "end_effector")

    @property
    def torso_height_joint(self):
        return self.mjcf_model.find("joint", "joint_torso_height")

    @property
    def mobile_forward_joint(self):
        return self.mjcf_model.find("joint", "joint_mobile_forward")

    @property
    def mobile_side_joint(self):
        return self.mjcf_model.find("joint", "joint_mobile_side")

    @property
    def mobile_yaw(self):
        return self.mjcf_model.find("joint", "joint_mobile_yaw")

    
    @property
    def position_limits(self):
        raise NotImplementedError
    
    @property
    def velocity_limits(self):
        raise NotImplementedError  
    
    @property
    def gripper_geoms(self):
        raise NotImplementedError
    
    @property
    def n_dof(self):
        return self._n_dof
    
    def get_qpos(self, physics):
        qposes = []
        # print("self.joints:", self.joints)
        #self.joints: [MJCF Element: <joint name="joint1"/>, MJCF Element: <joint name="joint2"/>, MJCF Element: <joint name="joint3"/>, MJCF Element: <joint name="joint4"/>, MJCF Element: <joint name="joint5"/>, MJCF Element: <joint name="joint6"/>, MJCF Element: <joint name="joint7"/>, MJCF Element: <joint name="finger_joint1" class="finger"/>, MJCF Element: <joint name="finger_joint2" class="finger"/>, MJCF Element: <joint name="joint_torso_height" type="slide" pos="0 0 0" axis="0 0 1" limited="true" range="0 0.34000000000000002" ref="0" armature="0" damping="0.01" frictionloss="1000"/>, MJCF Element: <joint name="joint_mobile_forward" type="slide" pos="-0.20999999999999999 0 0" axis="1 0 0" limited="false" armature="0" damping="0" frictionloss="250"/>, MJCF Element: <joint name="joint_mobile_side" type="slide" pos="-0.20999999999999999 0 0" axis="0 1 0" limited="false" armature="0" damping="0" frictionloss="250"/>, MJCF Element: <joint name="joint_mobile_yaw" type="hinge" pos="-0.20999999999999999 0 0" axis="0 0 1" limited="false" armature="0" damping="0" frictionloss="250"/>]

        # result = self.joints[:7] + self.joints[9:]
        # self.joints = result
        for joint in self.joints[:7]:
            # print("joint:",joint)
            qpos = physics.bind(joint).qpos
            qposes.append(qpos)
        for joint in self.joints[9:]:
            # print("mobile_joint:", joint)
            qpos = physics.bind(joint).qpos
            qposes.append(qpos)
        # qposes = np.array(qposes).reshape(-1)


        # print("qpos:", np.array(qposes).reshape(-1))
        return np.array(qposes).reshape(-1)

    def get_qvel(self, physics):
        qvels = []
        for joint in self.joints[:self.n_dof]:
            qvel = physics.bind(joint).qvel
            qvels.append(qvel)
        return np.array(qvels).reshape(-1)
    
    def get_qacc(self, physics):
        qaccs = []
        for joint in self.joints[:self.n_dof]:
            qacc = physics.bind(joint).qacc
            qaccs.append(qacc)
        return np.array(qaccs).reshape(-1)
    
    def set_base_position(self, pos):
        """
        change the base world position for the robot
        """
        assert len(pos) == 3, "pos must be a 3D vector"
        self.link_base.pos = pos
    
    def set_base_orientation(self, ori):
        """
        change the base world orientation for the robot in quaternion
        """
        assert len(ori) == 4 or len(ori) == 3, "quat must be a quaternion or euler"
        if len(ori) == 4:
            self.link_base.quat = ori
        elif len(ori) == 3:
            self.link_base.euler = ori
    
    def get_qpos_from_ee_pos(self, physics, pos, quat=None, inplace=False, **kwargs):
        """
        get the joint angles by inverse kinematics from the end effector pose
        """
        joint_names = []
        for joint in self.joints[:7]:
            p = self.mjcf_model.find("joint", joint.name)
            name = p.full_identifier
            joint_names.append(name)
        ik_result = qpos_from_site_pose(physics,
                                        site_name=self.end_effector_site.full_identifier,
                                        target_pos=pos,
                                        target_quat=quat,
                                        inplace=inplace,
                                        joint_names = joint_names,
                                        **kwargs)
        success = ik_result.success
        target_qpos = ik_result.qpos
        # print("target_qpos:",target_qpos)
        mobile_base_distance = self.get_mobilebase_distance(physics)

        t_pos = np.concatenate([target_qpos[4:11], mobile_base_distance])
        return success, t_pos
    
    def get_end_effector_pos(self, physics=None):
        # print("physics.bind(self.end_effector_site).xpos:",physics.bind(self.end_effector_site).xpos)
        return physics.bind(self.end_effector_site).xpos
    
    def get_end_effector_quat(self, physics=None):
        matrix = physics.bind(self.end_effector_site).xmat
        return matrix_to_quaternion(matrix)




    def get_mobile_base_pos(self, physics=None):
        return physics.bind(self.link_base).xpos

    def get_mobile_base_quat(self, physics=None):
        matrix = physics.bind(self.link_base).xmat
        return matrix_to_quaternion(matrix)

    def get_mobilebase_distance(self,physics=None):
        # return physics.bind(self.mobile_forward_joint).qpos
        qposes = []
        for joint in self.joints[9:]:
            if joint.name == "head_yaw":
                qpos = [0]
            else:
                qpos = physics.bind(joint).qpos
            qposes.append(qpos)
        return np.array(qposes).reshape(-1)

    def get_mobilebase_distance_continue_look(self,physics=None):
        # return physics.bind(self.mobile_forward_joint).qpos
        qposes = []
        for joint in self.joints[9:]:
            qpos = physics.bind(joint).qpos
            qposes.append(qpos)
        return np.array(qposes).reshape(-1)







    # def get_torso_height_pos(self, physics=None):
    #     return physics.bind(self.torso_height_joint).pos
    #
    # def get_mobile_forward_pos(self, physics=None):
    #     return physics.bind(self.mobile_forward_joint).xpos
    #
    # def get_mobile_side_pos(self, physics=None):
    #     return physics.bind(self.mobile_side_joint).xpos
    #
    # def get_mobile_yaw_pos(self, physics=None):
    #     return physics.bind(self.mobile_yaw_joint).xpos




