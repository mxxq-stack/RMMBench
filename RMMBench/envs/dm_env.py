import numpy as np
import math
import mediapy
import os
from dm_control import composer
from RMMBench.utils.utils import euler_to_quaternion, expand_mask, compute_camera_positions_and_bounds, CAMERA_NAME_MAP, euler_to_rotation_matrix
from RMMBench.utils.camera_utils import rotation_6d_to_matrix
from RMMBench.utils.depth2cloud import rotMatList2NPRotMat, quat2Mat, posRotMat2Mat, PointCloudGenerator

class LM4ManipDMEnv(composer.Environment):
    def __init__(self, reset_wait_step=10, **kwargs):
        super().__init__(**kwargs)
        self.timestep = 0
        self.reset_wait_step = reset_wait_step
        self.n_distractor = kwargs.get("n_distractor", 0)
        self.render_options = dict(
            height=560,
            width=560,
        )
        self.register_pcd_generator()
        
    def reset(self):
        self.timestep = 0
        timestep = super().reset()
        self.cancel_gravity_and_improve_fluid()
        for i in range(self.reset_wait_step):
            # print(f"reset {i}")
            self.step()
        self.reset_gravity_and_fluid(self.physics)
        for _ in range(self.reset_wait_step):
            self.step()
        self.register_pcd_generator()
        return timestep


    # def render(self, **kwargs):
    #     return self.physics.render(**kwargs)
    def render(self, **kwargs):
        segmentation = kwargs.get('segmentation', False)
        camera_id = kwargs.get('camera_id', -1)
        return self.physics.render(**kwargs)

    def step(self, action=None):
        # print("dm_env.step().action:",action)
        if action is None: # robot stay in static
            action = self.robot.get_qpos(self.physics)
            # print("len(action1):", len(action))

            # Uniformly extract the inner scalar values
            flat_action = []
            for a in action:
                # If it is a wrapper or an indexable object, but not a scalar
                if hasattr(a, '__getitem__') and not isinstance(a, (int, float)):
                    # a[0] is a list/array -> flatten it, otherwise take the scalar
                    if isinstance(a[0], (list, np.ndarray)):
                        flat_action.extend(a[0])
                    else:
                        flat_action.append(a[0])
                else:
                    # Plain scalar or list
                    flat_action.append(a)

            # Convert to a 1D numpy array
            action = np.array(flat_action)
            # action = np.concatenate([action, np.zeros(1)], axis=-1)
            ee_state = self.robot.get_ee_open_state(self.physics)
            if ee_state:
                griper_state = np.zeros((2))
            else:
                griper_state = 0.06 * np.ones((2))


            if len(action)==7:
                action = np.concatenate([action, griper_state], axis=-1)
            elif len(action)==11:
                action = np.concatenate([action[:7], griper_state,action[7:]], axis=-1)
            elif len(action) == 12:
                action = np.concatenate([action[:7], griper_state, action[7:]], axis=-1)
                # print("action:", action)
            # if len(action) ==9:
            #     action = np.concatenate([action, np.zeros((4))], axis=-1)


            # print("action:", action)
            # print("len(action):", len(action))
            return super().step(action)
        self.timestep += 1
        # print("dm_env_action:", action)
        return super().step(action)
    
    @property
    def mjmodel(self):
        return self.physics.model
    
    @property
    def mjdata(self):
        return self.physics.data

    @property
    def robot(self):
        return self.task.robot
    
    def get_element_by_name(self, name, type):
        if type == "camera":
            element = self.mjmodel.cam(name)
        else:
            element = self.task.get_element_by_name(name, type)
        return element
    
    def get_xpos_by_name(self, name, type="geom"):
        """
        get the position of the entity by name
        """
        element = self.get_element_by_name(name, type)
        data = self.physics.bind(element)
        return data.xpos
    
    def get_xquat_by_name(self, name, type="geom"):
        """
        get the orientation of the entity by name
        """
        element = self.get_element_by_name(name, type)
        data = self.physics.bind(element)
        return data.xquat
    
    def set_xpos_by_name(self, name, type, pos):
        element = self.get_element_by_name(name, type)
        assert len(pos) == 3 or pos.shape == (3, 0), "pos must be a 3D vector"
        element.pos = pos
        self.step()
        
    
    def set_xquat_by_name(self, name, type, quat=None, euler=None):
        obj = self.get_element_by_name(name, type)
        assert (quat is None) != (euler is None), "quat and euler must be provided exclusively"
        if euler is not None:
            quat = euler_to_quaternion(roll=euler[0], pitch=euler[1], yaw=euler[2])
        obj.quat = quat
        self.step()
    
    def attach_entity(self, entity):
        self.task.add_free_entity(entity)
        self.step()
    
    def get_ee_pos(self):
        return self.robot.get_end_effector_pos(self.physics)


    def get_ee_quat(self):
        return self.robot.get_end_effector_quat(self.physics)

    def get_init_ee_pose(self):
        return self.robot.get_init_pose(self.physics)

    def get_robot_wrist_camera_qpose(self):
        qpos = self.robot.get_wrist_camera_qpose(self.physics)
        return qpos
    
    def get_camera_matrix(self, cam_id, width, height):
        if cam_id >= self.physics.model.ncam:
            raise ValueError(f"cam_id {cam_id} is out of range")
        fovy = math.radians(self.physics.model.cam_fovy[cam_id])
        f = height / (2 * math.tan(fovy / 2))
        intrinsic_mat = np.array(((f, 0, width / 2), (0, f, height / 2), (0, 0, 1)))
        
        cam_pos = self.physics.data.cam_xpos[cam_id]
        c2b_r = rotMatList2NPRotMat(self.physics.model.cam_mat0[cam_id])
        b2w_r = quat2Mat([0, 1, 0, 0])
        cam_rot_mat = np.matmul(c2b_r, b2w_r)
        extrinsic_mat = posRotMat2Mat(cam_pos, cam_rot_mat)
        return intrinsic_mat, extrinsic_mat

    def get_camera_matrix_physics(self, cam_id, width, height):
        if cam_id >= self.physics.model.ncam:
            raise ValueError(f"cam_id {cam_id} is out of range")
        fovy = math.radians(self.physics.model.cam_fovy[cam_id])
        f = height / (2 * math.tan(fovy / 2))
        intrinsic_mat = np.array(((f, 0, width / 2), (0, f, height / 2), (0, 0, 1)))

        cam_pos = self.physics.data.cam_xpos[cam_id]
        # cam_xmat is MuJoCo's native camera rotation (camera-to-world, looking down -z).
        # It must be flipped 180° around the x axis to convert to the Open3D/OpenCV
        # convention (looking down +z), keeping the same coordinate convention as the
        # old quat2Mat([0,1,0,0]) code.
        cam_xmat = self.physics.data.cam_xmat[cam_id].reshape((3, 3))
        flip_x = np.diag([1, -1, -1]).astype(np.float64)
        cam_rot_mat = cam_xmat @ flip_x
        extrinsic_mat = posRotMat2Mat(cam_pos, cam_rot_mat)
        return intrinsic_mat, extrinsic_mat

    # def get_camera_matrix(self, cam_id, width, height):
    #     if cam_id >= self.physics.model.ncam:
    #         raise ValueError(f"cam_id {cam_id} is out of range")
    #     fovy = math.radians(self.physics.model.cam_fovy[cam_id])
    #     f = height / (2 * math.tan(fovy / 2))
    #     intrinsic_mat = np.array(((f, 0, width / 2), (0, f, height / 2), (0, 0, 1)))
    #
    #     cam_pos = self.physics.data.cam_xpos[cam_id]
    #     c2b_r = rotMatList2NPRotMat(self.physics.model.cam_mat0[cam_id])
    #     b2w_r = quat2Mat([0, 1, 0, 0])
    #     cam_rot_mat = np.matmul(c2b_r, b2w_r)
    #     extrinsic_mat = posRotMat2Mat(cam_pos, cam_rot_mat)

        # return intrinsic_mat, extrinsic_mat

    def get_observation(self, require_pcd=True):
        observation = dict()
        multi_view_rgb, multi_view_depth, multi_view_seg = [], [], []
        instrinsic_matrixs, extrinsic_matrixs = [], []
        for cam in range(self.physics.model.ncam):
            multi_view_rgb.append(self.render(camera_id=cam, **self.render_options))
            multi_view_depth.append(self.render(camera_id=cam, **self.render_options, depth=True))
            multi_view_seg.append(self.render(camera_id=cam, **self.render_options, segmentation=True))
            instrinsic, extrinsic = self.get_camera_matrix(cam, **self.render_options)
            instrinsic_matrixs.append(instrinsic)
            extrinsic_matrixs.append(extrinsic)
        observation["q_state"] = np.array(self.robot.get_qpos(self.physics))
        observation["q_velocity"] = np.array(self.robot.get_qvel(self.physics))
        observation["q_acceleration"] = np.array(self.robot.get_qacc(self.physics))
        observation["rgb"] = np.array(multi_view_rgb)
        observation["depth"] = np.array(multi_view_depth)
        observation["segmentation"] = np.array(multi_view_seg)
        observation["robot_mask"] = np.where((observation["segmentation"][..., 0] <= 72)&(observation["segmentation"][..., 0] > 0), 0, 1).astype(np.uint8)
        observation["instrinsic"] = np.array(instrinsic_matrixs)
        observation["extrinsic"] = np.array(extrinsic_matrixs)
        self.pcd_generator.physics = self.physics
        if require_pcd:
            observation["masked_point_cloud"] = self.pcd_generator.generate_pcd_from_rgbd(target_id=list(range(self.physics.model.ncam - 1)), 
                                                                                            rgb=multi_view_rgb,
                                                                                            depth=multi_view_depth,
                                                                                            mask=expand_mask(observation["robot_mask"]))
            observation["point_cloud"] = self.pcd_generator.generate_pcd_from_rgbd(target_id=list(range(self.physics.model.ncam - 1)),
                                                                                    rgb=multi_view_rgb,
                                                                                    depth=multi_view_depth)
        observation["ee_state"] = self.robot.get_ee_state(self.physics)
        observation["grasped_obj_name"] = self.get_grasped_entity()
        observation.update(self.task.task_observables)
        return observation
    
    def get_rgbd(self, cam):
        rgb = self.render(camera_id=cam, **self.render_options)
        depth = self.render(camera_id=cam, **self.render_options, depth=True)
        seg = self.render(camera_id=cam, **self.render_options, segmentation=True)

        height = self.render_options['height']
        width = self.render_options['width']
        if cam >= self.physics.model.ncam:
            raise ValueError(f"cam_id {cam} is out of range")
        fovy = math.radians(self.physics.model.cam_fovy[cam])
        f = height / (2 * math.tan(fovy / 2))
        intrinsic = np.array(((f, 0, width / 2), (0, f, height / 2), (0, 0, 1)))

        cam_pos = self.physics.data.cam_xpos[cam]
        cam_rot_mat = self.physics.data.cam_xmat[cam].reshape((3, 3))
        extrinsic = posRotMat2Mat(cam_pos, cam_rot_mat)

        return rgb, depth, seg, intrinsic, extrinsic

    def get_camera_parse_physics(self, cam):
        rgb = self.render(camera_id=cam, **self.render_options)
        depth = self.render(camera_id=cam, **self.render_options, depth=True)
        seg = self.render(camera_id=cam, **self.render_options, segmentation=True)

        intrinsic, extrinsic = self.get_camera_matrix_physics(cam, **self.render_options)


        return rgb, depth, seg, intrinsic, extrinsic

    def get_camera_parse(self, cam):
        rgb = self.render(camera_id=cam, **self.render_options)
        depth = self.render(camera_id=cam, **self.render_options, depth=True)
        seg = self.render(camera_id=cam, **self.render_options, segmentation=True)

        intrinsic, extrinsic = self.get_camera_matrix(cam, **self.render_options)
        height = self.render_options['height']
        width = self.render_options['width']


        return rgb, depth, seg, intrinsic, extrinsic
    
    def cancel_gravity_and_improve_fluid(self):
        """
        This function is used when initializing the scene.
        """
        self.task._arena.mjcf_model.option.flag.gravity = "disable"
        geoms = self.task._arena.mjcf_model.find_all("geom")
        for geom in geoms:
            geom.fluidshape = "ellipsoid"
            geom.fluidcoef = [1e4, 1e4, 1e4, 1e4, 1e4]
    
    def reset_gravity_and_fluid(self, physics):
        self.task._arena.mjcf_model.option.flag.gravity = "enable"
        geoms = self.task._arena.mjcf_model.find_all("geom")
        for geom in geoms:
            geom.fluidshape = "none"
            geom.fluidcoef = [0.5, 0.25, 1.5, 1.0, 1.0]
            physics.data.qvel = 0
    
    def register_pcd_generator(self):
        """
        Dynamically register the point cloud generator based on current robot pose.
        Only computes bounds; camera positions are managed by dm_task.reset_camera_views.
        """
        robot_info = self.robot.get_link_base_info(self.physics)
        robot_pos = robot_info["position"]
        robot_euler = robot_info["euler"]

        _, _, min_bound, max_bound = compute_camera_positions_and_bounds(
            robot_pos, robot_euler
        )

        self.pcd_generator = PointCloudGenerator(
            self.physics,
            min_bound=min_bound.tolist(),
            max_bound=max_bound.tolist(),
            **self.render_options
        )

    def update_pcd_generator(self):
        """
        External interface: recompute and update point cloud generator bounds
        and camera positions/orientations. Call before pick, place, etc. to
        ensure point cloud coverage matches the current robot pose.
        """
        # Update point cloud bounds
        self.register_pcd_generator()

        # Update left/right camera positions and orientations
        self._update_left_right_cameras

    @property
    def _update_left_right_cameras(self):
        """
        Update left/right camera positions and orientations based on current robot pose.
        Uses physics.bind() to modify camera elements at runtime, ensuring changes persist.
        """
        from RMMBench.utils.camera_utils import rotation_6d_to_matrix

        robot_info = self.robot.get_link_base_info(self.physics)
        robot_pos = robot_info["position"]
        robot_euler = robot_info["euler"]

        left_pos, right_pos, _, _ = compute_camera_positions_and_bounds(
            robot_pos, robot_euler
        )

        # Base xyaxes for robot facing +Y (yaw=1.57)
        right_xyaxes_base = np.array([0.733, 0.681, 0.000, -0.134, 0.144, 0.981])
        left_xyaxes_base = np.array([0.733, -0.681, 0.000, 0.134, 0.144, 0.981])

        # Compute delta rotation from base yaw=1.57 to current yaw
        delta_yaw = robot_euler[2] - 1.57
        R_delta = euler_to_rotation_matrix([0, 0, delta_yaw], order='xyz')

        # Helper to rotate xyaxes
        def rotate_xyaxes(xyaxes_base, R_delta):
            R_base = rotation_6d_to_matrix(xyaxes_base)
            R_new = R_delta @ R_base
            return " ".join([str(v) for v in np.concatenate([R_new[:, 0], R_new[:, 1]])])

        # Get camera elements from mjcf_model
        cameras = self.task._arena.mjcf_model.find_all("camera")

        # Update left camera via physics.bind()
        left_cam = cameras[CAMERA_NAME_MAP["left"]]
        left_cam.pos = " ".join([str(v) for v in left_pos])
        left_cam.xyaxes = rotate_xyaxes(left_xyaxes_base, R_delta)
        left_bound = self.physics.bind(left_cam)
        left_bound.pos[:] = left_pos
        # xyaxes is not directly bindable; mjcf_model change + forward() will propagate

        # Update right camera via physics.bind()
        right_cam = cameras[CAMERA_NAME_MAP["right"]]
        right_cam.pos = " ".join([str(v) for v in right_pos])
        right_cam.xyaxes = rotate_xyaxes(right_xyaxes_base, R_delta)
        right_bound = self.physics.bind(right_cam)
        right_bound.pos[:] = right_pos

        # Synchronize physics state after camera updates
        self.physics.forward()
    
    def get_grasped_entity(self):
        name_list = []
        entity_list = []

        for name, entity in self.task.entities.items():
            skip = any(obj in name for obj in ["fridge", "microwave"])
            if skip:
                continue
            if hasattr(entity, "is_grasped"):
                if entity.is_grasped(self.physics, self.robot):
                    name_list.append(name)
                    entity_list.append(entity)
        return name_list, entity_list
    
    def _reset_attempt(self):
        self.task.reset_distractors(n_distractor=self.n_distractor)
        self._hooks.refresh_entity_hooks()
        return super()._reset_attempt()

    def get_obstacle_pcd(self):
        multi_view_rgb, multi_view_depth, multi_view_seg = [], [], []
        for cam in range(self.physics.model.ncam):
            multi_view_rgb.append(self.render(camera_id=cam, **self.render_options))
            multi_view_depth.append(self.render(camera_id=cam, **self.render_options, depth=True))
            multi_view_seg.append(self.render(camera_id=cam, **self.render_options, segmentation=True))
        segmentation = np.array(multi_view_seg)
        total_mask = np.ones_like(segmentation[..., 0])
        robot_mask = np.where((segmentation[..., 0] <= 72)&(segmentation[..., 0] > 0), 0, 1).astype(np.uint8)
        total_mask *= robot_mask
        grasped_obj_name_list, grasped_obj = self.get_grasped_entity()
        print("grasped_obj_name_list:", grasped_obj_name_list)
        for name in grasped_obj_name_list:
            geom_ids = [self.physics.bind(geom).element_id for geom in self.task.entities[name].geoms]
            obj_mask = np.where((segmentation[..., 0] <= max(geom_ids))&(segmentation[..., 0] >= min(geom_ids)), 0, 1).astype(np.uint8)
            total_mask *= obj_mask
        obstacle_pcd= self.pcd_generator.generate_pcd_from_rgbd(target_id=list(range(self.physics.model.ncam )),
                                                            rgb=multi_view_rgb,
                                                        depth=multi_view_depth,
                                                        mask=expand_mask(total_mask))

        # self.pcd_visual(obstacle_pcd)
        return obstacle_pcd

    def pcd_visual(self,obstacle_pcd):
        import open3d as o3d
        tree_vertices = [(2.5403517503122752, -2.9032427886957692, 1.0341007319180848),
                         (2.4627196396346704, -2.98298011750748, 1.044493620402529),
                         (2.500927099020845, -2.953284060594328, 1.057076714261538),
                         (2.4401713680443016, -3.0274890305453668, 1.047738941414819),
                         (2.550544053326204, -2.9480692136221895, 1.0537656394334607),
                         (2.568204046638424, -3.0040349486222344, 1.0124620502138701),
                         (2.5648521363677648, -2.9953358220188946, 1.061585261569364),
                         (2.594077067485069, -3.035901059250555, 1.062189355542114),
                         (2.6419453353728515, -3.033724616256129, 1.0764684320289006),
                         (2.6711348771247527, -3.054813427408609, 1.1317309779961588),
                         (2.6693589702311393, -3.0744855630612933, 1.0857978392663963),
                         (2.700044011292814, -3.112531420644952, 1.0752649870521431),
                         (2.714721402768896, -3.1602932441954112, 1.0771041147237963),
                         (2.716493503702032, -3.210191631192234, 1.074456354651948),
                         (2.750341755272194, -3.237423546100998, 1.099209509522487),
                         (2.78672104715616, -3.2144730898499576, 1.1247011489340721),
                         (2.830341155105952, -3.193398490528954, 1.112325873344913),
                         (2.763200941348118, -3.285741004102575, 1.0989553722945844),
                         (2.724382293602151, -3.315962059950715, 1.1078884776112223),
                         (2.808479503965526, -3.306548112684321, 1.094842465807815),
                         (2.7322320224677776, -3.365221004884892, 1.1113436332817919),
                         (2.8330374899364834, -3.3117711429857386, 1.0516033020841637),
                         (2.7110835927058683, -3.410528214709433, 1.1113692559395059),
                         (2.8811776459215506, -3.3058233270994055, 1.0394728087753298),
                         (2.6142992236792146, -3.516144615671203, 1.1094313340177286),
                         (2.53965080968218, -3.5804482516311698, 1.09948918408272),
                         (2.580475560250588, -3.5515807120239202, 1.099419198004474),
                         (2.508567349283962, -3.617119678740361, 1.0857405505707054),
                         (2.660070552304407, -3.500903556679791, 1.122572703683976),
                         (2.7367554433821932, -3.45341701119576, 1.1101415675397324),
                         (2.7047597376634656, -3.491056561481638, 1.1024254111236527),
                         (2.922452293366636, -3.3332581680221893, 1.0460859004747856),
                         (2.4610701161782154, -3.6878408336881083, 1.0739919772185023),
                         (2.5057887068471425, -3.6668879592889345, 1.0818165988066293),
                         (2.970689913969611, -3.3208521972380156, 1.0504704012976498),
                         (2.7434754686449483, -3.5213366185802086, 1.0932487931912802),
                         (2.770586302552942, -3.56331073955373, 1.0950308855888193),
                         (2.807746130360402, -3.596188792916851, 1.0888518238176692),
                         (2.845511329845682, -3.623859096363801, 1.0712977925145557),
                         (2.849796538039291, -3.6735702969414747, 1.074527889705614),
                         (2.8711732553986398, -3.7181638965661867, 1.0671490803538703)]
        start_pos = (2.550544053326204, -2.9480692136221895, 1.0537656394334607)
        target_pos = (3.2999675712436187, -2.8501273913887983, 1.04)
        nearest_to_goal = (2.970689913969611, -3.3208521972380156, 1.0504704012976498)
        distance_to_goal = 0.5745565661664447

        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

        # Tree vertices - red
        tree_pcd = o3d.geometry.PointCloud()
        tree_pcd.points = o3d.utility.Vector3dVector(np.array(tree_vertices))
        tree_pcd.paint_uniform_color([1.0, 0.0, 0.0])

        # Start point - green
        start_pcd = o3d.geometry.PointCloud()
        start_pcd.points = o3d.utility.Vector3dVector(np.array([start_pos]))
        start_pcd.paint_uniform_color([0.0, 1.0, 0.0])

        # Goal point - blue
        goal_pcd = o3d.geometry.PointCloud()
        goal_pcd.points = o3d.utility.Vector3dVector(np.array([target_pos]))
        goal_pcd.paint_uniform_color([0.0, 0.0, 1.0])

        # Nearest point - yellow
        nearest_pcd = o3d.geometry.PointCloud()
        nearest_pcd.points = o3d.utility.Vector3dVector(np.array([nearest_to_goal]))
        nearest_pcd.paint_uniform_color([1.0, 1.0, 0.0])

        # Optional: color the obstacle point cloud
        if not obstacle_pcd.has_colors():
            obstacle_pcd.paint_uniform_color([0.7, 0.7, 0.7])

        # Visualize
        o3d.visualization.draw_geometries(
            [obstacle_pcd, tree_pcd, start_pcd, goal_pcd, nearest_pcd, coordinate_frame],
            window_name="RRT Failed Tree Visualization",
            width=1024,
            height=768,
        )
        exit()

    def get_intention_score(self, threshold=0.5, discrete=True):
        """
        Get the intention score of the task
        """
        return self.task.get_intention_score(self.physics, threshold, discrete)
    
    def get_task_progress(self):
        """
        Get the stage progress score of the task
        """
        return self.task.get_task_progress(self.physics)
    
    def get_expert_skill_sequence(self):
        """
        Get the expert demenstration of trajectory generation sequence
        """
        return self.task.get_expert_skill_sequence(self.physics)
    
    def save(self):
        """
        Save the task and env configuration
        """
        return self.task.save(self.physics)
    
    def get_robot_frame_position(self):
        return self.robot.get_base_position(self.physics)