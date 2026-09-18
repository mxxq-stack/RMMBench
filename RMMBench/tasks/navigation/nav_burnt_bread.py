from functools import partial

from RMMBench.tasks.navigation.base import CompositeNavigationTask
from RMMBench.tasks.config_manager import CompositeNavigationConfigManager
from RMMBench.utils.register import register
from RMMBench.utils.skill_lib import SkillLib


# =============================================================================
# Base classes for nav_burnt_bread tasks
# =============================================================================

class NavBurntBreadBaseConfigManager(CompositeNavigationConfigManager):
    """
    Base ConfigManager for nav_burnt_bread tasks.
    Based on fix_burnt_bread / fix_burnt_bread_0.

    Scene: ONE_WALL_LARGE (1 obstacle island_counter)

    Region 1 (manipulation preparation area): plate + bowl + bread + distractor; mid_container keeps bread_21 on the plate
    Region 2 (stove area): pan on stovetop_main_group

    fixed_fixture: ["stove_0"]

    Subclasses override: composite_configs, robocasa_scene, LAYOUT_CONFIG_TABLE,
    load_objects_for_layout, get_instruction, get_condition_config.
    """
    pass


class NavBurntBreadBaseTask(CompositeNavigationTask):
    """
    Base Task for nav_burnt_bread tasks.
    attach_objects includes all objects for easy debugging.
    """
    def build_from_config(self, eval=False, **kwargs):
        super().build_from_config(eval, **kwargs)
        self.reset_entities_positions()
        self.attach_entities_to_arena()

    def initialize_episode(self, physics, random_state):
        super().initialize_episode(physics, random_state)

    def attach_entities_to_arena(self):
        for key, entity in self.entities.items():
            if any(obj in key for obj in self.attach_objects):
                entity.detach()
                self._arena.attach(entity)

    def reset_entities_positions(self):
        if self.config_manager.composite_entities is not None:
            entities = self.config_manager.composite_entities
            for k in entities:
                entity = self.entities.get(f"{k}", None)
                if entity is None:
                    continue
                height = entity.get_placement_height()
                if height == 0:
                    height = 0.02
                entity.init_pos[2] += height


# =============================================================================
# nav_burnt_bread_0 (based on fix_burnt_bread_0)
# seen_object: ["bread_21", "bread_9"], distractor: ["bread_7", "cake_7"]
# mid_container: [{"plate_0": ["bread_21"]}]
# region1 seen_container: ["plate_0", "bowl_7"], region2 seen_container: ["pan_1"]
# =============================================================================

@register.add_config_manager("nav_burnt_bread_0")
class NavBurntBread0ConfigManager(NavBurntBreadBaseConfigManager):
    """
    Variant 0: Dual-area, robot faces region 1.
    Difficulty: tier_2 (obstacle, dual-area)
    Scene: ONE_WALL_LARGE_0
    Layout 0: island_counter_island_group (bottom) - pick bread, discard bowl
    Layout 1: stovetop_main_group (bottom) - cook on pan
    """

    def __init__(self, task_name, num_objects=[[1, 1], [0, 1]], **kwargs):
        super().__init__(task_name, num_objects, **kwargs)
        self.LAYOUT_CONFIG_TABLE = [
            {
                "workregion_offset": 0,
                "workregion_y_set": 0.1,
                "target_dim": (0.3, 0.25),
                "grid_size": [6, 6]
            },
            {
                "workregion_offset": -0.62,
                "workregion_y_set": 0.05,
                "target_dim": (0.25, 0.25),
                "grid_size": [2, 2],
            }
        ]
        self.CONTAINER_CONFIG_TABLE = [
            {
                "configs": [
                    {"offset": 0.18, "y_set": -0.03, "direction": "left", "z_set": 0.01},
                    {"offset": 0.25, "y_set": 0.1, "direction": "right", "z_set": 0.03, "orient_offset": [0, 0, 3.14]},
                ]
            },
            {
                "configs": [
                    {"offset": 0.18, "y_set": -0.03, "direction": "right", "z_set": 0.01},
                ]
            }
        ]

    def load_objects_for_layout(self, idx):
        super().load_objects_for_layout(idx)
        if idx == 0:
            pos = [1.075, -2.65, 0]
            self.target_object_info.append({"position": pos})
            self.target_bbox.append(self.center2bbox(pos))
        elif idx == 1:
            pos = [2.15, -1.68, 0]
            self.target_object_info.append({"position": pos})
            self.target_bbox.append(self.center2bbox(pos))

    def get_instruction(self, target_entity, target_container, **kwargs):
        instruction = ["Navigate to pick the fresh bread and discard the burnt one in the bowl, then go to the stove to pan-fry the bread."]
        self.config["task"]["instructions"] = instruction
        return self.config

    def get_condition_config(self, **kwargs):
        conditions_config = dict()
        conditions_config["asyn_sequence"] = [
            dict(
                contain_robot_pose=dict(
                    target_bbox=self.target_bbox[0],
                    robot="pandaomron"
                ),
                robot_orientation=dict(
                    target_orientation="bottom",
                    target_bbox=self.target_bbox[0],
                    robot="pandaomron"
                )
            ),
            dict(
                contain_robot_pose=dict(
                    target_bbox=self.target_bbox[1],
                    robot="pandaomron"
                ),
                robot_orientation=dict(
                    target_orientation="bottom",
                    target_bbox=self.target_bbox[1],
                    robot="pandaomron"
                )
            ),
        ]
        conditions_config["ordered_indices"] = [0, 1]

        self.target_object_info[0]["target_bbox"] = self.target_bbox[0]
        self.target_object_info[0]["target_orientation"] = "bottom"
        self.target_object_info[1]["target_bbox"] = self.target_bbox[1]
        self.target_object_info[1]["target_orientation"] = "bottom"

        self.config["task"]["conditions"] = conditions_config

    def reorder_target_object_info(self):
        pass


@register.add_task("nav_burnt_bread_0")
class NavBurntBread0Task(NavBurntBreadBaseTask):
    def __init__(self, task_name, robot, **kwargs):
        self.attach_objects = ["plate", "bowl", "pan", "stove", "bread", "cake", "donut"]
        super().__init__(task_name, robot=robot, **kwargs)

    def get_expert_skill_sequence(self, physics):
        skill_sequence = [
            partial(SkillLib.moveforward),
            partial(SkillLib.moveforward),
            partial(SkillLib.rotate_left),
            partial(SkillLib.moveforward),
            partial(SkillLib.moveforward),
            partial(SkillLib.rotate_right),
            partial(SkillLib.moveforward),
            partial(SkillLib.moveforward),
            partial(SkillLib.end),
        ]
        return skill_sequence


# -----------------------------------------------------------------------------
# nav_burnt_bread_1: V1 dual-area forced detour
# -----------------------------------------------------------------------------

@register.add_config_manager("nav_burnt_bread_1")
class NavBurntBread1ConfigManager(NavBurntBread0ConfigManager):
    """
    Variant 1: Dual-area, robot in front of obstacle, forced detour.
    Difficulty: tier_3 (obstacle, dual-area, forced detour)
    Scene: ONE_WALL_LARGE_1
    Layout 0: counter_2_main_group (bottom) - pick bread, discard bowl
    Layout 1: stovetop_main_group (bottom) - cook on pan
    """

    def __init__(self, task_name, num_objects=[[1, 1], [0, 1]], **kwargs):
        super().__init__(task_name, num_objects, **kwargs)
        self.composite_configs = [
            {
                "seen_object": ["bread_21", "bread_9"],
                "distractor": ["bread_7", "cake_7"],
                "mid_container": [{"plate_0": ["bread_21"]}],
                "seen_container": ["plate_0", "bowl_7"],
                "fixture_surface": "counter_2_main_group",
                "destination_position": "bottom"
            },
            {
                "seen_container": ["pan_1"],
                "fixture_surface": "stovetop_main_group",
                "destination_position": "bottom"
            }
        ]
        self.robocasa_scene = "ONE_WALL_LARGE_1"
        self.work_info = [None] * len(self.composite_configs)
        self.sampled_points = [None] * len(self.composite_configs)
        if not self.composite_configs:
            raise ValueError(f"Composite navigation task {task_name} requires 'composite_configs'")

        self.LAYOUT_CONFIG_TABLE = [
            {
                "workregion_offset": 0.2,
                "workregion_y_set": 0.02,
                "target_dim": (0.3, 0.25),
                "grid_size": [6, 6]
            },
            {
                "workregion_offset": -0.65,
                "workregion_y_set": 0.1,
                "target_dim": (0.3, 0.25),
                "grid_size": [6, 6]
            }
        ]

    def load_objects_for_layout(self, idx):
        super().load_objects_for_layout(idx)
        if idx == 0:
            pos = [0.75, -2.2, 0]
            self.target_object_info.append({"position": pos})
            self.target_bbox.append(self.center2bbox(pos))
        elif idx == 1:
            pos = [2.15, -1.68, 0]
            self.target_object_info.append({"position": pos})
            self.target_bbox.append(self.center2bbox(pos))

    def get_instruction(self, target_entity, target_container, **kwargs):
        instruction = ["Navigate to pick the fresh bread and discard the burnt one in the bowl, then go around the island to the stove to pan-fry the bread."]
        self.config["task"]["instructions"] = instruction
        return self.config


@register.add_task("nav_burnt_bread_1")
class NavBurntBread1Task(NavBurntBread0Task):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)
        self.robot_params = {
            "pos": [3, -4, 0.0],
            "euler": [0, 0, 0],
        }
        self.robot.set_init_pos(**self.robot_params)


# -----------------------------------------------------------------------------
# nav_burnt_bread_2: V2 single-area obstacle
# -----------------------------------------------------------------------------

@register.add_config_manager("nav_burnt_bread_2")
class NavBurntBread2ConfigManager(NavBurntBread0ConfigManager):
    """
    Variant 2: Single-area, robot in front of obstacle.
    Difficulty: tier_2 (obstacle, single-area, near obstacle)
    Scene: ONE_WALL_LARGE_2
    Layout 0: island_counter_island_group (left) - all operations in one area
    pan merged into seen_container with plate and bowl.
    """

    def __init__(self, task_name, num_objects=[[1, 1]], **kwargs):
        super().__init__(task_name, num_objects, **kwargs)
        self.composite_configs = [
            {
                "seen_object": ["bread_21", "bread_9"],
                "distractor": ["bread_7", "cake_7"],
                "mid_container": [{"plate_0": ["bread_21"]}],
                "seen_container": ["plate_0", "bowl_7", "pan_1"],
                "fixture_surface": "island_counter_island_group",
                "destination_position": "left"
            }
        ]
        self.robocasa_scene = "ONE_WALL_LARGE_2"
        self.work_info = [None] * len(self.composite_configs)
        self.sampled_points = [None] * len(self.composite_configs)
        if not self.composite_configs:
            raise ValueError(f"Composite navigation task {task_name} requires 'composite_configs'")

        self.LAYOUT_CONFIG_TABLE = [
            {
                "workregion_offset": -0.65,
                "workregion_y_set": 0.1,
                "target_dim": (0.3, 0.25),
                "grid_size": [6, 6]
            }
        ]
        self.CONTAINER_CONFIG_TABLE = [
            {
                "configs": [
                    {"offset": 0.18, "y_set": -0.03, "direction": "left", "z_set": 0.01},
                    {"offset": 0.25, "y_set": 0.1, "direction": "right", "z_set": 0.03, "orient_offset": [0, 0, 3.14]},
                    {"offset": 0.3, "y_set": -0.05, "direction": "right", "z_set": 0.01},
                ]
            }
        ]

    def load_objects_for_layout(self, idx):
        super().load_objects_for_layout(idx)
        if idx == 0:
            pos = [2.15, -1.68, 0]
            self.target_object_info.append({"position": pos})
            self.target_bbox.append(self.center2bbox(pos))

    def get_instruction(self, target_entity, target_container, **kwargs):
        instruction = ["Navigate around the island to pick the fresh bread, discard the burnt one in the bowl, and pan-fry the bread on the pan."]
        self.config["task"]["instructions"] = instruction
        return self.config

    def get_condition_config(self, **kwargs):
        conditions_config = dict()
        conditions_config["asyn_sequence"] = [
            dict(
                contain_robot_pose=dict(
                    target_bbox=self.target_bbox[0],
                    robot="pandaomron"
                ),
                robot_orientation=dict(
                    target_orientation="left",
                    target_bbox=self.target_bbox[0],
                    robot="pandaomron"
                )
            ),
        ]
        self.target_object_info[0]["target_bbox"] = self.target_bbox[0]
        self.target_object_info[0]["target_orientation"] = "left"
        self.config["task"]["conditions"] = conditions_config


@register.add_task("nav_burnt_bread_2")
class NavBurntBread2Task(NavBurntBread0Task):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)
        self.robot_params = {
            "pos": [3, -4, 0.0],
            "euler": [0, 0, 1.57],
        }
        self.robot.set_init_pos(**self.robot_params)


# =============================================================================
# nav_burnt_bread_3 (based on fix_burnt_bread_1)
# seen_object: ["bread_21", "bread_3"], distractor: ["bread_10", "donut_15"]
# mid_container: [{"plate_11": ["bread_21"]}]
# region1 seen_container: ["plate_11", "bowl_2"], region2 seen_container: ["pan_1"]
# =============================================================================

@register.add_config_manager("nav_burnt_bread_3")
class NavBurntBread3ConfigManager(NavBurntBread0ConfigManager):
    """
    Variant 0: Dual-area, robot faces region 1.
    Difficulty: tier_2 (obstacle, dual-area)
    Scene: ONE_WALL_LARGE_3
    Layout 0: island_counter_island_group (right) - pick bread, discard bowl
    Layout 1: stovetop_main_group (bottom) - cook on pan
    """

    def __init__(self, task_name, num_objects=[[1, 1], [0, 1]], **kwargs):
        super().__init__(task_name, num_objects, **kwargs)
        self.composite_configs = [
            {
                "seen_object": ["bread_21", "bread_3"],
                "distractor": ["bread_10", "donut_15"],
                "mid_container": [{"plate_11": ["bread_21"]}],
                "seen_container": ["plate_11", "bowl_2"],
                "fixture_surface": "island_counter_island_group",
                "destination_position": "right"
            },
            {
                "seen_container": ["pan_1"],
                "fixture_surface": "stovetop_main_group",
                "destination_position": "bottom"
            }
        ]
        self.robocasa_scene = "ONE_WALL_LARGE_3"
        self.work_info = [None] * len(self.composite_configs)
        self.sampled_points = [None] * len(self.composite_configs)
        if not self.composite_configs:
            raise ValueError(f"Composite navigation task {task_name} requires 'composite_configs'")

    def get_instruction(self, target_entity, target_container, **kwargs):
        instruction = ["Navigate to pick the fresh bread and discard the burnt one in the bowl, then go to the stove to pan-fry the bread."]
        self.config["task"]["instructions"] = instruction
        return self.config

    def get_condition_config(self, **kwargs):
        conditions_config = dict()
        conditions_config["asyn_sequence"] = [
            dict(
                contain_robot_pose=dict(
                    target_bbox=self.target_bbox[0],
                    robot="pandaomron"
                ),
                robot_orientation=dict(
                    target_orientation="right",
                    target_bbox=self.target_bbox[0],
                    robot="pandaomron"
                )
            ),
            dict(
                contain_robot_pose=dict(
                    target_bbox=self.target_bbox[1],
                    robot="pandaomron"
                ),
                robot_orientation=dict(
                    target_orientation="bottom",
                    target_bbox=self.target_bbox[1],
                    robot="pandaomron"
                )
            ),
        ]
        conditions_config["ordered_indices"] = [0, 1]

        self.target_object_info[0]["target_bbox"] = self.target_bbox[0]
        self.target_object_info[0]["target_orientation"] = "right"
        self.target_object_info[1]["target_bbox"] = self.target_bbox[1]
        self.target_object_info[1]["target_orientation"] = "bottom"

        self.config["task"]["conditions"] = conditions_config

    def reorder_target_object_info(self):
        pass


@register.add_task("nav_burnt_bread_3")
class NavBurntBread3Task(NavBurntBread0Task):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)

    def get_expert_skill_sequence(self, physics):
        skill_sequence = [
            partial(SkillLib.moveforward),
            partial(SkillLib.moveforward),
            partial(SkillLib.rotate_right),
            partial(SkillLib.moveforward),
            partial(SkillLib.moveforward),
            partial(SkillLib.rotate_left),
            partial(SkillLib.moveforward),
            partial(SkillLib.moveforward),
            partial(SkillLib.end),
        ]
        return skill_sequence


# -----------------------------------------------------------------------------
# nav_burnt_bread_4: V1 dual-area forced detour (based on fix_burnt_bread_1)
# -----------------------------------------------------------------------------

@register.add_config_manager("nav_burnt_bread_4")
class NavBurntBread4ConfigManager(NavBurntBread3ConfigManager):
    """
    Variant 1: Dual-area, robot in front of obstacle, forced detour.
    Difficulty: tier_3 (obstacle, dual-area, forced detour)
    Scene: ONE_WALL_LARGE_4
    Layout 0: island_counter_island_group (left) - pick bread, discard bowl
    Layout 1: stovetop_main_group (bottom) - cook on pan
    """

    def __init__(self, task_name, num_objects=[[1, 1], [0, 1]], **kwargs):
        super().__init__(task_name, num_objects, **kwargs)
        self.composite_configs = [
            {
                "seen_object": ["bread_21", "bread_3"],
                "distractor": ["bread_10", "donut_15"],
                "mid_container": [{"plate_11": ["bread_21"]}],
                "seen_container": ["plate_11", "bowl_2"],
                "fixture_surface": "island_counter_island_group",
                "destination_position": "left"
            },
            {
                "seen_container": ["pan_1"],
                "fixture_surface": "stovetop_main_group",
                "destination_position": "bottom"
            }
        ]
        self.robocasa_scene = "ONE_WALL_LARGE_4"
        self.work_info = [None] * len(self.composite_configs)
        self.sampled_points = [None] * len(self.composite_configs)
        if not self.composite_configs:
            raise ValueError(f"Composite navigation task {task_name} requires 'composite_configs'")

        self.LAYOUT_CONFIG_TABLE = [
            {
                "workregion_offset": 0.2,
                "workregion_y_set": 0.02,
                "target_dim": (0.3, 0.25),
                "grid_size": [6, 6]
            },
            {
                "workregion_offset": -0.65,
                "workregion_y_set": 0.1,
                "target_dim": (0.3, 0.25),
                "grid_size": [6, 6]
            }
        ]

    def load_objects_for_layout(self, idx):
        super().load_objects_for_layout(idx)
        if idx == 0:
            pos = [0.75, -2.2, 0]
            self.target_object_info.append({"position": pos})
            self.target_bbox.append(self.center2bbox(pos))
        elif idx == 1:
            pos = [2.15, -1.68, 0]
            self.target_object_info.append({"position": pos})
            self.target_bbox.append(self.center2bbox(pos))

    def get_instruction(self, target_entity, target_container, **kwargs):
        instruction = ["Navigate to pick the fresh bread and discard the burnt one in the bowl, then go around the island to the stove to pan-fry the bread."]
        self.config["task"]["instructions"] = instruction
        return self.config

    def get_condition_config(self, **kwargs):
        conditions_config = dict()
        conditions_config["asyn_sequence"] = [
            dict(
                contain_robot_pose=dict(
                    target_bbox=self.target_bbox[0],
                    robot="pandaomron"
                ),
                robot_orientation=dict(
                    target_orientation="left",
                    target_bbox=self.target_bbox[0],
                    robot="pandaomron"
                )
            ),
            dict(
                contain_robot_pose=dict(
                    target_bbox=self.target_bbox[1],
                    robot="pandaomron"
                ),
                robot_orientation=dict(
                    target_orientation="bottom",
                    target_bbox=self.target_bbox[1],
                    robot="pandaomron"
                )
            ),
        ]
        conditions_config["ordered_indices"] = [0, 1]

        self.target_object_info[0]["target_bbox"] = self.target_bbox[0]
        self.target_object_info[0]["target_orientation"] = "left"
        self.target_object_info[1]["target_bbox"] = self.target_bbox[1]
        self.target_object_info[1]["target_orientation"] = "bottom"

        self.config["task"]["conditions"] = conditions_config


@register.add_task("nav_burnt_bread_4")
class NavBurntBread4Task(NavBurntBread3Task):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)
        self.robot_params = {
            "pos": [3, -4, 0.0],
            "euler": [0, 0, 0],
        }
        self.robot.set_init_pos(**self.robot_params)


# -----------------------------------------------------------------------------
# nav_burnt_bread_5: V2 single-area obstacle (based on fix_burnt_bread_1)
# -----------------------------------------------------------------------------

@register.add_config_manager("nav_burnt_bread_5")
class NavBurntBread5ConfigManager(NavBurntBread3ConfigManager):
    """
    Variant 2: Single-area, robot in front of obstacle.
    Difficulty: tier_2 (obstacle, single-area, near obstacle)
    Scene: ONE_WALL_LARGE_5
    Layout 0: counter_2_main_group (bottom) - all operations in one area
    pan merged into seen_container with plate and bowl.
    """

    def __init__(self, task_name, num_objects=[[1, 1]], **kwargs):
        super().__init__(task_name, num_objects, **kwargs)
        self.composite_configs = [
            {
                "seen_object": ["bread_21", "bread_3"],
                "distractor": ["bread_10", "donut_15"],
                "mid_container": [{"plate_11": ["bread_21"]}],
                "seen_container": ["plate_11", "bowl_2", "pan_1"],
                "fixture_surface": "counter_2_main_group",
                "destination_position": "bottom"
            }
        ]
        self.robocasa_scene = "ONE_WALL_LARGE_5"
        self.work_info = [None] * len(self.composite_configs)
        self.sampled_points = [None] * len(self.composite_configs)
        if not self.composite_configs:
            raise ValueError(f"Composite navigation task {task_name} requires 'composite_configs'")

        self.LAYOUT_CONFIG_TABLE = [
            {
                "workregion_offset": -0.65,
                "workregion_y_set": 0.1,
                "target_dim": (0.3, 0.25),
                "grid_size": [6, 6]
            }
        ]
        self.CONTAINER_CONFIG_TABLE = [
            {
                "configs": [
                    {"offset": 0.18, "y_set": -0.03, "direction": "left", "z_set": 0.01},
                    {"offset": 0.25, "y_set": 0.1, "direction": "right", "z_set": 0.03, "orient_offset": [0, 0, 3.14]},
                    {"offset": 0.3, "y_set": -0.05, "direction": "right", "z_set": 0.01},
                ]
            }
        ]

    def load_objects_for_layout(self, idx):
        super().load_objects_for_layout(idx)
        if idx == 0:
            pos = [2.15, -1.68, 0]
            self.target_object_info.append({"position": pos})
            self.target_bbox.append(self.center2bbox(pos))

    def get_instruction(self, target_entity, target_container, **kwargs):
        instruction = ["Navigate to pick the fresh bread, discard the burnt one in the bowl, and pan-fry the bread on the pan."]
        self.config["task"]["instructions"] = instruction
        return self.config

    def get_condition_config(self, **kwargs):
        conditions_config = dict()
        conditions_config["asyn_sequence"] = [
            dict(
                contain_robot_pose=dict(
                    target_bbox=self.target_bbox[0],
                    robot="pandaomron"
                ),
                robot_orientation=dict(
                    target_orientation="bottom",
                    target_bbox=self.target_bbox[0],
                    robot="pandaomron"
                )
            ),
        ]
        self.target_object_info[0]["target_bbox"] = self.target_bbox[0]
        self.target_object_info[0]["target_orientation"] = "bottom"
        self.config["task"]["conditions"] = conditions_config


@register.add_task("nav_burnt_bread_5")
class NavBurntBread5Task(NavBurntBread3Task):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)
        self.robot_params = {
            "pos": [3, -4, 0.0],
            "euler": [0, 0, 1.57],
        }
        self.robot.set_init_pos(**self.robot_params)


# =============================================================================
# nav_burnt_bread_6 (based on fix_burnt_bread_2)
# seen_object: ["bread_21", "bread_9"], distractor: ["bread_11", "cake_9", "donut_13", "bread_20"]
# mid_container: [{"plate_12": ["bread_21"]}]
# region1 seen_container: ["plate_12", "bowl_5"], region2 seen_container: ["pan_2"]
# =============================================================================

@register.add_config_manager("nav_burnt_bread_6")
class NavBurntBread6ConfigManager(NavBurntBread0ConfigManager):
    """
    Variant 0: Dual-area, robot faces region 1.
    Difficulty: tier_2 (obstacle, dual-area)
    Scene: ONE_WALL_LARGE_6
    Layout 0: island_counter_island_group (bottom) - pick bread, discard bowl
    Layout 1: stovetop_main_group (bottom) - cook on pan
    """

    def __init__(self, task_name, num_objects=[[1, 1], [0, 1]], **kwargs):
        super().__init__(task_name, num_objects, **kwargs)
        self.composite_configs = [
            {
                "seen_object": ["bread_21", "bread_9"],
                "distractor": ["bread_11", "cake_9", "donut_13", "bread_20"],
                "mid_container": [{"plate_12": ["bread_21"]}],
                "seen_container": ["plate_12", "bowl_5"],
                "fixture_surface": "island_counter_island_group",
                "destination_position": "bottom"
            },
            {
                "seen_container": ["pan_2"],
                "fixture_surface": "stovetop_main_group",
                "destination_position": "bottom"
            }
        ]
        self.robocasa_scene = "ONE_WALL_LARGE_6"
        self.work_info = [None] * len(self.composite_configs)
        self.sampled_points = [None] * len(self.composite_configs)
        if not self.composite_configs:
            raise ValueError(f"Composite navigation task {task_name} requires 'composite_configs'")

    def get_instruction(self, target_entity, target_container, **kwargs):
        instruction = ["Navigate to pick the fresh bread and discard the burnt one in the bowl, then go to the stove to pan-fry the bread."]
        self.config["task"]["instructions"] = instruction
        return self.config


@register.add_task("nav_burnt_bread_6")
class NavBurntBread6Task(NavBurntBread0Task):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)

    def get_expert_skill_sequence(self, physics):
        skill_sequence = [
            partial(SkillLib.moveforward),
            partial(SkillLib.moveforward),
            partial(SkillLib.rotate_left),
            partial(SkillLib.moveforward),
            partial(SkillLib.moveforward),
            partial(SkillLib.rotate_right),
            partial(SkillLib.moveforward),
            partial(SkillLib.moveforward),
            partial(SkillLib.end),
        ]
        return skill_sequence


# -----------------------------------------------------------------------------
# nav_burnt_bread_7: V1 dual-area forced detour (based on fix_burnt_bread_2)
# -----------------------------------------------------------------------------

@register.add_config_manager("nav_burnt_bread_7")
class NavBurntBread7ConfigManager(NavBurntBread6ConfigManager):
    """
    Variant 1: Dual-area, robot in front of obstacle, forced detour.
    Difficulty: tier_3 (obstacle, dual-area, forced detour)
    Scene: ONE_WALL_LARGE_7
    Layout 0: counter_2_main_group (bottom) - pick bread, discard bowl
    Layout 1: stovetop_main_group (bottom) - cook on pan
    """

    def __init__(self, task_name, num_objects=[[1, 1], [0, 1]], **kwargs):
        super().__init__(task_name, num_objects, **kwargs)
        self.composite_configs = [
            {
                "seen_object": ["bread_21", "bread_9"],
                "distractor": ["bread_11", "cake_9", "donut_13", "bread_20"],
                "mid_container": [{"plate_12": ["bread_21"]}],
                "seen_container": ["plate_12", "bowl_5"],
                "fixture_surface": "counter_2_main_group",
                "destination_position": "bottom"
            },
            {
                "seen_container": ["pan_2"],
                "fixture_surface": "stovetop_main_group",
                "destination_position": "bottom"
            }
        ]
        self.robocasa_scene = "ONE_WALL_LARGE_7"
        self.work_info = [None] * len(self.composite_configs)
        self.sampled_points = [None] * len(self.composite_configs)
        if not self.composite_configs:
            raise ValueError(f"Composite navigation task {task_name} requires 'composite_configs'")

        self.LAYOUT_CONFIG_TABLE = [
            {
                "workregion_offset": 0.2,
                "workregion_y_set": 0.02,
                "target_dim": (0.3, 0.25),
                "grid_size": [6, 6]
            },
            {
                "workregion_offset": -0.65,
                "workregion_y_set": 0.1,
                "target_dim": (0.3, 0.25),
                "grid_size": [6, 6]
            }
        ]

    def load_objects_for_layout(self, idx):
        super().load_objects_for_layout(idx)
        if idx == 0:
            pos = [0.75, -2.2, 0]
            self.target_object_info.append({"position": pos})
            self.target_bbox.append(self.center2bbox(pos))
        elif idx == 1:
            pos = [2.15, -1.68, 0]
            self.target_object_info.append({"position": pos})
            self.target_bbox.append(self.center2bbox(pos))

    def get_instruction(self, target_entity, target_container, **kwargs):
        instruction = ["Navigate to pick the fresh bread and discard the burnt one in the bowl, then go around the island to the stove to pan-fry the bread."]
        self.config["task"]["instructions"] = instruction
        return self.config


@register.add_task("nav_burnt_bread_7")
class NavBurntBread7Task(NavBurntBread6Task):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)
        self.robot_params = {
            "pos": [3, -4, 0.0],
            "euler": [0, 0, 0],
        }
        self.robot.set_init_pos(**self.robot_params)


# -----------------------------------------------------------------------------
# nav_burnt_bread_8: V2 single-area obstacle (based on fix_burnt_bread_2)
# -----------------------------------------------------------------------------

@register.add_config_manager("nav_burnt_bread_8")
class NavBurntBread8ConfigManager(NavBurntBread6ConfigManager):
    """
    Variant 2: Single-area, robot in front of obstacle.
    Difficulty: tier_2 (obstacle, single-area, near obstacle)
    Scene: ONE_WALL_LARGE_8
    Layout 0: island_counter_island_group (right) - all operations in one area
    pan merged into seen_container with plate and bowl.
    """

    def __init__(self, task_name, num_objects=[[1, 1]], **kwargs):
        super().__init__(task_name, num_objects, **kwargs)
        self.composite_configs = [
            {
                "seen_object": ["bread_21", "bread_9"],
                "distractor": ["bread_11", "cake_9", "donut_13", "bread_20"],
                "mid_container": [{"plate_12": ["bread_21"]}],
                "seen_container": ["plate_12", "bowl_5", "pan_2"],
                "fixture_surface": "island_counter_island_group",
                "destination_position": "right"
            }
        ]
        self.robocasa_scene = "ONE_WALL_LARGE_8"
        self.work_info = [None] * len(self.composite_configs)
        self.sampled_points = [None] * len(self.composite_configs)
        if not self.composite_configs:
            raise ValueError(f"Composite navigation task {task_name} requires 'composite_configs'")

        self.LAYOUT_CONFIG_TABLE = [
            {
                "workregion_offset": -0.65,
                "workregion_y_set": 0.1,
                "target_dim": (0.3, 0.25),
                "grid_size": [6, 6]
            }
        ]
        self.CONTAINER_CONFIG_TABLE = [
            {
                "configs": [
                    {"offset": 0.18, "y_set": -0.03, "direction": "left", "z_set": 0.01},
                    {"offset": 0.25, "y_set": 0.1, "direction": "right", "z_set": 0.03, "orient_offset": [0, 0, 3.14]},
                    {"offset": 0.3, "y_set": -0.05, "direction": "right", "z_set": 0.01},
                ]
            }
        ]

    def load_objects_for_layout(self, idx):
        super().load_objects_for_layout(idx)
        if idx == 0:
            pos = [2.15, -1.68, 0]
            self.target_object_info.append({"position": pos})
            self.target_bbox.append(self.center2bbox(pos))

    def get_instruction(self, target_entity, target_container, **kwargs):
        instruction = ["Navigate around the island to pick the fresh bread, discard the burnt one in the bowl, and pan-fry the bread on the pan."]
        self.config["task"]["instructions"] = instruction
        return self.config

    def get_condition_config(self, **kwargs):
        conditions_config = dict()
        conditions_config["asyn_sequence"] = [
            dict(
                contain_robot_pose=dict(
                    target_bbox=self.target_bbox[0],
                    robot="pandaomron"
                ),
                robot_orientation=dict(
                    target_orientation="right",
                    target_bbox=self.target_bbox[0],
                    robot="pandaomron"
                )
            ),
        ]
        self.target_object_info[0]["target_bbox"] = self.target_bbox[0]
        self.target_object_info[0]["target_orientation"] = "right"
        self.config["task"]["conditions"] = conditions_config


@register.add_task("nav_burnt_bread_8")
class NavBurntBread8Task(NavBurntBread6Task):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)
        self.robot_params = {
            "pos": [3, -4, 0.0],
            "euler": [0, 0, 1.57],
        }
        self.robot.set_init_pos(**self.robot_params)
