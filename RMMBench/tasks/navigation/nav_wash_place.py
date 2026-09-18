from functools import partial

from RMMBench.tasks.navigation.base import CompositeNavigationTask
from RMMBench.tasks.config_manager import CompositeNavigationConfigManager
from RMMBench.utils.register import register
from RMMBench.utils.skill_lib import SkillLib


# =============================================================================
# Base classes for nav_wash_place tasks
# =============================================================================

class NavWashPlaceBaseConfigManager(CompositeNavigationConfigManager):
    """
    Base ConfigManager for nav_wash_place tasks.
    Subclasses override: composite_configs, robocasa_scene, LAYOUT_CONFIG_TABLE,
    load_objects_for_layout, get_instruction, get_condition_config.
    """
    pass


class NavWashPlaceBaseTask(CompositeNavigationTask):
    """
    Base Task for nav_wash_place tasks.
    Subclasses override: attach_objects, robot initial position.
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
# wash_place_carrot navigation variants
# =============================================================================

@register.add_config_manager("nav_wash_place_carrot_0")
class NavWashPlaceCarrot0ConfigManager(NavWashPlaceBaseConfigManager):
    """
    Variant 0: Dual-area, robot faces one of the areas.
    Difficulty: tier_2 (obstacle, dual-area)
    Scene: U_SHAPED_LARGE_6
    Layout 0: counter_1_main_group (bottom) - pick carrot
    Layout 1: sink_island_group (top) - wash and place
    """

    def __init__(self, task_name, num_objects=[[2, 1], [0, 1]], **kwargs):
        super().__init__(task_name, num_objects, **kwargs)
        self.LAYOUT_CONFIG_TABLE = [
            {
                "workregion_offset": 0,
                "workregion_y_set": 0.1,
                "target_dim": (0.3, 0.25),
                "grid_size": [6, 6]
            },
            {
                "workregion_offset": -0.35,
                "workregion_y_set": 0.1,
                "target_dim": (0.3, 0.25),
                "grid_size": [6, 6]
            }
        ]
        self.CONTAINER_CONFIG_TABLE = [
            {},  # Layout 0: no container
            {
                "configs": [
                    {"offset": -0.1, "y_set": 0.1, "direction": "right", "z_set": 0.01},
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
        instruction = ["Navigate to pick the carrot, then go to the sink to wash and place it on the plate."]
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
                    target_orientation="top",
                    target_bbox=self.target_bbox[1],
                    robot="pandaomron"
                )
            ),
        ]
        conditions_config["ordered_indices"] = [0, 1]

        self.target_object_info[0]["target_bbox"] = self.target_bbox[0]
        self.target_object_info[0]["target_orientation"] = "bottom"
        self.target_object_info[1]["target_bbox"] = self.target_bbox[1]
        self.target_object_info[1]["target_orientation"] = "top"

        self.config["task"]["conditions"] = conditions_config

    def reorder_target_object_info(self):
        pass


@register.add_task("nav_wash_place_carrot_0")
class NavWashPlaceCarrot0Task(NavWashPlaceBaseTask):
    def __init__(self, task_name, robot, **kwargs):
        self.attach_objects = ["plate", "sink", "carrot", "radish", "cucumber"]
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


@register.add_config_manager("nav_wash_place_carrot_1")
class NavWashPlaceCarrot1ConfigManager(NavWashPlaceCarrot0ConfigManager):
    """
    Variant 1: Dual-area, robot in front of obstacle, forced detour.
    Difficulty: tier_3 (obstacle, dual-area, forced detour)
    Scene: U_SHAPED_LARGE_3
    Layout 0: counter_1_left_group (right) - pick carrot
    Layout 1: sink_island_group (top) - wash and place
    """

    def __init__(self, task_name, num_objects=[[2, 1], [0, 1]], **kwargs):
        super().__init__(task_name, num_objects, **kwargs)
        self.composite_configs = [
            {
                "seen_object": ["carrot_1"],
                "distractor": ["radish_6", "cucumber_2"],
                "fixture_surface": "counter_1_left_group",
                "destination_position": "right"
            },
            {
                "seen_container": ["plate_0"],
                "fixture_surface": "sink_island_group",
                "destination_position": "top"
            }
        ]
        self.robocasa_scene = "U_SHAPED_LARGE_3"
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
        instruction = ["Navigate to pick the carrot from the left counter, then go to the sink to wash and place it on the plate."]
        self.config["task"]["instructions"] = instruction
        return self.config


@register.add_task("nav_wash_place_carrot_1")
class NavWashPlaceCarrot1Task(NavWashPlaceCarrot0Task):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)
        self.robot_params = {
            "pos": [3, -4, 0.0],
            "euler": [0, 0, 0],
        }
        self.robot.set_init_pos(**self.robot_params)


# =============================================================================
# wash_place_apple navigation variants
# =============================================================================

@register.add_config_manager("nav_wash_place_apple_0")
class NavWashPlaceApple0ConfigManager(NavWashPlaceBaseConfigManager):
    """
    Variant 0: Dual-area, robot faces one of the areas.
    Difficulty: tier_2 (obstacle, dual-area)
    Scene: U_SHAPED_LARGE_4
    Layout 0: counter_main_main_group (bottom) - pick apple
    Layout 1: sink_island_group (top) - wash and place
    """

    def __init__(self, task_name, num_objects=[[3, 1], [0, 1]], **kwargs):
        super().__init__(task_name, num_objects, **kwargs)
        self.LAYOUT_CONFIG_TABLE = [
            {
                "workregion_offset": 0,
                "workregion_y_set": 0.1,
                "target_dim": (0.3, 0.25),
                "grid_size": [6, 6]
            },
            {
                "workregion_offset": -0.35,
                "workregion_y_set": 0.1,
                "target_dim": (0.3, 0.25),
                "grid_size": [6, 6]
            }
        ]
        self.CONTAINER_CONFIG_TABLE = [
            {},  # Layout 0: no container
            {
                "configs": [
                    {"offset": -0.1, "y_set": 0.1, "direction": "right", "z_set": 0.01},
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
        instruction = ["Navigate to pick the apple, then go to the sink to wash and place it on the plate."]
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
                    target_orientation="top",
                    target_bbox=self.target_bbox[1],
                    robot="pandaomron"
                )
            ),
        ]
        conditions_config["ordered_indices"] = [0, 1]

        self.target_object_info[0]["target_bbox"] = self.target_bbox[0]
        self.target_object_info[0]["target_orientation"] = "bottom"
        self.target_object_info[1]["target_bbox"] = self.target_bbox[1]
        self.target_object_info[1]["target_orientation"] = "top"

        self.config["task"]["conditions"] = conditions_config

    def reorder_target_object_info(self):
        pass


@register.add_task("nav_wash_place_apple_0")
class NavWashPlaceApple0Task(NavWashPlaceBaseTask):
    def __init__(self, task_name, robot, **kwargs):
        self.attach_objects = ["plate", "sink", "apple", "pear", "peach", "pomegranate"]
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


@register.add_config_manager("nav_wash_place_apple_1")
class NavWashPlaceApple1ConfigManager(NavWashPlaceApple0ConfigManager):
    """
    Variant 1: Dual-area, robot in front of obstacle, forced detour.
    Difficulty: tier_3 (obstacle, dual-area, forced detour)
    Scene: U_SHAPED_LARGE_5
    Layout 0: counter_1_right_group (left) - pick apple
    Layout 1: sink_island_group (top) - wash and place
    """

    def __init__(self, task_name, num_objects=[[3, 1], [0, 1]], **kwargs):
        super().__init__(task_name, num_objects, **kwargs)
        self.composite_configs = [
            {
                "seen_object": ["apple_22"],
                "distractor": ["pear_10", "peach_3", "pomegranate_1"],
                "fixture_surface": "counter_1_right_group",
                "destination_position": "left"
            },
            {
                "seen_container": ["plate_7"],
                "fixture_surface": "sink_island_group",
                "destination_position": "top"
            }
        ]
        self.robocasa_scene = "U_SHAPED_LARGE_5"
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
        instruction = ["Navigate to pick the apple from the right counter, then go to the sink to wash and place it on the plate."]
        self.config["task"]["instructions"] = instruction
        return self.config


@register.add_task("nav_wash_place_apple_1")
class NavWashPlaceApple1Task(NavWashPlaceApple0Task):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)
        self.robot_params = {
            "pos": [3, -4, 0.0],
            "euler": [0, 0, 0],
        }
        self.robot.set_init_pos(**self.robot_params)


# =============================================================================
# wash_place_cucumber navigation variants
# =============================================================================

@register.add_config_manager("nav_wash_place_cucumber_0")
class NavWashPlaceCucumber0ConfigManager(NavWashPlaceBaseConfigManager):
    """
    Variant 0: Dual-area, robot faces one of the areas.
    Difficulty: tier_2 (obstacle, dual-area)
    Scene: U_SHAPED_LARGE_7
    Layout 0: island_island_group (bottom) - pick cucumber
    Layout 1: sink_island_group (top) - wash and place
    """

    def __init__(self, task_name, num_objects=[[4, 1], [0, 1]], **kwargs):
        super().__init__(task_name, num_objects, **kwargs)
        self.LAYOUT_CONFIG_TABLE = [
            {
                "workregion_offset": 0.7,
                "workregion_y_set": 0.1,
                "target_dim": (0.3, 0.25),
                "grid_size": [6, 6]
            },
            {
                "workregion_offset": -0.35,
                "workregion_y_set": 0.1,
                "target_dim": (0.3, 0.25),
                "grid_size": [6, 6]
            }
        ]
        self.CONTAINER_CONFIG_TABLE = [
            {},  # Layout 0: no container
            {
                "configs": [
                    {"offset": -0.1, "y_set": 0.1, "direction": "right", "z_set": 0.01},
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
        instruction = ["Navigate to pick the cucumber, then go to the sink to wash and place it in the tray."]
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
                    target_orientation="top",
                    target_bbox=self.target_bbox[1],
                    robot="pandaomron"
                )
            ),
        ]
        conditions_config["ordered_indices"] = [0, 1]

        self.target_object_info[0]["target_bbox"] = self.target_bbox[0]
        self.target_object_info[0]["target_orientation"] = "bottom"
        self.target_object_info[1]["target_bbox"] = self.target_bbox[1]
        self.target_object_info[1]["target_orientation"] = "top"

        self.config["task"]["conditions"] = conditions_config

    def reorder_target_object_info(self):
        pass


@register.add_task("nav_wash_place_cucumber_0")
class NavWashPlaceCucumber0Task(NavWashPlaceBaseTask):
    def __init__(self, task_name, robot, **kwargs):
        self.attach_objects = ["tray", "sink", "cucumber", "zucchini", "eggplant", "carrot", "bell_pepper"]
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


@register.add_config_manager("nav_wash_place_cucumber_1")
class NavWashPlaceCucumber1ConfigManager(NavWashPlaceCucumber0ConfigManager):
    """
    Variant 1: Dual-area, robot in front of obstacle, forced detour.
    Difficulty: tier_3 (obstacle, dual-area, forced detour)
    Scene: U_SHAPED_LARGE_9
    Layout 0: counter_main_main_group (bottom) - pick cucumber
    Layout 1: sink_island_group (top) - wash and place
    """

    def __init__(self, task_name, num_objects=[[4, 1], [0, 1]], **kwargs):
        super().__init__(task_name, num_objects, **kwargs)
        self.composite_configs = [
            {
                "seen_object": ["cucumber_2"],
                "distractor": ["zucchini_7", "eggplant_1", "carrot_4", "bell_pepper_3"],
                "fixture_surface": "counter_main_main_group",
                "destination_position": "bottom"
            },
            {
                "seen_container": ["tray_7"],
                "fixture_surface": "sink_island_group",
                "destination_position": "top"
            }
        ]
        self.robocasa_scene = "U_SHAPED_LARGE_9"
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
        instruction = ["Navigate to pick the cucumber from the main counter, then go to the sink to wash and place it in the tray."]
        self.config["task"]["instructions"] = instruction
        return self.config


@register.add_task("nav_wash_place_cucumber_1")
class NavWashPlaceCucumber1Task(NavWashPlaceCucumber0Task):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)
        self.robot_params = {
            "pos": [3, -4, 0.0],
            "euler": [0, 0, 0],
        }
        self.robot.set_init_pos(**self.robot_params)


# =============================================================================
# wash_place_carrot navigation variants (continued)
# =============================================================================

@register.add_config_manager("nav_wash_place_carrot_1")
class NavWashPlaceCarrot1ConfigManager(NavWashPlaceCarrot0ConfigManager):
    """
    Variant 1: Dual-area, robot in front of obstacle, forced detour.
    Difficulty: tier_3 (obstacle, dual-area, forced detour)
    Scene: U_SHAPED_LARGE_3
    Layout 0: counter_1_left_group (right) - pick carrot
    Layout 1: sink_island_group (top) - wash and place
    """

    def __init__(self, task_name, num_objects=[[2, 1], [0, 1]], **kwargs):
        super().__init__(task_name, num_objects, **kwargs)
        self.composite_configs = [
            {
                "seen_object": ["carrot_1"],
                "distractor": ["radish_6", "cucumber_2"],
                "fixture_surface": "counter_1_left_group",
                "destination_position": "right"
            },
            {
                "seen_container": ["plate_0"],
                "fixture_surface": "sink_island_group",
                "destination_position": "top"
            }
        ]
        self.robocasa_scene = "U_SHAPED_LARGE_3"
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
        instruction = ["Navigate to pick the carrot from the left counter, then go to the sink to wash and place it on the plate."]
        self.config["task"]["instructions"] = instruction
        return self.config


@register.add_task("nav_wash_place_carrot_1")
class NavWashPlaceCarrot1Task(NavWashPlaceCarrot0Task):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)
        self.robot_params = {
            "pos": [3, -4,0.0],  # in front of the fridge
            "euler": [0, 0, 0],
        }
        self.robot.set_init_pos(**self.robot_params)


@register.add_config_manager("nav_wash_place_carrot_2")
class NavWashPlaceCarrot2ConfigManager(NavWashPlaceCarrot0ConfigManager):
    """
    Variant 2: Single-area, robot in front of obstacle.
    Difficulty: tier_2 (obstacle, single-area, near obstacle)
    Scene: U_SHAPED_LARGE_2
    Layout 0: sink_island_group (top) - wash and place carrot
    """

    def __init__(self, task_name, num_objects=[[2, 1]], **kwargs):
        super().__init__(task_name, num_objects, **kwargs)
        self.composite_configs = [
            {
                "seen_object": ["carrot_1"],
                "distractor": ["radish_6", "cucumber_2"],
                "seen_container": ["plate_0"],
                "fixture_surface": "sink_island_group",
                "destination_position": "top"
            }
        ]
        self.robocasa_scene = "U_SHAPED_LARGE_2"
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
                    {"offset": 0.16, "y_set": -0.05, "direction": "right", "z_set": 0}
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
        instruction = ["Navigate to the sink to wash the carrot and place it on the plate."]
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
                    target_orientation="top",
                    target_bbox=self.target_bbox[0],
                    robot="pandaomron"
                )
            ),
        ]
        self.target_object_info[0]["target_bbox"] = self.target_bbox[0]
        self.target_object_info[0]["target_orientation"] = "top"
        self.config["task"]["conditions"] = conditions_config


@register.add_task("nav_wash_place_carrot_2")
class NavWashPlaceCarrot2Task(NavWashPlaceCarrot0Task):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)
        self.robot_params = {
            "pos": [3, -4,0.0],
            "euler": [0, 0, 1.57],
        }
        self.robot.set_init_pos(**self.robot_params)


@register.add_config_manager("nav_wash_place_carrot_3")
class NavWashPlaceCarrot3ConfigManager(NavWashPlaceCarrot2ConfigManager):
    """
    Variant 3: Single-area, obstacle present but robot starts in open space.
    Difficulty: tier_2 (obstacle, single-area, open start)
    Scene: U_SHAPED_LARGE_8
    Layout 0: sink_island_group (top) - wash and place carrot
    """

    def __init__(self, task_name, num_objects=[[2, 1]], **kwargs):
        super().__init__(task_name, num_objects, **kwargs)
        self.composite_configs = [
            {
                "seen_object": ["carrot_1"],
                "distractor": ["radish_6", "cucumber_2"],
                "seen_container": ["plate_0"],
                "fixture_surface": "sink_island_group",
                "destination_position": "top"
            }
        ]
        self.robocasa_scene = "U_SHAPED_LARGE_1"

        self.work_info = [None] * len(self.composite_configs)
        self.sampled_points = [None] * len(self.composite_configs)
        if not self.composite_configs:
            raise ValueError(f"Composite navigation task {task_name} requires 'composite_configs'")

    def get_instruction(self, target_entity, target_container, **kwargs):
        instruction = ["Navigate to the sink to wash the carrot and place it on the plate."]
        self.config["task"]["instructions"] = instruction
        return self.config


@register.add_task("nav_wash_place_carrot_3")
class NavWashPlaceCarrot3Task(NavWashPlaceCarrot2Task):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)
        self.robot_params = {
            "pos": [5, -4,0.0],  # in front of the fridge
            "euler": [0, 0, 1.57],
        }
        self.robot.set_init_pos(**self.robot_params)
