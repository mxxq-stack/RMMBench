from functools import partial

from RMMBench.tasks.navigation.base import CompositeNavigationTask
from RMMBench.tasks.config_manager import CompositeNavigationConfigManager
from RMMBench.utils.register import register
from RMMBench.utils.skill_lib import SkillLib


# =============================================================================
# Base classes for nav_pick_snacks tasks
# =============================================================================

class NavPickSnacksBaseConfigManager(CompositeNavigationConfigManager):
    """
    Base ConfigManager for nav_pick_snacks tasks.
    Based on pick_snack_single (pick_chocolate / pick_chip).
    Scene: G_SHAPED_LARGE (obstacle optional).

    Subclasses override: composite_configs, robocasa_scene, LAYOUT_CONFIG_TABLE,
    load_objects_for_layout, get_instruction, get_condition_config.
    """
    pass


class NavPickSnacksBaseTask(CompositeNavigationTask):
    """
    Base Task for nav_pick_snacks tasks.
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
# pick_chocolate navigation variants (G_SHAPED_LARGE)
# =============================================================================

@register.add_config_manager("nav_pick_chocolate_0")
class NavPickChocolate0ConfigManager(NavPickSnacksBaseConfigManager):
    """
    Variant 0: Dual-area, robot faces one of the areas.
    Difficulty: tier_2 (obstacle, dual-area)
    Scene: G_SHAPED_LARGE_6
    Layout 0: counter_main_main_group (bottom) - pick chocolate
    Layout 1: counter_1_left_group (right) - place on plate
    """

    def __init__(self, task_name, num_objects=[[4, 1], [0, 1]], **kwargs):
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
        instruction = ["Navigate to pick the chocolate, then go to the counter to place it on the plate."]
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
                    target_orientation="right",
                    target_bbox=self.target_bbox[1],
                    robot="pandaomron"
                )
            ),
        ]
        conditions_config["ordered_indices"] = [0, 1]

        self.target_object_info[0]["target_bbox"] = self.target_bbox[0]
        self.target_object_info[0]["target_orientation"] = "bottom"
        self.target_object_info[1]["target_bbox"] = self.target_bbox[1]
        self.target_object_info[1]["target_orientation"] = "right"

        self.config["task"]["conditions"] = conditions_config

    def reorder_target_object_info(self):
        pass


@register.add_task("nav_pick_chocolate_0")
class NavPickChocolate0Task(NavPickSnacksBaseTask):
    def __init__(self, task_name, robot, **kwargs):
        self.attach_objects = [
            "plate", "chocolate", "bagged_food", "bar", "boxed_food", "chips",
        ]
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


@register.add_config_manager("nav_pick_chocolate_1")
class NavPickChocolate1ConfigManager(NavPickChocolate0ConfigManager):
    """
    Variant 1: Dual-area, robot in front of obstacle, forced detour.
    Difficulty: tier_3 (obstacle, dual-area, forced detour)
    Scene: G_SHAPED_LARGE_3
    Layout 0: counter_main_main_group (bottom) - pick chocolate
    Layout 1: counter_1_front_group (right) - place on plate
    """

    def __init__(self, task_name, num_objects=[[4, 1], [0, 1]], **kwargs):
        super().__init__(task_name, num_objects, **kwargs)
        self.composite_configs = [
            {
                "seen_object": ["chocolate_2", "chocolate_5", "chocolate_6"],
                "distractor": ["bagged_food_8", "bar_10", "boxed_food_0", "chips_7"],
                "fixture_surface": "counter_main_main_group",
                "destination_position": "bottom"
            },
            {
                "seen_container": ["plate_2"],
                "fixture_surface": "counter_1_front_group",
                "destination_position": "right"
            }
        ]
        self.robocasa_scene = "G_SHAPED_LARGE_3"
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
                "workregion_offset": 0,
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
        instruction = ["Navigate to pick the chocolate from the main counter, then go around the island to place it on the plate."]
        self.config["task"]["instructions"] = instruction
        return self.config


@register.add_task("nav_pick_chocolate_1")
class NavPickChocolate1Task(NavPickChocolate0Task):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)
        self.robot_params = {
            "pos": [2, -5, 0.0],
            "euler": [0, 0, 0],
        }
        self.robot.set_init_pos(**self.robot_params)


@register.add_config_manager("nav_pick_chocolate_2")
class NavPickChocolate2ConfigManager(NavPickChocolate0ConfigManager):
    """
    Variant 2: Single-area, robot in front of obstacle.
    Difficulty: tier_2 (obstacle, single-area, near obstacle)
    Scene: G_SHAPED_LARGE_2
    Layout 0: counter_1_front_group (right) - pick chocolate and place on plate
    """

    def __init__(self, task_name, num_objects=[[4, 1]], **kwargs):
        super().__init__(task_name, num_objects, **kwargs)
        self.composite_configs = [
            {
                "seen_object": ["chocolate_2", "chocolate_5", "chocolate_6"],
                "distractor": ["bagged_food_8", "bar_10", "boxed_food_0", "chips_7"],
                "seen_container": ["plate_2"],
                "fixture_surface": "counter_1_front_group",
                "destination_position": "right"
            }
        ]
        self.robocasa_scene = "G_SHAPED_LARGE_2"
        self.work_info = [None] * len(self.composite_configs)
        self.sampled_points = [None] * len(self.composite_configs)
        if not self.composite_configs:
            raise ValueError(f"Composite navigation task {task_name} requires 'composite_configs'")

        self.LAYOUT_CONFIG_TABLE = [
            {
                "workregion_offset": -0,
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
        instruction = ["Navigate to the counter to pick the chocolate and place it on the plate."]
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


@register.add_task("nav_pick_chocolate_2")
class NavPickChocolate2Task(NavPickChocolate0Task):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)
        self.robot_params = {
            "pos": [3, -1, 0.0],
            "euler": [0, 0, 1.57],
        }
        self.robot.set_init_pos(**self.robot_params)


@register.add_config_manager("nav_pick_chocolate_3")
class NavPickChocolate3ConfigManager(NavPickChocolate2ConfigManager):
    """
    Variant 3: Single-area, obstacle present but robot starts in open space.
    Difficulty: tier_2 (obstacle, single-area, open start)
    Scene: G_SHAPED_LARGE_8
    Layout 0: counter_1_front_group (right) - pick chocolate and place on plate
    """

    def __init__(self, task_name, num_objects=[[4, 1]], **kwargs):
        super().__init__(task_name, num_objects, **kwargs)
        self.composite_configs = [
            {
                "seen_object": ["chocolate_2", "chocolate_5", "chocolate_6"],
                "distractor": ["bagged_food_8", "bar_10", "boxed_food_0", "chips_7"],
                "seen_container": ["plate_2"],
                "fixture_surface": "counter_1_front_group",
                "destination_position": "right"
            }
        ]
        self.robocasa_scene = "G_SHAPED_LARGE_8"

        self.work_info = [None] * len(self.composite_configs)
        self.sampled_points = [None] * len(self.composite_configs)
        if not self.composite_configs:
            raise ValueError(f"Composite navigation task {task_name} requires 'composite_configs'")

    def get_instruction(self, target_entity, target_container, **kwargs):
        instruction = ["Navigate to the counter to pick the chocolate and place it on the plate."]
        self.config["task"]["instructions"] = instruction
        return self.config


@register.add_task("nav_pick_chocolate_3")
class NavPickChocolate3Task(NavPickChocolate2Task):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)
        self.robot_params = {
            "pos": [3, -1, 0.0],
            "euler": [0, 0, 1.57],
        }
        self.robot.set_init_pos(**self.robot_params)


# =============================================================================
# pick_chip navigation variants (G_SHAPED_LARGE)
# =============================================================================

@register.add_config_manager("nav_pick_chip_0")
class NavPickChip0ConfigManager(NavPickSnacksBaseConfigManager):
    """
    Variant 0: Dual-area, robot faces one of the areas.
    Difficulty: tier_2 (obstacle, dual-area)
    Scene: G_SHAPED_LARGE_5
    Layout 0: counter_main_main_group (bottom) - pick chips
    Layout 1: counter_1_left_group (right) - place on plate
    """

    def __init__(self, task_name, num_objects=[[4, 1], [0, 1]], **kwargs):
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
        instruction = ["Navigate to pick the chips, then go to the counter to place them on the plate."]
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
                    target_orientation="right",
                    target_bbox=self.target_bbox[1],
                    robot="pandaomron"
                )
            ),
        ]
        conditions_config["ordered_indices"] = [0, 1]

        self.target_object_info[0]["target_bbox"] = self.target_bbox[0]
        self.target_object_info[0]["target_orientation"] = "bottom"
        self.target_object_info[1]["target_bbox"] = self.target_bbox[1]
        self.target_object_info[1]["target_orientation"] = "right"

        self.config["task"]["conditions"] = conditions_config

    def reorder_target_object_info(self):
        pass


@register.add_task("nav_pick_chip_0")
class NavPickChip0Task(NavPickSnacksBaseTask):
    def __init__(self, task_name, robot, **kwargs):
        self.attach_objects = [
            "plate", "chips", "bagged_food", "bar", "boxed_food", "chocolate",
        ]
        super().__init__(task_name, robot=robot, **kwargs)
        self.robot_params = {
            "pos": [3, -3, 0.0],
            "euler": [0, 0, 1.57],
        }
        self.robot.set_init_pos(**self.robot_params)

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


@register.add_config_manager("nav_pick_chip_1")
class NavPickChip1ConfigManager(NavPickChip0ConfigManager):
    """
    Variant 1: Dual-area, robot in front of obstacle, forced detour.
    Difficulty: tier_3 (obstacle, dual-area, forced detour)
    Scene: G_SHAPED_LARGE_1
    Layout 0: counter_main_main_group (bottom) - pick chips
    Layout 1: counter_1_front_group (right) - place on plate
    """

    def __init__(self, task_name, num_objects=[[4, 1], [0, 1]], **kwargs):
        super().__init__(task_name, num_objects, **kwargs)
        self.composite_configs = [
            {
                "seen_object": ["chips_13", "chips_14", "chips_6"],
                "distractor": ["bagged_food_8", "bar_10", "boxed_food_0", "chocolate_5"],
                "fixture_surface": "counter_main_main_group",
                "destination_position": "bottom"
            },
            {
                "seen_container": ["plate_2"],
                "fixture_surface": "counter_1_front_group",
                "destination_position": "right"
            }
        ]
        self.robocasa_scene = "G_SHAPED_LARGE_1"
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
                "workregion_offset": 0,
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
        instruction = ["Navigate to pick the chips from the main counter, then go around the island to place them on the plate."]
        self.config["task"]["instructions"] = instruction
        return self.config


@register.add_task("nav_pick_chip_1")
class NavPickChip1Task(NavPickChip0Task):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)
        self.robot_params = {
            "pos": [3, -1, 0.0],
            "euler": [0, 0, 1.57],
        }
        self.robot.set_init_pos(**self.robot_params)


@register.add_config_manager("nav_pick_chip_2")
class NavPickChip2ConfigManager(NavPickChip0ConfigManager):
    """
    Variant 2: Single-area, robot in front of obstacle.
    Difficulty: tier_2 (obstacle, single-area, near obstacle)
    Scene: G_SHAPED_LARGE_7
    Layout 0: counter_1_front_group (right) - pick chips and place on plate
    """

    def __init__(self, task_name, num_objects=[[4, 1]], **kwargs):
        super().__init__(task_name, num_objects, **kwargs)
        self.composite_configs = [
            {
                "seen_object": ["chips_13", "chips_14", "chips_6"],
                "distractor": ["bagged_food_8", "bar_10", "boxed_food_0", "chocolate_5"],
                "seen_container": ["plate_2"],
                "fixture_surface": "counter_1_front_group",
                "destination_position": "right"
            }
        ]
        self.robocasa_scene = "G_SHAPED_LARGE_7"
        self.work_info = [None] * len(self.composite_configs)
        self.sampled_points = [None] * len(self.composite_configs)
        if not self.composite_configs:
            raise ValueError(f"Composite navigation task {task_name} requires 'composite_configs'")

        self.LAYOUT_CONFIG_TABLE = [
            {
                "workregion_offset": 0,
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
        instruction = ["Navigate to the counter to pick the chips and place them on the plate."]
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


@register.add_task("nav_pick_chip_2")
class NavPickChip2Task(NavPickChip0Task):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)
        self.robot_params = {
            "pos": [3, -1, 0.0],
            "euler": [0, 0, 1.57],
        }
        self.robot.set_init_pos(**self.robot_params)


@register.add_config_manager("nav_pick_chip_3")
class NavPickChip3ConfigManager(NavPickChip2ConfigManager):
    """
    Variant 3: Single-area, obstacle present but robot starts in open space.
    Difficulty: tier_2 (obstacle, single-area, open start)
    Scene: G_SHAPED_LARGE_9
    Layout 0: counter_1_front_group (right) - pick chips and place on plate
    """

    def __init__(self, task_name, num_objects=[[4, 1]], **kwargs):
        super().__init__(task_name, num_objects, **kwargs)
        self.composite_configs = [
            {
                "seen_object": ["chips_13", "chips_14", "chips_6"],
                "distractor": ["bagged_food_8", "bar_10", "boxed_food_0", "chocolate_5"],
                "seen_container": ["plate_2"],
                "fixture_surface": "counter_1_front_group",
                "destination_position": "right"
            }
        ]
        self.robocasa_scene = "G_SHAPED_LARGE_9"

        self.work_info = [None] * len(self.composite_configs)
        self.sampled_points = [None] * len(self.composite_configs)
        if not self.composite_configs:
            raise ValueError(f"Composite navigation task {task_name} requires 'composite_configs'")

    def get_instruction(self, target_entity, target_container, **kwargs):
        instruction = ["Navigate to the counter to pick the chips and place them on the plate."]
        self.config["task"]["instructions"] = instruction
        return self.config


@register.add_task("nav_pick_chip_3")
class NavPickChip3Task(NavPickChip2Task):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)
        self.robot_params = {
            "pos": [5, -4, 0.0],
            "euler": [0, 0, 1.57],
        }
        self.robot.set_init_pos(**self.robot_params)
