import random
from functools import partial

import numpy as np

from RMMBench.tasks.base_task import PrimitiveTask
from RMMBench.tasks.config_manager import BenchTaskConfigManager
from RMMBench.utils.register import register
from RMMBench.utils.skill_lib import SkillLib


@register.add_config_manager("store_yogurt")
class StoreYogurtConfigManager(BenchTaskConfigManager):
    """
    Single-target storage task: put yogurt_1 into the cabinet counter_0 for storage.

    Config source: task_config.json["store_cabinet"]
    - seen_object:    ["yogurt_1"] (the original requirement says "yogurt", but "yogurt" is not registered
                       in name2class_xml — only "yogurt_1"/"yogurt_2" exist — so "yogurt_1" is used)
    - distractor:     ["hamburger", "scone"], num_objects=[2], all placed on the countertop as distractors
    - seen_container: ["counter_0"] (cabinet body, following the proven usage in test_1.py)
    - robocasa_scene: ONE_WALL_LARGE_1
    - fixture_surface: island_counter_island_group
    - destination_position: bottom
    - inherited: BenchTaskConfigManager (both target_entity / target_container are str)
    """

    def __init__(self, task_name, num_objects=[1], **kwargs):
        super().__init__(task_name, num_objects, **kwargs)

    def get_object_info(self, workregion_offset=0, workregion_y_set=0.1,
                        target_dim=(0.2, 0.3), grid_size=[4, 1]):
        """Override workspace parameters, following pick_snack_single.py."""
        super().get_object_info(workregion_offset, workregion_y_set,
                                target_dim=target_dim, grid_size=grid_size)

    def load_containers(self, target_container, offset=0.4, y_set=0.5,
                        direction="left", z_set=0.1, orient_offset=[0, 0, 0]):
        """
        Override container placement parameters.
        The counter_0 cabinet placement parameters (offset/z_set/orient_offset) follow the proven
        counter_0-as-container configuration in test_1.py.
        """
        if target_container is not None:
            if self.work_info and self.target_container:
                container_info = self.get_container_info_from_workregion(
                    self.work_info,
                    anchor=self.destination_position,
                    offset=offset,
                    y_set=y_set,
                    z_set=z_set,
                    direction=direction,
                )
                container_config = self.get_entity_config(
                    target_container,
                    position=[container_info["position"][0],
                              container_info["position"][1],
                              container_info["position"][2]],
                    orientation=np.array(container_info["orientation"]) + orient_offset,
                )
                self.config["task"]["components"].append(container_config)
            else:
                container_config = self.get_entity_config(target_container)
                self.config["task"]["components"].append(container_config)

    def get_condition_config(self, target_entity, target_container, **kwargs):
        """
        Success condition: the yogurt is inside the cabinet counter_0 and at rest.
        target_entity is a str and needs to be wrapped in a list.
        """
        conditions_config = dict(
            contain_v=dict(
                container=target_container,
                entities=[target_entity],
                vel_th=0.01,
            )
        )
        self.config["task"]["conditions"] = conditions_config

    def get_instruction(self, target_entity, target_container, **kwargs):
        instruction = ["Put the yogurt away into the cabinet."]
        self.config["task"]["instructions"] = instruction
        return self.config


@register.add_task("store_yogurt")
class StoreYogurtTask(PrimitiveTask):
    """
    Task flow: pick (yogurt_1) → place (counter_0) → end
    Objects that need to be fixed: counter (the cabinet is a component and must be attached to the arena)
    If cabinet-door interaction is needed, prepend the following to the skill sequence:
        partial(SkillLib.pick, body_name="counter_0/door_handle_main"),
        partial(SkillLib.pull),
    """

    def __init__(self, task_name, robot, **kwargs):
        self.attach_objects = [
            "counter",  # container
            # "yogurt",  # target object
            # "hamburger", "scone",  # distractors
        ]
        super().__init__(task_name, robot=robot, **kwargs)

    def build_from_config(self, eval=False, **kwargs):
        super().build_from_config(eval, **kwargs)
        self.reset_entities_positions()
        self.attach_entities_to_arena()

    def reset_entities_positions(self):
        """Height adaptation: adjust the z coordinate according to each object's own height to avoid initial penetration/clipping."""
        if self.config_manager.all_entities is not None:
            entities = self.config_manager.all_entities
            for k in entities:
                entity = self.entities.get(k)
                if entity is None:
                    continue
                height = entity.get_placement_height()
                if height == 0.0:
                    height+=0.02
                if "yogurt" in k:
                    height-=0.01

                entity.init_pos[2] += height

    def attach_entities_to_arena(self):
        for key, entity in self.entities.items():
            if any(obj in key for obj in self.attach_objects):
                entity.detach()
                self._arena.attach(entity)

    def get_expert_skill_sequence(self, physics):
        """
        Expert skill sequence: pick → place → end
        target_entity is a single str, passed directly to SkillLib.pick.
        """
        target_entity = self.config_manager.target_entity
        print("self.target_container", self.target_container)
        skill_sequence = [
            partial(SkillLib.pick, body_name="counter_0/door_handle_main"),
            partial(SkillLib.pull),
            #
            partial(SkillLib.pick, target_entity_name=target_entity),
            partial(SkillLib.place, target_container_name=self.target_container),
            partial(SkillLib.pick, body_name="counter_0/door_handle_main"),
            partial(SkillLib.push),
            partial(SkillLib.end),
        ]
        return skill_sequence
