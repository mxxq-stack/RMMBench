import os


def generate_task_template(task_name, output_path,task_class_manager):
    # Convert snake_case (cook_steak) to UpperCamelCase (CookSteak)
    class_name_suffix = "".join([word.capitalize() for word in task_name.split("_")])

    # Template string
    template = f'''import random
from asyncio import wait_for

from RMMBench.tasks.base_task import PrimitiveTask
from RMMBench.tasks.config_manager import {task_class_manager}
from RMMBench.utils.register import register
from RMMBench.utils.utils import flatten_list, grid_sample
import re

from functools import partial
import numpy as np
from RMMBench.utils.skill_lib import SkillLib

@register.add_config_manager("{task_name}")
class {class_name_suffix}ConfigManager({task_class_manager}):
    def __init__(self,
                 task_name,
                 num_objects=[1],
                 **kwargs):
        super().__init__(task_name, num_objects, **kwargs)
        self.get_object_info

    def load_containers(self, target_container):
        # TODO: Implement container loading logic
        pass

    def load_objects(self, target_entity):
        # TODO: Implement object loading logic
        pass

    def load_init_containers(self, init_container):
        # TODO: Implement initial containers logic
        pass

    def get_instruction(self, target_entity, target_container, **kwargs):
        super().get_instruction(target_entity, target_container, **kwargs)
        # TODO: Implement instruction generation

    def get_condition_config(self, target_entity, target_container, **kwargs):
        """
        self.check_entity_xquat_class = ["Toy","Book","Drink","Mug","CommonGraspedEntity"]
        """
        conditions_config = dict(
            contain_v=dict(
                container=f"{{target_container}}",
                entities=[f"{{target_entity}}"],
                vel_th=0.01,
            )
        )

        self.config["task"]["conditions"] = conditions_config


@register.add_task("{task_name}")
class {class_name_suffix}Task(PrimitiveTask):
    def __init__(self, task_name, robot, **kwargs):
        # Define the objects that need to be attached to the arena for this task
        self.attach_objects = []  # fill in the names of the objects to attach
        super().__init__(task_name, robot=robot, **kwargs)

    def build_from_config(self, eval=False, **kwargs):
        super().build_from_config(eval, **kwargs)
        self.attach_entities_to_arena()

    def initialize_episode(self, physics, random_state):
        super().initialize_episode(physics, random_state)
        for key, entity in self.entities.items():
            self.settle_object(key, entity, physics)

    def attach_entities_to_arena(self):
        """Generic attachment logic"""
        for key, entity in self.entities.items():
            if any(obj in key for obj in self.attach_objects):
                entity.detach()
                self._arena.attach(entity)
'''

    # Make sure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Write the file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(template)

    print(f"模板已成功生成至: {output_path}")


# --- Usage example ---
if __name__ == "__main__":
    file_path = "/Users/lh/work/RMMBench/RMMBench/tasks/robocasa_task"
    task_manager_class = {
        "1":"NavigationConfigManager",
        "2":"BenchTaskConfigManager",

    }

    py_name = "wash_fruit_tidy_table"
    path = os.path.join(file_path, py_name+".py")
    task = "wash_fruit"  # the task name you want to generate


    generate_task_template(task, path,task_manager_class["2"])