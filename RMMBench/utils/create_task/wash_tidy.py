import json
import os

# Define the task configuration data
tasks_to_process = {
    "cool_milk_serve_bread": {
        "sub_tasks": ["cool_milk_in_sink", "cool_milk_serve_bread_on_tray"],
        "instr_1": "Place the milk into the sink to cool it down.",
        "instr_2": "Place the milk into the sink and move the bread to the tray.",
        "attach_objs": '["tray"]'
    },
    "wash_meat_organize_cheese": {
        "sub_tasks": ["wash_meat_in_sink", "wash_meat_organize_cheese_on_tray"],
        "instr_1": "Put the meat in the sink.",
        "instr_2": "Put the meat in the sink and organize the cheese slices on the tray.",
        "attach_objs": '["tray"]'
    },
    "wash_cutlery_return_seasoning": {
        "sub_tasks": ["wash_knives_forks_in_sink", "wash_cutlery_return_seasoning_to_tray"],
        "instr_1": "Put the used tableware into the sink.",
        "instr_2": "Put the used tableware into the sink and return the seasoning bottles to the tray.",
        "attach_objs": '["tray"]'
    },
    "ice_juice_arrange_cake": {
        "sub_tasks": ["ice_juice_bottle_in_sink", "ice_juice_arrange_cake_on_tray"],
        "instr_1": "Place the juice in the sink for icing.",
        "instr_2": "Place the juice in the sink for icing and arrange the cake slices on the tray.",
        "attach_objs": '["tray"]'
    },
"ice_cola_move_snacks": {
        "sub_tasks":  ["ice_cola_in_sink", "ice_cola_move_snacks_to_tray"],
        "instr_1": "Put the cola into the sink for icing.",
        "instr_2": "Put the cola into the sink for icing and move the snacks to the tray.",
        "attach_objs": '["tray"]'
    },
}

# Base template string
TEMPLATE = """import random
from RMMBench.tasks.base_task import PrimitiveTask
from RMMBench.tasks.config_manager import ClassifyConfigManager
from RMMBench.utils.register import register
from functools import partial
from RMMBench.utils.skill_lib import SkillLib

# --- {Base} Config Manager ---
@register.add_config_manager("{base_task}")
class {Base}ConfigManager(ClassifyConfigManager):
    def __init__(self, task_name, num_objects=[1], **kwargs):
        super().__init__(task_name, num_objects, **kwargs)

    def load_containers(self, target_container, offset=0.16, y_set=-0.05, direction="right"):
        pass

    def get_object_info(self, workregion_offset=-0.62, workregion_y_set=0.07, target_dim=(0.3, 0.25), grid_size=[6, 6]):
        super().get_object_info(workregion_offset, workregion_y_set, target_dim=target_dim, grid_size=grid_size)

    def load_init_containers(self, init_container):
        pass

    def get_instruction(self, target_entity, target_container, **kwargs):
        instruction = ["{instr_1}"]
        self.config["task"]["instructions"] = instruction
        return self.config

    def get_condition_config(self, target_entity, target_container, **kwargs):
        conditions_config = dict(
            scene_contain=dict(
                scene=self.robocasa_scene,
                container_name="sink_island_group",
                entities=self.target_entity,
                vel_th=0.01,
            ),
            scene_not_contain=dict(
                scene=self.robocasa_scene,
                container_name="sink_island_group",
                entities=self.other_entity,
                vel_th=0.01,
            )
        )
        self.config["task"]["conditions"] = conditions_config

# --- {Full} Config Manager ---
@register.add_config_manager("{full_task}")
class {Full}ConfigManager({Base}ConfigManager):
    def __init__(self, task_name, num_objects=[1], **kwargs):
        super().__init__(task_name, num_objects, **kwargs)

    def get_object_info(self, workregion_offset=-0.6, workregion_y_set=0.07, target_dim=(0.3, 0.25), grid_size=[6, 6]):
        super().get_object_info(workregion_offset, workregion_y_set, target_dim=target_dim, grid_size=grid_size)


    def get_instruction(self, target_entity, target_container, **kwargs):
        instruction = ["{instr_2}"]
        self.config["task"]["instructions"] = instruction
        return self.config  # must return the config object

    def load_containers(self, target_container,offset=0.3,y_set = -0.05,direction="right"):
        if target_container is not None:
            if self.work_info and self.target_container:
                container_info = self.get_container_info_from_workregion(self.work_info,
                                                                         anchor=self.destination_position,
                                                                         offset=offset,
                                                                         y_set = y_set,
                                                                        direction = direction
                                                                         )
                container_config = self.get_entity_config(target_container,
                                                          position=container_info["position"],
                                                          orientation=container_info["orientation"],)
                self.config["task"]["components"].append(container_config)
            else:
                container_config = self.get_entity_config(target_container)
                self.config["task"]["components"].append(container_config)

    def get_condition_config(self, target_entity, target_container, **kwargs):
        conditions_config = dict(
            contain_v=dict(
                container=target_container,
                entities=self.other_entity,
                vel_th=0.01,
            )
            ,
            scene_contain=dict(
                scene=self.robocasa_scene,
                container_name="sink_island_group",
                entities=self.target_entity,
                vel_th=0.01,
            )
        )
        self.config["task"]["conditions"] = conditions_config

# --- {Base} Task Class ---
@register.add_task("{base_task}")
class {Base}Task(PrimitiveTask):
    def __init__(self, task_name, robot, **kwargs):
        self.attach_objects = {attach_objs}
        super().__init__(task_name, robot=robot, **kwargs)

    def build_from_config(self, eval=False, **kwargs):
        super().build_from_config(eval, **kwargs)
        self.attach_entities_to_arena()


    def initialize_episode(self, physics, random_state):
        super().initialize_episode(physics, random_state)
        for key, entity in self.entities.items():
            self.settle_object(key, entity, physics)

    def attach_entities_to_arena(self):
        for key, entity in self.entities.items():
            if any(obj in key for obj in self.attach_objects):
                entity.detach()
                self._arena.attach(entity)

    def get_expert_skill_sequence(self, physics):
        init_ee_pos, init_ee_quat = self.robot.get_end_effector_pos(physics), self.robot.get_end_effector_quat(physics)
        sink_placement = [2.62, -2.16, 1.04] 
        skill_sequence = []
        for entity in self.config_manager.target_entity:
            skill_sequence.extend([
                partial(SkillLib.pick, target_entity_name=entity),
                partial(SkillLib.place, target_pos=sink_placement),
                partial(SkillLib.reset, target_pos=list(init_ee_pos), target_quat=init_ee_quat),
            ])
        skill_sequence.append(partial(SkillLib.end))
        return skill_sequence

# --- {Full} Task Class ---
@register.add_task("{full_task}")
class {Full}Task({Base}Task):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)

    def get_expert_skill_sequence(self, physics):
        init_ee_pos, init_ee_quat = list(self.robot.get_end_effector_pos(physics)), self.robot.get_end_effector_quat(physics)
        sink_placement = [2.62, -2.16, 0.94]
        skill_sequence = []
        for entity in self.config_manager.target_entity:
            skill_sequence.extend([
                partial(SkillLib.pick, target_entity_name=entity),
                partial(SkillLib.place, target_pos=sink_placement),
                partial(SkillLib.reset, target_pos=init_ee_pos, target_quat=init_ee_quat),
            ])
        for entity in self.config_manager.other_entity:
            skill_sequence.extend([
                partial(SkillLib.pick, target_entity_name=entity),
                partial(SkillLib.place, target_container_name=self.target_container),
                partial(SkillLib.reset, target_pos=init_ee_pos, target_quat=init_ee_quat),
            ])
        skill_sequence.append(partial(SkillLib.end))
        return skill_sequence
"""

def to_camel_case(text):
    # split('_') breaks the string into a list of words
    # capitalize() uppercases the first letter of each word and lowercases the rest
    return "".join(word.capitalize() for word in text.split("_"))

def batch_write_task_files_py(target_dir, data):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for file_name, info in data.items():
        file_path = os.path.join(target_dir, f"{file_name}.py")

        # Fill in the template contents
        class_ = []
        for sub_task in info["sub_tasks"]:
            class_.append(to_camel_case(sub_task))
        Base = class_[0]
        Full = class_[1]

        # Print the results for verification
        print(f"Base: {Base}")  # output: WashFruit
        print(f"Full: {Full}")  # output: WashFruitTidyTable
        content = TEMPLATE.format(
            base_task=info["sub_tasks"][0],
            full_task=info["sub_tasks"][1],
            instr_1=info["instr_1"],
            instr_2=info["instr_2"],
            attach_objs=info["attach_objs"],
            Base=Base,
            Full=Full
        )

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Success: Wrote to {file_path}")

def write_task_json_config(task_name,path):
    # This can be changed dynamically based on the loop
    new_task_content = {
        "robot": {
            "position": [2.15, -1.7, 0.0],
            "euler": [0, 0, -1.57]
        },
        "task": {
            "asset": {
                "seen_object": [["apple_0"], ["lemon_1"]],
                "unseen_object": [["corn_3"], ["carrot_0"]],
                "seen_container": ["tray_4"],
                "random_scene": False,
                "robocasa_scene": "ONE_WALL_LARGE_1",
                "destination_position": "top",
                "fixture_surface": "island_counter_island_group"
            },
            "components": [],
            "scene": {
                "name": "empty"
            }
        }
    }

    json_path = path

    # 2. Read the existing JSON file and update it
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

    # Add the new task to the dictionary
    data[task_name] = new_task_content

    # 3. Write the JSON file back
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)



def write_camera_json_config(task_name,path):
    # This can be changed dynamically based on the loop
    new_task_content = {
        "forward": {
          "pos": "2 -4 5",
          "xyaxes": "1.000 0.015 -0 0.008 1.65 0.5",
          "fovy": "65"
        }
    }

    json_path = path

    # 2. Read the existing JSON file and update it
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

    # Add the new task to the dictionary
    data[task_name] = new_task_content

    # 3. Write the JSON file back
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


# Run the writes

# Fill in the .py files
RMMBench_path = "/Users/lh/work/RMMBench"
py_path = os.path.join(RMMBench_path, "RMMBench/tasks/robocasa_task")

# Write the .py files
batch_write_task_files_py(py_path, tasks_to_process)

# Fill in the task JSON
task_json_path = os.path.join(RMMBench_path, "RMMBench/configs/task_config.json")
camera_json_path = os.path.join(RMMBench_path, "RMMBench/configs/camera_config.json")
for task_name in tasks_to_process.keys():
    print(f"Writing task {task_name}")
    # Commented out
    # write_task_json_config(task_name, task_json_path)
    # write_camera_json_config(task_name, camera_json_path)
# Fill in the camera JSON