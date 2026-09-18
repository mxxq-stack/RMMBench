import json
import os
from pathlib import Path


# Define the task configuration data
tasks_to_process = {

        "select_fruit_closest_juicer": {
            "sub_tasks": [
                "cool_juice_in_sink",
                "cool_juice_return_seasoning"
            ],
            "instr_1": "Place the juice into the sink to cool it down.",
            "instr_2": "Place the juice into the sink and return the seasoning to the tray.",
            "attach_objs": ["tray"]
        }

}

# Base template string
TEMPLATE = """
import random

from open3d.examples.pipelines.colored_icp_registration import target

from RMMBench.tasks.dm_task import *
from RMMBench.tasks.base_task import PrimitiveTask
from RMMBench.tasks.config_manager import BenchTaskConfigManager
from RMMBench.utils.register import register
from RMMBench.utils.utils import flatten_list,grid_sample


@register.add_config_manager("{base_task}")
class {Base}ConfigManager(BenchTaskConfigManager):
    def __init__(self,
                 task_name,
                 num_objects=[4],
                 **kwargs):
        super().__init__(task_name, num_objects, **kwargs)

    def load_containers(self, target_container):
        juicer_config = self.get_entity_config(target_container,position=[0.28, -0.1, 0.78])
        self.config["task"]["components"].append(juicer_config)

    def load_init_containers(self, init_container):
        pass

    def get_instruction(self, target_entity, target_container, **kwargs):
        instruction = [f"{instr_1}"]
        self.config["task"]["instructions"] = instruction

    def get_condition_config(self, target_entity, target_container, **kwargs):
        conditions_config = dict(
            contain=dict(
                container=f"{{target_container}}",
                entities=[f"{{target_entity}}"],
            )
        )
        self.config["task"]["conditions"] = conditions_config

    def load_objects(self, target_entity):
        objects = []
        other_objects = flatten_list(self.seen_object)
        other_objects.remove(target_entity)
        objects.extend(random.sample(other_objects, self.num_object - 1))
        jucier_pos = [-0.2,0.1,0.83]
        jucier_config = self.get_entity_config(self.unseen_object[0],position=jucier_pos,
                                               orientation=[0,0,1.57])
        self.config["task"]["components"].append(jucier_config)
        target_entity_pos = [jucier_pos[0],jucier_pos[1]-0.2,jucier_pos[2]-0.03]
        target_config = self.get_entity_config(target_entity,target_entity_pos,
                                               orientation=[0,0,1.57])
        self.config["task"]["components"].append(target_config)
        for i, object in enumerate(objects):
            object_config = self.get_entity_config(object, position=
            [target_entity_pos[0]+0.12*i-0.2,target_entity_pos[1]-0.15,target_entity_pos[2]])
            self.config["task"]["components"].append(object_config)




@register.add_task(f"{base_task}")
class {Base}Task(PrimitiveTask):
    def __init__(self, task_name, robot, **kwargs):
        self.attach_objects = {attach_objs}
        super().__init__(task_name, robot=robot, **kwargs)

    def build_from_config(self, eval=False, **kwargs):
        super().build_from_config(eval, **kwargs)
        self.attach_entities_to_arena()

    def attach_entities_to_arena(self):
        for key, entity in self.entities.items():
            if any(obj in key for obj in self.attach_objects):
                entity.detach()
                self._arena.attach(entity)

    def get_expert_skill_sequence(self, physics):
        c_pos = list(np.array(self.entities[self.target_container].get_place_point(physics))[0])
        skill_sequence = [
            partial(SkillLib.pick, target_entity_name=self.target_entity),
            partial(SkillLib.place, target_pos=[c_pos[0],c_pos[1],c_pos[2]],target_container_name=self.target_container),
            partial(SkillLib.end),
        ]
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
        "task": {
            "random_ignored_entities": [
                "table"
            ],
            "ngrid": [
                10,
                10
            ],
            "workspace": [
                -0.3,
                0,
                -0.2,
                0.1,
                0.8,
                1.5
            ],
            "random_init": False,
            "asset": {
                "seen_object": [
                    "apple_0",
                    "orange_1",
                    "lemon_0",
                    "mango_1"
                ],
                "unseen_object": [
                    "juicer"
                ],
                "seen_container": [
                    "plate_4"
                ]
            },
            "components": [
                {
                    "name": "table",
                    "xml_path": "obj/meshes/table/table.xml",
                    "class": "Table",
                    "materials": [
                        "wood0",
                        "wood1",
                        "stone0"
                    ],
                    "randomness": {
                        "texture": True
                    }
                }
            ],
            "scene": {
                "name": "kitchen_2"
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
current_file = Path(__file__).resolve()
vla_root = str(current_file.parents[4])
print("vla_root:", vla_root)
# Fill in the .py files
py_path = os.path.join(vla_root, "RMMBench/tasks/robocasa_task")

# Write the .py files
batch_write_task_files_py(py_path, tasks_to_process)

# Fill in the task JSON
task_json_path = os.path.join(vla_root, "RMMBench/configs/task_config.json")
camera_json_path = os.path.join(vla_root, "RMMBench/configs/camera_config.json")
for task_name in tasks_to_process.keys():
    print(f"Writing task {task_name}")
    # Commented out
    write_task_json_config(task_name, task_json_path)
    write_camera_json_config(task_name, camera_json_path)
# Fill in the camera JSON