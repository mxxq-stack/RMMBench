import json
import numpy as np
import random
import os


class TaskSceneGenerator:
    def __init__(self, config_path="/Users/lh/work/RMMBench/RMMBench/configs/robocasa_scenes_config/robocasa_scene_config.json"):
        with open(config_path, "r") as f:
            self.all_configs = json.load(f)

    # --- Keep the previously written get_fixture_workspace, get_workregion, etc. methods here ---
    # ... (previous code omitted) ...

    def export_to_rmmbench_format(self, scene_name, task_name, workregion,
                                  target_entity, other_entity=None,
                                  container=None, init_container=None,
                                  task_type="long_sequence_1",
                                  output_filename="rmmbench_task_config.json"):
        """
        Integration method: generate a complete RMMBench-format JSON directly from the object list
        """
        # 1. Call the previous placement logic to obtain specific pos and quat
        placements = self.arrange_task(
            workregion=workregion,
            task_type=task_type,
            target_entity=target_entity,
            other_entity=other_entity,
            container=container,
            init_container=init_container
        )

        # 2. Compute the robot position
        bbox = workregion["bbox"]
        anchor = workregion["anchor"]
        wr_center_x = (bbox[0] + bbox[1]) / 2
        wr_center_y = (bbox[2] + bbox[3]) / 2

        # 1. Robot placement computation logic (offset 0.1m from the boundary edge)
        offset = -0.2
        # Unpack the workregion boundaries
        x_min, x_max, y_min, y_max = bbox

        if anchor == 'bottom':
            # Centered on the X axis, Y axis shifted outward from the minimum value
            robot_pos = [wr_center_x, y_min - offset, 0]
        elif anchor == 'top':
            # Centered on the X axis, Y axis shifted outward from the maximum value
            robot_pos = [wr_center_x, y_max + offset, 0]
        elif anchor == 'left':
            # Centered on the Y axis, X axis shifted outward from the minimum value
            robot_pos = [x_min - offset, wr_center_y, 0]
        elif anchor == 'right':
            # Centered on the Y axis, X axis shifted outward from the maximum value
            robot_pos = [x_max + offset, wr_center_y, 0]
        else:
            # Default: same as bottom
            robot_pos = [wr_center_x, y_min - offset, 0]

        # 3. Build the standard RMMBench dictionary structure
        rmmbench_config = {
            task_name: {
                "robot": {
                    "position": robot_pos
                },
                "task": {
                    "random_init": False,
                    "ngrid": [10, 10],
                    "asset": {
                        "seen_object": target_entity,
                        "unseen_object": other_entity if other_entity else [],
                        "mid_contain": {},  # left empty for now
                        "seen_container": [container] if container else [],
                        "init_container": [init_container] if init_container else [],
                        "robocasa_scene": scene_name,
                        "placements": placements  # detailed world-coordinate placement data
                    },
                    "components": [],
                    "scene": {
                        "name": "empty"
                    }
                }
            }
        }


        with open(output_filename, "w") as f:
            json.dump(rmmbench_config, f, indent=4)

        print(f"任务 '{task_name}' 已成功导出至 {output_filename}")
        return rmmbench_config

    def get_fixture_workspace(self, scene_name, fixture_name):
        """
        Extract the original workspace from the JSON given a scene name and fixture name
        """
        scene_dict = self.all_configs.get(scene_name, {})
        # Iterate over all groups in the scene (main_group, island_group, etc.)
        for group_name, fixtures in scene_dict.items():
            if fixture_name in fixtures:
                return fixtures[fixture_name]["workspace"]

        raise ValueError(f"在场景 '{scene_name}' 中找不到设施 '{fixture_name}'")

    def get_workregion(self, workspace, target_dim=(0.8, 0.4), anchor='bottom', offset_rate=0):
        """
        Scale the original workspace and align it to the ideal 0.8x0.4 operating region
        to obtain the operating region's world coordinates:
                x_min, x_max, y_min, y_max, z_min, z_max = workspace
                z_max: height of the top surface above the ground origin

        """
        x_min, x_max, y_min, y_max, z_min, z_max = workspace
        ws_lx, ws_ly = x_max - x_min, y_max - y_min

        # 1. Fit to the actual dimensions
        reg_lx, reg_ly = min(target_dim[0], ws_lx), min(target_dim[1], ws_ly)

        # Default centered position
        start_x = x_min + (ws_lx - reg_lx) / 2
        start_y = y_min + (ws_ly - reg_ly) / 2

        # 2. Adjust position based on the anchor and determine orientations
        if anchor == 'bottom':
            start_y = y_min
            start_x += offset_rate * (ws_lx - reg_lx) / 2
            robot_orientation = [0, 0, 1.57]
            object_orientation = [0, 0, 0]
        elif anchor == 'top':
            start_y = y_max - reg_ly
            start_x += offset_rate * (ws_lx - reg_lx) / 2
            robot_orientation = [0, 0, -1.57]
            object_orientation = [0, 0, 3.14]

        elif anchor == 'left':
            start_x = x_min
            start_y += offset_rate * (ws_ly - reg_ly) / 2
            robot_orientation = [0, 0, 0]
            object_orientation = [0, 0, -1.57]

        elif anchor == 'right':
            start_x = x_max - reg_lx
            start_y += offset_rate * (ws_ly - reg_ly) / 2
            robot_orientation = [0, 0, 3.14]
            object_orientation = [0, 0, 1.57]


        # 3. Out-of-bounds safety check
        start_x = max(x_min, min(start_x, x_max - reg_lx))
        start_y = max(y_min, min(start_y, y_max - reg_ly))

        return {
            "workregion": [start_x, start_x + reg_lx, start_y, start_y + reg_ly],
            "z": z_max,
            "robot_orientation": robot_orientation,
            "object_orientation": object_orientation,
            "anchor": anchor
        }

    def arrange_task(self, workregion, task_type, target_entity, other_entity=None, container=None,
                     init_container=None):
        """
        Core placement logic
        """
        bbox = workregion["bbox"]
        z = workregion["z"]
        orientation = workregion["orientation"]
        anchor = workregion["anchor"]

        placements = {}
        other_entity = other_entity or []

        # Split into left and right sub-regions
        mid_x = (bbox[0] + bbox[1]) / 2
        left_zone = [bbox[0], mid_x, bbox[2], bbox[3]]
        right_zone = [mid_x, bbox[1], bbox[2], bbox[3]]

        if task_type == "simple_grasp":
            # Strategy: objects on the left, container on the right
            if container:
                placements[container] = self._get_center_pos(right_zone, z)

            objs_to_place = target_entity + other_entity
            if init_container: objs_to_place.append(init_container)

            grid = self._generate_grid(left_zone, z, n_grid=(5, 5))
            random.shuffle(grid)
            for i, obj in enumerate(objs_to_place):
                placements[obj] = grid[i]

        elif task_type == "long_sequence_1":
            # Strategy: container on the left (microwave etc.), manipulated objects on the right (milk, cups, etc.)
            if container:
                placements[container] = self._get_center_pos(left_zone, z)

            grid = self._generate_grid(right_zone, z, n_grid=(8, 8))
            grid = self._sort_grid_by_anchor(grid, anchor)

            # Target and init_container stay close to the robot side, with staggered indices to avoid parallel overlap
            placements[target_entity[0]] = grid[0]
            if init_container:
                # Pick a point with some depth that does not fully overlap
                placements[init_container] = grid[min(4, len(grid) - 1)]

            # Remaining objects serve as background distractors
            far_grid = grid[max(10, len(grid) // 2):]
            random.shuffle(far_grid)
            for i, obj in enumerate(other_entity):
                if i < len(far_grid):
                    placements[obj] = far_grid[i]

        # Format the output as a world-coordinate dictionary
        final_output = {}
        for name, pos in placements.items():
            final_output[name] = {
                "pos": pos.tolist() if isinstance(pos, np.ndarray) else pos,
                "orientation": orientation
            }
        return final_output

    # --- Internal helper utilities ---
    def _get_center_pos(self, zone, z):
        return np.array([(zone[0] + zone[1]) / 2, (zone[2] + zone[3]) / 2, z])

    def _generate_grid(self, zone, z, n_grid=(5, 5)):
        margin = 0.04  # reserved edge margin
        xs = np.linspace(zone[0] + margin, zone[1] - margin, n_grid[0])
        ys = np.linspace(zone[2] + margin, zone[3] - margin, n_grid[1])
        return [np.array([x, y, z]) for x in xs for y in ys]

    def _sort_grid_by_anchor(self, grid, anchor):
        if anchor == 'bottom': return sorted(grid, key=lambda p: p[1])
        if anchor == 'top':    return sorted(grid, key=lambda p: -p[1])
        if anchor == 'left':   return sorted(grid, key=lambda p: p[0])
        if anchor == 'right':  return sorted(grid, key=lambda p: -p[0])
        return grid


# --- Actual run example ---
if __name__ == "__main__":
    # Define the base path to ensure all files live in the same directory
    base_path = "/Users/lh/work/RMMBench/"
    config_path = "/Users/lh/work/RMMBench/RMMBench/configs/robocasa_scenes_config/robocasa_scene_config.json"
    save_path = os.path.join(base_path, "rmmbench_task_config.json")
    generator = TaskSceneGenerator(config_path)

    # 1. Get the region for a specific scene and fixture
    ws = generator.get_fixture_workspace("ONE_WALL_LARGE", "counter_main_main_group")

    # 2. Generate the workregion (assumed to be the main counter, close to the robot side)
    workregion = generator.get_workregion(ws, anchor='bottom', offset_rate=0.2)

    scene_name = "ONE_WALL_LARGE_1"
    target_entity = ["milk_bottle"]
    init_container = "mug_1"
    container = "microwave_1"
    other_entity = ["bread_slice", "suger_0"]

    # 4. Run the export
    final_config = generator.export_to_rmmbench_format(
        scene_name=scene_name,
        task_name="heat_milk_task",
        workregion=workregion,
        target_entity=target_entity,
        other_entity=other_entity,
        container=container,
        init_container=init_container,
        task_type="long_sequence_1",
        output_filename=save_path  # use the absolute path
    )

    print("\n--- RMMBench 任务配置已生成 ---")
    # Print the robot position for verification
    print(f"Robot Position: {final_config['heat_milk_task']['robot']['position']}")