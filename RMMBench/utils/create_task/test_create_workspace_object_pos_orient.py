import json
import numpy as np
import random
import os


def get_work_info(workspace, target_dim=(0.4, 0.4), anchor='bottom', offset_dist=0.0,y_set = 0.3):
    """
    offset_dist: actual physical distance to move (unit: meters). By default, when the anchor is 'bottom',
    this refers to the point nearest to the robot arm's left side: xmin, ymin.
    Positive values move toward the positive direction of the coordinate axis, negative values toward the
    negative direction. A positive offset_dist moves toward the robot arm's left side.
    A positive y_set moves the workspace away from the robot.
    """
    x_min, x_max, y_min, y_max, z_min, z_max = workspace
    ws_lx, ws_ly = x_max - x_min, y_max - y_min  # extents

    # 1. Fit to the actual dimensions (must not exceed the table itself)
    reg_lx, reg_ly = min(target_dim[0], ws_lx), min(target_dim[1], ws_ly)

    # Initial anchor point
    # start_x = x_min + (ws_lx - reg_lx) / 2
    # start_y = y_min + (ws_ly - reg_ly) / 2

    # 2. Adjust position based on the anchor and determine orientations
    if anchor == 'bottom':
        start_y = y_min+y_set
        start_x = x_min + (ws_lx - reg_lx) / 2-offset_dist
        reg_lx, reg_ly = min(target_dim[0], ws_lx), min(target_dim[1], ws_ly)
        workregion = [start_x, start_x + reg_lx, start_y, start_y + reg_ly,z_min, z_max]
        robot_orientation = [0, 0, 1.57]
        object_orientation = [0, 0, 0]

    elif anchor == 'top':
        # 'top' is rarely used
        reg_lx, reg_ly = min(target_dim[0], ws_lx), min(target_dim[1], ws_ly)
        start_y = y_max-y_set
        start_x = x_min + (ws_lx - reg_lx) / 2+reg_lx+offset_dist
        workregion = [start_x - reg_lx, start_x,start_y - reg_ly, start_y ,z_min, z_max]
        robot_orientation = [0, 0, -1.57]
        object_orientation = [0, 0, 3.14]

    elif anchor == 'left':
        reg_lx, reg_ly = min(target_dim[1], ws_lx), min(target_dim[0], ws_ly)
        start_x = x_min+y_set
        start_y = y_max-(ws_ly-reg_ly)/2+offset_dist
        workregion = [start_x, start_x + reg_lx, start_y - reg_ly,start_y, z_min, z_max]
        robot_orientation = [0, 0, 0]
        object_orientation = [0, 0, -1.57]

    elif anchor == 'right':
        reg_lx, reg_ly = min(target_dim[1], ws_lx), min(target_dim[0], ws_ly)
        start_x = x_max-y_set
        start_y = y_max-(ws_ly-reg_ly)/2-reg_ly-offset_dist
        workregion = [start_x- reg_lx, start_x , start_y,start_y + reg_ly, z_min, z_max]

        robot_orientation = [0, 0, 3.14]
        object_orientation = [0, 0, 1.57]

    # 3. Out-of-bounds safety check (very important: prevents an oversized offset_dist from
    # pushing the region off the table)
    # start_x = max(x_min, min(start_x, x_max - reg_lx))
    # start_y = max(y_min, min(start_y, y_max - reg_ly))
    print("workregion:",workregion)
    # Calculate robot xy position based on workregion edge closest to robot
    wx_min, wx_max, wy_min, wy_max, _, _ = workregion
    center_x = (wx_min + wx_max) / 2
    center_y = (wy_min + wy_max) / 2
    robot_offset = 0.4  # distance from workregion edge to robot
    if anchor == 'bottom':
        robot_xy = [center_x, wy_min - robot_offset]
    elif anchor == 'top':
        robot_xy = [center_x, wy_max + robot_offset]
    elif anchor == 'left':
        robot_xy = [wx_min - robot_offset, center_y]
    elif anchor == 'right':
        robot_xy = [wx_max + robot_offset, center_y]
    else:
        robot_xy = [center_x, center_y]
    print(f"[robot_position] anchor={anchor}, workregion_edge_xy={robot_xy}, "
          f"workregion={[wx_min, wx_max, wy_min, wy_max]}")
    return {
        "workregion": workregion,
        "z": z_max,
        "robot_orientation": robot_orientation,
        "object_orientation": object_orientation,
        "anchor": anchor,
        "robot_xy": robot_xy
    }



def get_robot_pose(work_info, offset=0.27):
    """Compute the robot position and orientation based on the anchor"""
    x_min, x_max, y_min, y_max = work_info["workregion"]
    robot_ori = work_info["robot_orientation"]
    anchor = work_info["anchor"]

    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2

    if anchor == 'bottom':
        robot_pos = [center_x, y_min - offset, 0.0]
    elif anchor == 'top':
        robot_pos = [center_x, y_max + offset, 0.0]
    elif anchor == 'left':
        robot_pos = [x_min - offset, center_y, 0.0]
    elif anchor == 'right':
        robot_pos = [x_max + offset, center_y, 0.0]
    else:
        robot_pos = [center_x, center_y, 0.0]

    return {"position": robot_pos, "orientation": robot_ori}


def long_horizon_grasp_arrange(work_info, container_name, objects_to_place, grid_nums=(10, 10), split_rate=0.1,**kwargs):
    """
    Long-horizon task arrangement logic:
    - objects_to_place: [[targets...], [others...]]
    - The returned structure contains objects: {target_entity: [], other_entity: []}
    """
    x_min, x_max, y_min, y_max = work_info["workregion"]
    z_surface = work_info["z"]
    obj_ori = work_info["object_orientation"]
    anchor = work_info["anchor"]

    target_names = objects_to_place[0]
    other_names = [item for sublist in objects_to_place[1:] for item in sublist]

    # --- Grid partitioning logic (same as before) ---
    x_centers = np.linspace(x_min, x_max, grid_nums[0])
    y_centers = np.linspace(y_min, y_max, grid_nums[1])

    container_grids = []
    potential_object_grids = []

    for x in x_centers:
        for y in y_centers:
            is_c = False
            if anchor == 'left' and y > y_max - (y_max - y_min) * split_rate:
                is_c = True
            elif anchor == 'right' and y < y_min + (y_max - y_min) * split_rate:
                is_c = True
            elif anchor == 'bottom' and x < x_min + (x_max - x_min) * split_rate:
                is_c = True
            elif anchor == 'top' and x > x_max - (x_max - x_min) * split_rate:
                is_c = True
            if is_c:
                container_grids.append((x, y))
            else:
                potential_object_grids.append((x, y))

    # --- Depth partitioning (the nearest 40% of the region) ---
    target_grids, other_grids = [], []
    for gx, gy in potential_object_grids:
        is_near = False
        if anchor == 'bottom' and gy < y_min + (y_max - y_min) * 0.4:
            is_near = True
        elif anchor == 'top' and gy > y_max - (y_max - y_min) * 0.4:
            is_near = True
        elif anchor == 'left' and gx < x_min + (x_max - x_min) * 0.4:
            is_near = True
        elif anchor == 'right' and gx > x_max - (x_max - x_min) * 0.4:
            is_near = True

        if is_near:
            target_grids.append((gx, gy))
        else:
            other_grids.append((gx, gy))

    # --- Build the return result ---
    arrange_res = {
        "container": {"name": container_name, "position": None, "orientation": obj_ori},
        "objects": {
            "target_entity": [],
            "other_entity": []
        }
    }

    if container_grids:
        c_pos = container_grids[len(container_grids) // 2]
        arrange_res["container"]["position"] = [float(c_pos[0]), float(c_pos[1]), float(z_surface)]

    # Place the targets
    s_target_grids = random.sample(target_grids, min(len(target_names), len(target_grids)))
    for name, pos in zip(target_names, s_target_grids):
        arrange_res["objects"]["target_entity"].append({
            "name": name,
            "position": [float(pos[0]), float(pos[1]), float(z_surface + 0.02)],
            "orientation": obj_ori
        })

    # Place the distractors
    s_other_grids = random.sample(other_grids, min(len(other_names), len(other_grids)))
    for name, pos in zip(other_names, s_other_grids):
        arrange_res["objects"]["other_entity"].append({
            "name": name,
            "position": [float(pos[0]), float(pos[1]), float(z_surface + 0.02)],
            "orientation": obj_ori
        })

    return arrange_res

def simple_grasp_place_arrange(work_info, container_name, objects_to_place, grid_nums=(10, 10), split_rate=0.3,**kwargs):
    """Divide the region into grids for the container and the objects"""
    x_min, x_max, y_min, y_max = work_info["workregion"]
    z_surface = work_info["z"]
    obj_ori = work_info["object_orientation"]
    anchor = work_info["anchor"]

    x_centers = np.linspace(x_min, x_max, grid_nums[0])
    y_centers = np.linspace(y_min, y_max, grid_nums[1])

    container_grids = []
    object_grids = []

    for x in x_centers:
        for y in y_centers:
            is_container_area = False
            if anchor == 'left':
                if y > y_max - (y_max - y_min) * split_rate: is_container_area = True
            elif anchor == 'right':
                if y < y_min + (y_max - y_min) * split_rate: is_container_area = True
            elif anchor == 'bottom':
                if x < x_min + (x_max - x_min) * split_rate: is_container_area = True
            elif anchor == 'top':
                if x > x_max - (x_max - x_min) * split_rate: is_container_area = True

            if is_container_area:
                container_grids.append((x, y))
            else:
                object_grids.append((x, y))

    # Build the local configuration

    container_config = {"name": container_name, "position": None, "orientation": obj_ori}
    objects_configs = []

    if container_grids:
        c_pos = container_grids[len(container_grids) // 2]
        container_config["position"] = [float(c_pos[0]), float(c_pos[1]), float(z_surface)]

    sampled_grids = random.sample(object_grids, min(len(objects_to_place), len(object_grids)))
    for i, obj_name in enumerate(objects_to_place[:len(sampled_grids)]):
        pos_xy = sampled_grids[i]
        objects_configs.append({
            "name": obj_name,
            "position": [float(pos_xy[0]), float(pos_xy[1]), float(z_surface + 0.02)],
            "orientation": obj_ori
        })

    return {"container": container_config, "objects": objects_configs}


def generate_and_save_task_config(task_name, scene_name, fixture_name, robot_config, arrange_res,**kwargs):
    """
    Concrete JSON construction and file-append logic
    """
    targets = arrange_res["objects"]["target_entity"]
    others = arrange_res["objects"]["other_entity"]

    # Build the asset section (without poses)
    asset = {
        "seen_object": [obj["name"] for obj in targets],
        "unseen_object": [obj["name"] for obj in others],
        "seen_container": [arrange_res["container"]["name"]] if arrange_res.get("container") else [],
        "robocasa_scene": scene_name,
        "fixture_top": fixture_name
    }

    # Build the components section (with poses)
    components = []
    if arrange_res.get("container"):
        components.append(arrange_res["container"])
    components.extend(targets)
    components.extend(others)

    new_entry = {
        "robot": {
            "position": robot_config["position"],
            "euler": robot_config["orientation"]
        },
        "task": {
            "asset": asset,
            "components": [],
            "scene": {"name": "empty"}
        }
    }

    # Path and save logic
    save_path = "/Users/lh/work/RMMBench/RMMBench/configs/task_config.json"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    config_data = {}
    if os.path.exists(save_path):
        try:
            with open(save_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            pass

    config_data[task_name] = new_entry

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=4, ensure_ascii=False)

    print(f"Task '{task_name}' synchronized to {save_path}")


def layout_objects_with_robot(work_info, container_name, objects_to_place,
                             task_name=None, scene_name=None, arrange_type="simple",get_json = None, **kwargs):
    """
    Dispatch hub: assemble the full robot-and-objects configuration and synchronize it to task_config.json
    """
    # 1. Get the robot pose
    robot_config = get_robot_pose(work_info)

    # 2. Call the concrete arrange method (mapped via the arrange_type string)
    if arrange_type == "simple":
        res = simple_grasp_place_arrange(work_info, container_name, objects_to_place, **kwargs)
        # Normalize internally so the save function can recognize it
        arrange_res = {
            "container": res["container"],
            "objects": {"target_entity": res["objects"], "other_entity": []}
        }
    elif arrange_type == "long":
        arrange_res = long_horizon_grasp_arrange(work_info, container_name, objects_to_place, **kwargs)
    else:
        raise ValueError(f"Unknown arrange type: {arrange_type}")

    # --- Added step: run the save function ---
    if get_json:
        fixture_name = kwargs.get("fixture_name", "counter_main_main_group")
        generate_and_save_task_config(
            task_name=task_name,
            scene_name=scene_name,
            fixture_name=fixture_name,
            robot_config=robot_config,
            arrange_res=arrange_res
        )

    # 3. Assemble the full task_config (keep the original return structure unchanged)
    task_config = {
        "robot": robot_config,
        "container": arrange_res["container"],
        "objects": arrange_res["objects"]
    }
    print("task_config:", task_config)
    return task_config



# get_workregion, layout_objects_by_anchor

if __name__ == "__main__":
    # workspace = [.25,2.25,-0.6000000000000001,0.0, 0.0, 0.92]  # counter_main
    with open("/Users/lh/work/RMMBench/RMMBench/configs/robocasa_scenes_config/robocasa_scene_config.json","r") as f:
        configs = json.load(f)
    scene_name = "ONE_WALL_LARGE"
    workspace = configs[scene_name]["island_counter_island_group"]["workspace"]
    work_info = get_work_info(workspace,anchor='left')
    # configs = layout_objects_with_robot(work_info, "delonghi_2", [["cup_1"],["plate_0","sugar","milk_0"]], arrange_type="long")

    # # Case A: simple task (Simple)
    # layout_objects_with_robot(
    #     work_info=work_info,
    #     container_name="tray_1",
    #     objects_to_place=["apple_1", "orange_1"],
    #     task_name="simple_fruit_sorting",
    #     scene_name="ONE_WALL_LARGE_1",
    #     arrange_type="simple",
    #     fixture_name="counter_main_main_group"  # passed in via kwargs
    # )

    # Case B: long-horizon task (Long)
    configs = layout_objects_with_robot(
        work_info=work_info,
        container_name="delonghi_2",
        # Note: the first sub-list is target_entity, the rest are other_entity
        objects_to_place=[["cup_1"],["plate_0","sugar","milk_0"]],
        task_name="get_coffee",
        scene_name="ONE_WALL_LARGE_1",
        arrange_type="long",
        fixture_name="counter_2_main_group",
        # get_json=True
    )

    print(configs)