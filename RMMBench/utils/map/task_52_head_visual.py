import os
import sys
import traceback

import numpy as np
import open3d
import cv2
from dm_control import viewer

from RMMBench.envs import load_env
from RMMBench.tasks import *
import json

from RMMBench.utils.utils import extract_containers,get_geom_ids_by_prefix,get_entity_mask_from_seg,get_placement_top_down_view

from pathlib import Path

from RMMBench.robots.single_arm.franka import Franka
# Check whether the point cloud is empty

# ===================== Parameter settings =====================
OVERWRITE = True  # True: overwrite existing images; False: skip existing images
ROBOT = "pandaomron"
TIME_LIMIT = 1000
IMG_PATH = "/Users/lh/work/RMMBench/RMMBench/utils/map/52_head_img"
ERROR_LOG = "/Users/lh/work/RMMBench/RMMBench/utils/map/52_head_img/error_log.json"

# Camera indices (see CAMERA_NAME_MAP in RMMBench/utils/utils.py)
#   right:0  left:1  forward:2  wrist:3  base_opposite:4  head:5
CAM_WRIST = 3   # wrist view
CAM_HEAD = 5    # head view

# 16 .py files -> (py file name, [list of task names])
TASK_MAP = {
    "pick_snack_single": ["pick_chocolate","pick_chip"],
    "pick_multi_snack_single_container": ["select_bagged_snacks", "cluster_bagged&bar_snack"],

    "wash_fruit_veg": ["wash_place_carrot", "wash_place_apple", "wash_place_cucumber"],
    "wash_multi_fruit_veg": ["wash_veg_salad", "wash_fruit_plate", "wash_broccoli_pepper"],
    "wash_fruit_shelf": ["wash_fruit_from_shelf"],

    "cook_meat_pan_stove": ["cook_steak_0", "cook_chicken_breast_0", "cook_sausage_0"],
    "reheat_microwave": ["reheat_steak_0", "heat_scone_0", "heat_waffle_0", "warm_baguette_0", "warm_croissant_0"],
    "store_meat_fridge": ["store_lamb_chicken_0", "store_pork_bacon_0", "store_sausage_fish_0"],
    "prepare_meat_tool": ["prepare_pork_tongs_0", "prepare_chicken_spatula_0", "prepare_pork_chop_spoon_0"],

    "pick_bread_toaster": ["select_bread_toaster_0", "select_bread_toaster_1", "select_bread_toaster_2"],
    "pick_bread_prep": ["place_bread_jam_knife_0", "place_waffle_syrup_fork_0", "place_bagel_butter_knife_0", "place_bread_jam_knife_1", "place_waffle_syrup_fork_1", "place_bagel_butter_knife_1"],

    "pick_kitchen_tool": ["pick_can_opener_0", "pick_whisk_0", "pick_bottle_opener_0"],
    "pick_multi_kitchen_tool": ["pick_can_opener_bottle_opener_0", "pick_whisk_knife_0", "pick_spatula_ladle_0"],

    "bring_cold_drink": ["bring_cold_water", "bring_cold_milk", "bring_cold_boxed_drink", "bring_non_alcoholic_drink", "bring_child_drink", "bring_no_sugar_drink"],
    "pick_drink_choice_no_fridge": ["pick_non_alcoholic_drink", "pick_child_drink", "pick_no_sugar_drink"],
    "store_item_fridge": ["store_non_alcoholic_drink", "store_child_drink", "store_no_sugar_drink"],
    "fix_burnt_bread": ["fix_burnt_bread_0", "fix_burnt_bread_1", "fix_burnt_bread_2"],

    "classify_snacks_on_shelf": ["classify_chocolate_snacks"],
    "uncover_object": ["uncover_fruit"],
    "store_cabinet": ["store_yogurt"],
    "select_same_class_cabinet": ["select_same_cake", "select_same_banana"],

}
# ====================================================

os.makedirs(IMG_PATH, exist_ok=True)

errors = []
total = sum(len(v) for v in TASK_MAP.values())
idx = 0

for py_name, task_names in TASK_MAP.items():
    for task_name in task_names:
        idx += 1
        head_file = os.path.join(IMG_PATH, f"{task_name}_head.png")
        wrist_file = os.path.join(IMG_PATH, f"{task_name}_wrist.png")

        # Skip existing images (controlled by OVERWRITE; skip only when both head/wrist exist)
        if os.path.exists(head_file) and os.path.exists(wrist_file) and not OVERWRITE:
            print(f"[{idx}/{total}] SKIP (exists): {task_name}")
            continue

        print(f"[{idx}/{total}] Loading: {task_name}  (from {py_name}.py)")
        try:
            env = load_env(task_name, robot=ROBOT, time_limit=TIME_LIMIT)
            env.reset()
            rgb_data = env.get_observation()
            vlm_img_input = [cv2.cvtColor(rgb_data["rgb"][i], cv2.COLOR_BGR2RGB) for i in range(6)]
            cv2.imwrite(head_file, vlm_img_input[CAM_HEAD])    # index 5 = head view
            cv2.imwrite(wrist_file, vlm_img_input[CAM_WRIST])  # index 3 = wrist view
            print(f"  -> saved head : {head_file}")
            print(f"  -> saved wrist: {wrist_file}")
            env.close()
        except Exception as e:
            err_msg = f"{e.__class__.__name__}: {e}"
            errors.append({"task_name": task_name, "py_file": py_name, "error": err_msg})
            print(f"  -> ERROR: {err_msg}")
            traceback.print_exc()
            # Make sure env is closed
            try:
                env.close()
            except:
                pass

# Save the error log
if errors:
    with open(ERROR_LOG, "w") as f:
        json.dump(errors, f, indent=2, ensure_ascii=False)
    print(f"\n{len(errors)} task(s) failed. Error log saved to: {ERROR_LOG}")
else:
    print(f"\nAll {total} tasks completed successfully.")

# ===================== Generate HTML preview =====================
HTML_FILE = os.path.join(IMG_PATH, "52_head_wrist_gallery.html")

html_parts = []
html_parts.append("<!DOCTYPE html>")
html_parts.append("<html><head><meta charset='utf-8'>")
html_parts.append("<title>52 Head & Wrist View Gallery</title>")
html_parts.append("<style>")
html_parts.append("  body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }")
html_parts.append("  h2 { color: #333; border-bottom: 2px solid #ccc; padding-bottom: 5px; margin-top: 40px; }")
html_parts.append("  .row { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 20px; }")
html_parts.append("  .item { text-align: center; width: 19%; min-width: 200px; }")
html_parts.append("  .item img { width: 100%; border: 1px solid #ddd; border-radius: 4px; }")
html_parts.append("  .item .label { font-size: 12px; color: #555; margin-top: 4px; word-break: break-all; }")
html_parts.append("  .item .view { font-size: 11px; color: #888; margin-top: 2px; }")
html_parts.append("  .item.missing img { background: #eee; height: 150px; }")
html_parts.append("  .item.missing .label { color: #c00; }")
html_parts.append("</style></head><body>")
html_parts.append("<h1>52 Head & Wrist View Gallery</h1>")

for group_name, task_names in TASK_MAP.items():
    html_parts.append(f"<h2>{group_name} ({len(task_names)} tasks)</h2>")
    html_parts.append("<div class='row'>")
    for task_name in task_names:
        head_img = f"{task_name}_head.png"
        wrist_img = f"{task_name}_wrist.png"
        html_parts.append(f"  <div class='item'>")
        # head view
        head_path = os.path.join(IMG_PATH, head_img)
        if os.path.exists(head_path):
            html_parts.append(f"    <img src='{head_img}' alt='{task_name} head'>")
        else:
            html_parts.append(f"    <img src='' alt='head missing' class='missing'>")
        html_parts.append(f"    <div class='view'>head</div>")
        # wrist view
        wrist_path = os.path.join(IMG_PATH, wrist_img)
        if os.path.exists(wrist_path):
            html_parts.append(f"    <img src='{wrist_img}' alt='{task_name} wrist'>")
        else:
            html_parts.append(f"    <img src='' alt='wrist missing' class='missing'>")
        html_parts.append(f"    <div class='view'>wrist</div>")
        html_parts.append(f"    <div class='label'>{task_name}</div>")
        html_parts.append(f"  </div>")
    html_parts.append("</div>")

html_parts.append("</body></html>")

with open(HTML_FILE, "w") as f:
    f.write("\n".join(html_parts))
print(f"HTML gallery saved to: {HTML_FILE}")

print("Done.")
