import os
import sys
import json
import traceback

import numpy as np
import open3d
import cv2
from dm_control import viewer

from RMMBench.envs import load_env
from RMMBench.tasks import *

from RMMBench.utils.utils import extract_containers, get_geom_ids_by_prefix, get_entity_mask_from_seg, get_placement_top_down_view

from pathlib import Path

from RMMBench.robots.single_arm.franka import Franka

# ===================== Parameter settings =====================
ROBOT = "pandaomron"
TIME_LIMIT = 1000
IMG_PATH = "/Users/lh/work/RMMBench/RMMBench/utils/map/nav_forward_img"
ERROR_LOG = "/Users/lh/work/RMMBench/RMMBench/utils/map/nav_forward_img/error_log.json"

# Index of the forward view in the rgb array
FORWARD_INDEX = 2

# List of nav subtasks
TASK_MAP = {
    # "nav_wash_place": [
    #     "nav_wash_place_carrot_0", "nav_wash_place_carrot_1", "nav_wash_place_carrot_2", "nav_wash_place_carrot_3",
    #     "nav_wash_place_apple_0", "nav_wash_place_apple_1",
    #     "nav_wash_place_cucumber_0", "nav_wash_place_cucumber_1",
    # ],
    "nav_pick_snacks": [
        "nav_pick_chocolate_0", "nav_pick_chocolate_1", "nav_pick_chocolate_2", "nav_pick_chocolate_3",
        "nav_pick_chip_0", "nav_pick_chip_1", "nav_pick_chip_2", "nav_pick_chip_3",
    ],
    # "nav_cook_steak": [
    #     "nav_cook_steak_0", "nav_cook_steak_1", "nav_cook_steak_2", "nav_cook_steak_3",
    # ],
    # "nav_select_tool": [
    #     "nav_select_tool_galley_0", "nav_select_tool_galley_1", "nav_select_tool_galley_2",
    #     "nav_select_tool_l_large_0", "nav_select_tool_l_large_1", "nav_select_tool_l_large_2",
    #     "nav_select_tool_l_small_0", "nav_select_tool_l_small_1", "nav_select_tool_l_small_2",
    # ],
    # "nav_burnt_bread": [
    #     "nav_burnt_bread_0", "nav_burnt_bread_1", "nav_burnt_bread_2",
    #     "nav_burnt_bread_3", "nav_burnt_bread_4", "nav_burnt_bread_5",
    #     "nav_burnt_bread_6", "nav_burnt_bread_7", "nav_burnt_bread_8",
    # ],
    # "nav_store_item_fridge": [
    #     "nav_store_non_alcoholic_drink_0", "nav_store_non_alcoholic_drink_1",
    #     "nav_store_non_alcoholic_drink_2", "nav_store_non_alcoholic_drink_3",
    #     "nav_store_child_drink_0", "nav_store_child_drink_1",
    #     "nav_store_child_drink_2", "nav_store_child_drink_3",
    #     "nav_store_no_sugar_drink_0", "nav_store_no_sugar_drink_1",
    #     "nav_store_no_sugar_drink_2", "nav_store_no_sugar_drink_3",
    # ],
}
# ====================================================

os.makedirs(IMG_PATH, exist_ok=True)

errors = []
total = sum(len(v) for v in TASK_MAP.values())
idx = 0

for py_name, task_names in TASK_MAP.items():
    for task_name in task_names:
        idx += 1
        out_file = os.path.join(IMG_PATH, f"{task_name}.png")

        # Skip existing images
        if os.path.exists(out_file):
            print(f"[{idx}/{total}] SKIP (exists): {task_name}")
            continue

        print(f"[{idx}/{total}] Loading: {task_name}  (from {py_name}.py)")
        try:
            env = load_env(task_name, robot=ROBOT, time_limit=TIME_LIMIT)
            env.reset()
            rgb_data = env.get_observation()
            num_cams = rgb_data["rgb"].shape[0]
            if FORWARD_INDEX >= num_cams:
                raise IndexError(
                    f"forward index {FORWARD_INDEX} out of range, only {num_cams} cameras available"
                )
            forward_img = cv2.cvtColor(rgb_data["rgb"][FORWARD_INDEX], cv2.COLOR_BGR2RGB)
            cv2.imwrite(out_file, forward_img)
            print(f"  -> saved: {out_file}")
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
HTML_FILE = os.path.join(IMG_PATH, "nav_forward_gallery.html")
IMAGES_PER_ROW = 5

html_parts = []
html_parts.append("<!DOCTYPE html>")
html_parts.append("<html><head><meta charset='utf-8'>")
html_parts.append("<title>Nav Forward View Gallery</title>")
html_parts.append("<style>")
html_parts.append("  body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }")
html_parts.append("  h2 { color: #333; border-bottom: 2px solid #ccc; padding-bottom: 5px; margin-top: 40px; }")
html_parts.append("  .row { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 20px; }")
html_parts.append("  .item { text-align: center; width: 19%; min-width: 200px; }")
html_parts.append("  .item img { width: 100%; border: 1px solid #ddd; border-radius: 4px; }")
html_parts.append("  .item .label { font-size: 12px; color: #555; margin-top: 4px; word-break: break-all; }")
html_parts.append("  .item.missing img { background: #eee; height: 150px; }")
html_parts.append("  .item.missing .label { color: #c00; }")
html_parts.append("</style></head><body>")
html_parts.append("<h1>Nav Forward View Gallery</h1>")

for group_name, task_names in TASK_MAP.items():
    html_parts.append(f"<h2>{group_name} ({len(task_names)} tasks)</h2>")
    html_parts.append("<div class='row'>")
    for task_name in task_names:
        img_file = f"{task_name}.png"
        img_path = os.path.join(IMG_PATH, img_file)
        if os.path.exists(img_path):
            html_parts.append(f"  <div class='item'>")
            html_parts.append(f"    <img src='{img_file}' alt='{task_name}'>")
            html_parts.append(f"    <div class='label'>{task_name}</div>")
            html_parts.append(f"  </div>")
        else:
            html_parts.append(f"  <div class='item missing'>")
            html_parts.append(f"    <img src='' alt='missing'>")
            html_parts.append(f"    <div class='label'>{task_name} (missing)</div>")
            html_parts.append(f"  </div>")
    html_parts.append("</div>")

html_parts.append("</body></html>")

with open(HTML_FILE, "w") as f:
    f.write("\n".join(html_parts))
print(f"HTML gallery saved to: {HTML_FILE}")

print("Done.")
