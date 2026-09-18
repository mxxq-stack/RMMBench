import os
import sys
import open3d
import cv2
from dm_control import viewer

from VLABench.envs import load_env
from VLABench.tasks import *
from VLABench.robots import *
import json

from VLABench.utils.utils import extract_containers,get_geom_ids_by_prefix,get_entity_mask_from_seg,get_placement_top_down_view

from pathlib import Path
import traceback
from datetime import datetime

# Add project root to path
sys.path.insert(0, "/Users/lh/work/vlabench_etc/VLABench")

from VLABench.configs.constant import name2class_xml
import VLABench.tasks.components as components

# Configuration
TASK_CONFIG_PATH = "/Users/lh/work/vlabench_etc/VLABench/VLABench/configs/task_config.json"
SAVE_ROOT = "/Users/lh/work/vlabench_etc/VLABench/assets_sorting/common"
SKIPPED_DIR = "/Users/lh/work/vlabench_etc/VLABench/assets_sorting/skipped"
TASK_NAME = "test_texture"
TIME_LIMIT = 1000

# Container classes to exclude (when False) or include (when True)
CONTAINER_CLASSES = {
    'CommonContainer', 'FlatContainer', 'Plate', 'Vase', 'PlaceMat',
    'Stove', 'Shelf', 'CoffeeMachine', 'Microwave', 'Fridge',
    'Juicer', 'TubeStand', 'Mirrors', 'CuttingBoard',
    'ContainerWithDrawer', 'BilliardTable', 'Table'
}
#Counter

# Set to True to process only containers, False to process non-containers
PROCESS_CONTAINERS = True

# Tools holder to exclude
EXCLUDED_CATEGORIES = {'tools_holder'}


def get_class_name(cls):
    """Get class name from components.XXX"""
    if hasattr(cls, '__name__'):
        return cls.__name__
    return str(cls).split('.')[-1]


def extract_category_and_path(obj_name, value):
    """Extract category and subcategory from xml path"""
    # We inspect the source code representation of value[1] to extract the path
    # without executing the function
    
    path_str = str(value[1]) if len(value) > 1 else ""
    
    # Extract path from get_object_list(os.path.join(xml_root, "...")) or direct string
    import re
    
    # Try to find the path in quotes
    matches = re.findall(r'["\'](obj/meshes/[^"\']+)["\']', path_str)
    
    if not matches:
        # Fallback: try other patterns
        matches = re.findall(r'obj/meshes/[\w/_.-]+', path_str)
    
    if matches:
        path_part = matches[0]
        parts = path_part.split("/")
        
        # Remove .xml at the end if present
        if parts[-1].endswith(".xml"):
            parts = parts[:-1]
        
        if len(parts) >= 3:
            category = parts[2]  # e.g., "fruit", "ingredient", "tablewares"
            
            # Extract base object name (without _0, _1, etc. suffix)
            base_name = obj_name
            if '_' in obj_name and obj_name.split('_')[-1].isdigit():
                base_name = '_'.join(obj_name.split('_')[:-1])
            
            # Build subdirs: use base_name as subdirectory
            subdirs = [base_name]
            
            return category, subdirs
    
    # Fallback: infer from obj_name patterns
    fruit_names = {"apple", "apricot", "avocado", "banana", "cantaloupe", 
                   "cherry", "coconut", "dates", "grapes", "kiwi", "lemon",
                   "lime", "mango", "orange", "peach", "pear", "pineapple",
                   "pomegranate", "raspberry", "tangerine", "watermelon"}
    
    ingredient_names = {"artichoke", "bell_pepper", "broccoli", "carrot",
                        "cauliflower", "celery", "corn", "cucumber",
                        "eggplant", "garlic", "ginger", "leek", "lettuce",
                        "mushroom", "onion", "parsley", "peas", "potato",
                        "pumpkin", "radish", "spinach", "squash", "tomato",
                        "turnip", "zucchini"}
    
    # Extract base name for fallback
    base_name = obj_name
    if '_' in obj_name and obj_name.split('_')[-1].isdigit():
        base_name = '_'.join(obj_name.split('_')[:-1])
    
    if any(obj_name.startswith(f) for f in fruit_names):
        return "fruit", [base_name]
    elif any(obj_name.startswith(i) for i in ingredient_names):
        return "ingredient", [base_name]
    elif obj_name.startswith("bottle_opener") or obj_name.startswith("knife") or obj_name.startswith("spoon"):
        return "tablewares", [base_name]
    elif obj_name.startswith("book") or obj_name.startswith("cotton") or obj_name.startswith("one_hundred"):
        return "book", [base_name]
    
    return None, None


def get_save_path(obj_name, category, subdirs):
    """Build save path for object image"""
    if category is None:
        category = "unknown"
    
    # Build directory path: SAVE_ROOT/category/subdir1/subdir2/...
    dir_parts = [SAVE_ROOT, category] + subdirs
    save_dir = os.path.join(*dir_parts)
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(save_dir, f"{obj_name}.png")
    return save_path


def load_task_config():
    """Load task config from JSON file"""
    with open(TASK_CONFIG_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_task_config(config):
    """Save task config to JSON file"""
    with open(TASK_CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)


def filter_objects(test_mode=False):
    """Filter objects: exclude containers, tools_holder, and base objects that have variants"""
    filtered = []
    
    # First pass: extract path strings from value[1]
    # value[1] can be: a string (single xml path), a list of strings (multiple variants), or a function call
    obj_paths = {}
    for obj_name, value in name2class_xml.items():
        path_val = value[1] if len(value) > 1 else ""
        
        # Handle different types
        if isinstance(path_val, str):
            # Direct string path - this is a variant
            obj_paths[obj_name] = path_val
        elif isinstance(path_val, list) and len(path_val) > 0:
            # List of paths - this is a base object with variants
            # Take the first path for comparison purposes
            obj_paths[obj_name] = path_val[0] if isinstance(path_val[0], str) else str(path_val[0])
        else:
            # Fallback: try to extract from string representation
            path_str = str(path_val)
            import re
            match = re.search(r'["\'](obj/meshes/[^"\']+)["\']', path_str)
            if match:
                obj_paths[obj_name] = match.group(1)
            else:
                obj_paths[obj_name] = ""
    
    # Second pass: find base objects that have variants
    # A base object has variants if its value[1] is a list (multiple paths)
    # OR if another object's path starts with its path
    base_with_variants = set()
    for obj_name, value in name2class_xml.items():
        path_val = value[1] if len(value) > 1 else ""
        # If value[1] is a list, it's a base object with variants
        if isinstance(path_val, list):
            base_with_variants.add(obj_name)
            continue
        
        # Also check path containment
        path = obj_paths.get(obj_name, "")
        if not path:
            continue
        for other_name, other_path in obj_paths.items():
            if obj_name != other_name and other_path.startswith(path) and len(other_path) > len(path):
                base_with_variants.add(obj_name)
                break
    
    for obj_name, value in name2class_xml.items():
        # Test mode: only process banana objects
        if test_mode and not obj_name.startswith("banana"):
            continue
        
        # Skip base objects that have variants (e.g., "banana" has "banana_0", so skip "banana")
        if obj_name in base_with_variants:
            continue
        
        cls = value[0]
        class_name = get_class_name(cls)
        
        # Check if object is container or not
        is_container = class_name in CONTAINER_CLASSES
        if PROCESS_CONTAINERS and not is_container:
            # Processing containers, skip non-containers
            continue
        if not PROCESS_CONTAINERS and is_container:
            # Processing non-containers, skip containers
            continue
        
        # Extract category from path
        category, subdirs = extract_category_and_path(obj_name, value)
        
        # Skip tools_holder category
        if category in EXCLUDED_CATEGORIES:
            continue
        
        # Skip if category is None (couldn't determine)
        if category is None:
            continue
        
        filtered.append((obj_name, value, category, subdirs))
    
    return filtered


def capture_object_image(obj_name, task_config):
    """Load environment and capture image for a single object"""
    # Modify seen_object
    task_config[TASK_NAME]["task"]["asset"]["seen_object"] = [obj_name]

    # Save config
    save_task_config(task_config)
    
    # Load environment
    env = load_env(TASK_NAME, robot="pandaomron", time_limit=TIME_LIMIT)
    
    # Reset environment
    env.reset()
    
    # Get observation
    rgb_data = env.get_observation()
    
    # Get top view (camera 2, forward)
    img = cv2.cvtColor(rgb_data["rgb"][2], cv2.COLOR_BGR2RGB)

    #wrist cam
    # img = cv2.cvtColor(rgb_data["rgb"][3], cv2.COLOR_BGR2RGB)

    # Close environment
    env.close()
    
    return img


def main():
    # Create directories
    os.makedirs(SAVE_ROOT, exist_ok=True)
    os.makedirs(SKIPPED_DIR, exist_ok=True)
    
    # Load task config
    task_config = load_task_config()
    
    # Filter objects (process all objects)
    objects = filter_objects(test_mode=False)
    obj_type = "container" if PROCESS_CONTAINERS else "non-container"
    print(f"Found {len(objects)} {obj_type} objects to process")
    
    # Process each object
    for i, (obj_name, value, category, subdirs) in enumerate(objects):
        print(f"[{i+1}/{len(objects)}] Processing: {obj_name} (category: {category})")
        
        try:
            # Get save path
            save_path = get_save_path(obj_name, category, subdirs)
            
            # Check if file already exists
            if os.path.exists(save_path):
                print(f"  Skipped (already exists): {save_path}")
                continue
            
            # Capture image
            img = capture_object_image(obj_name, task_config)
            
            # Save image
            cv2.imwrite(save_path, img)
            print(f"  Saved: {save_path}")
            
        except Exception as e:
            error_msg = str(e)
            tb = traceback.format_exc()
            print(f"\n  ERROR processing {obj_name}: {error_msg}")
            
            # Save error info to skipped directory
            skipped_path = os.path.join(SKIPPED_DIR, f"{obj_name}.txt")
            with open(skipped_path, 'w', encoding='utf-8') as f:
                f.write(f"Object: {obj_name}\n")
                f.write(f"Category: {category}\n")
                f.write(f"Error: {error_msg}\n\n")
                f.write(f"Traceback:\n{tb}\n")
            print(f"  Error info saved to: {skipped_path}")
            print(f"  Continuing to next object...\n")
            continue
    
    print(f"\nDone! Processed {len(objects)} objects successfully.")


if __name__ == "__main__":
    main()
