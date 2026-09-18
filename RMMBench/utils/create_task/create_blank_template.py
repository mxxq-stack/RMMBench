from RMMBench.utils.paths import PROJECT_ROOT
import os
import re




def create_rmmbench_task_placeholder(task_name, task_list, base_path="/Users/lh/work/RMMBench"):
    """
    Automate creation of empty task files and registration of the config mapping, appending the mapping
    to the end of the name2config dictionary.
    """
    # 1. Define the paths
    robocasa_task_dir = os.path.join(base_path, "RMMBench/tasks/robocasa_task")
    robocasa_init_path = os.path.join(robocasa_task_dir, "__init__.py")
    config_init_path = os.path.join(base_path, "RMMBench/configs/__init__.py")
    new_task_file = os.path.join(robocasa_task_dir, f"{task_name}.py")

    # --- Step 1: create an empty task_name.py ---
    if not os.path.exists(new_task_file):
        with open(new_task_file, "w") as f:
            pass
        print(f"Successfully created empty file: {new_task_file}")
    else:
        print(f"File {task_name}.py already exists.")

    # --- Step 2: update the imports in robocasa_task/__init__.py ---
    import_line = f"from RMMBench.tasks.robocasa_task.{task_name} import *\n"
    with open(robocasa_init_path, "r") as f:
        robocasa_init_content = f.read()

    if import_line not in robocasa_init_content:
        with open(robocasa_init_path, "a") as f:
            if robocasa_init_content and not robocasa_init_content.endswith('\n'):
                f.write('\n')
            f.write(import_line)
        print(f"Added import to {robocasa_init_path}")

    # --- Step 3: update the end of the name2config dictionary in configs/__init__.py ---
    with open(config_init_path, "r") as f:
        content = f.read()

    # Check whether the key already exists
    if f'"{task_name}":' not in content:
        # Core logic: find the last closing brace before the end of the name2config dictionary
        # Use a regex to match the last '}', making sure it looks like the dictionary's ending
        # [ \t]* matches any possible indentation
        pattern = r"(name2config\s*=\s*\{.*?)(\n\s*\})"

        # Build the new entry: note the leading indentation and the trailing comma
        new_entry = f'\n    "{task_name}": {task_list},'

        # Use re.DOTALL to match across lines
        if re.search(pattern, content, re.DOTALL):
            # \1 is the dictionary body, \2 is the trailing newline + closing brace
            updated_content = re.sub(pattern, f"\\1{new_entry}\\2", content, flags=re.DOTALL)

            with open(config_init_path, "w") as f:
                f.write(updated_content)
            print(f"Successfully appended '{task_name}' to the end of name2config.")
        else:
            # If the regex match fails (e.g., an extremely unusual format), fall back to a simple replacement
            print("Regex match failed, trying fallback...")
            updated_content = content.replace("}", f'    "{task_name}": {task_list},\n}}', 1)
    else:
        print(f"Mapping for '{task_name}' already exists.")


# --- Usage example ---
if __name__ == "__main__":
    # Define the task mapping: {task_name: task_list}
    RMMBENCH_PATH = PROJECT_ROOT
    tasks_to_process = {
        # Option 1: pre-meal preparation (already provided)
        # "wash_fruit_tidy_table": ["wash_fruit", "wash_fruit_tidy_table"],

        # Option 2: breakfast preparation
        "cool_milk_serve_bread": ["cool_milk_in_sink", "cool_milk_serve_bread_on_tray"],

        # Option 3: kitchen-waste handling and storage
        "wash_meat_organize_cheese": ["wash_meat_in_sink", "wash_meat_organize_cheese_on_tray"],

        # Option 4: post-meal cleanup and returning seasonings
        "wash_cutlery_return_seasoning": ["wash_knives_forks_in_sink", "wash_cutlery_return_seasoning_to_tray"],

        # Option 5: afternoon tea time
        "ice_juice_arrange_cake": ["ice_juice_bottle_in_sink", "ice_juice_arrange_cake_on_tray"],

        # Option 6: party preparation
        "ice_wine_move_snacks": ["ice_wine_bottle_in_sink", "ice_wine_move_snacks_to_tray"],

        # Option 7: recreation-area tidying
        "clean_billiards_stack_blocks": ["clean_billiards_in_sink", "clean_billiards_stack_blocks_on_tray"]
    }
    # Batch loop invocation
    for name, list_items in tasks_to_process.items():
        vla_root = PROJECT_ROOT
        print("vla_root:", vla_root)
        exit()
        create_rmmbench_task_placeholder(name, list_items,RMMBENCH_PATH)
        print(f"已生成任务占位符: {name}")
