import shutil
from pathlib import Path

# Define your root directory
base_dir = Path("/Users/lh/work/vlabench_etc/VLABench/VLABench/assets/obj/meshes/containers/table_shelf")


def main():
    if not base_dir.exists():
        print(f"❌ 找不到目录: {base_dir}")
        return

    # Iterate over all subfolders under tools_holder (e.g. tool_holder_0, tool_holder_1...)
    for parent_dir in base_dir.iterdir():
        if parent_dir.is_dir():

            # Build the expected nested inner-folder path
            # e.g. .../tool_holder_0/tool_holder_0
            child_dir = parent_dir / parent_dir.name

            # If the nested structure really exists
            if child_dir.exists() and child_dir.is_dir():
                print(f"⏳ 正在处理: {parent_dir.name}")

                # Set up a temporary staging directory name, placed under base_dir
                temp_dir = base_dir / f"{parent_dir.name}_temp"

                try:
                    # 1. Move the inner folder out next to the original as a temporary folder
                    shutil.move(str(child_dir), str(temp_dir))

                    # 2. Fully remove the outer folder (along with any leftover files inside)
                    shutil.rmtree(parent_dir)

                    # 3. Rename the temporary folder to the outer folder's name
                    temp_dir.rename(parent_dir)

                    print(f"✅ 成功替换: {parent_dir.name}")

                except Exception as e:
                    print(f"❌ 处理 {parent_dir.name} 时发生错误: {e}")
            else:
                # If there is no nested structure, skip it
                pass

    print("-" * 50)
    print("🎉 批量替换整理完毕！")


if __name__ == "__main__":
    main()