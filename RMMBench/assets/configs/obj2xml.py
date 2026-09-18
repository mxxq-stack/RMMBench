import subprocess
from pathlib import Path

# 1. Define the [parent directory] containing all tool shelf models
base_dir = Path("/Users/lh/work/vlabench_etc/VLABench/VLABench/assets/obj/meshes/containers/table_shelf")


def main():
    # Make sure the directory exists
    if not base_dir.exists() or not base_dir.is_dir():
        print(f"❌ 错误: 找不到目录 {base_dir}")
        return

    # 2. Iterate over all subfolders under the parent directory
    # .iterdir() gets everything in the directory, .is_dir() filters out folders
    target_dirs = [d for d in base_dir.iterdir() if d.is_dir()]

    if not target_dirs:
        print(f"⚠️ 在 {base_dir} 下没有找到任何子文件夹。")
        return

    for target_dir in target_dirs:
        folder_name = target_dir.name

        print("-" * 50)
        print(f"⏳ 开始处理: {folder_name}")
        print(f"路径: {target_dir}")

        # 3. Build the command list to execute
        command = [
            "obj2mjcf",
            "--obj-dir", str(target_dir),
            "--save-mjcf",
            "--compile-model",
            "--decompose",
            "--overwrite"
        ]

        # 4. Run the command
        try:
            # check=True raises an exception when the command fails (non-zero exit code)
            # capture_output=True lets us capture the exact error text on failure
            result = subprocess.run(
                command,
                check=True,
                text=True,
                capture_output=True
            )
            print(f"✅ 成功生成 MJCF: {folder_name}")

        except subprocess.CalledProcessError as e:
            print(f"❌ 转换失败: {folder_name}")
            print(f"🔴 详细报错信息:\n{e.stderr}")

    print("-" * 50)
    print("🎉 所有文件夹批量处理完毕！")


if __name__ == "__main__":
    main()