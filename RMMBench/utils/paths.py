"""
Centralized management of project paths (single source of truth).

Anywhere a path inside the project is needed, import it from this module
instead of using hardcoded absolute paths or relying on environment variables
(e.g. os.getenv("RMMBENCH_ROOT")). This way, even when switching machines or
containers, as long as the directory structure is unchanged, no path-related
code needs to be modified at all.

Usage example:
    from RMMBench.utils.paths import PROJECT_ROOT, REPO_ROOT

    scene_config = PROJECT_ROOT / "configs" / "robocasa_scenes_config" / "robocasa_scene_config.json"
"""

from pathlib import Path

# __file__          = .../RMMBench/RMMBench/utils/paths.py
# .resolve()        -> Resolves to an absolute path and handles symlinks
# .parent           = .../RMMBench/RMMBench/utils/
# .parent.parent    = .../RMMBench/RMMBench/          <- PROJECT_ROOT (Python package root)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Repository root: .../RMMBench
REPO_ROOT = _PROJECT_ROOT.parent

# Project (package) root: .../RMMBench/RMMBench
PROJECT_ROOT = _PROJECT_ROOT
