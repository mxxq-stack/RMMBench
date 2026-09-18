import open3d as o3d
import os
import mediapy
import argparse
import cv2
import traceback
import json
from dm_control import viewer
from tqdm import tqdm
from datetime import datetime
from scipy.spatial.transform import Rotation as R
from RMMBench.robots import Franka
from RMMBench.configs import name2config
from RMMBench.configs.task_class import TASK_CLASS
from RMMBench.configs.prompt.vlm_prompt import VLAMessageHandler
from RMMBench.envs import load_env
from RMMBench.tasks import *
from RMMBench.utils.skill_lib import SkillLib
from RMMBench.utils.utils import is_target_out_of_reach, get_logger, calculate_mask_bbox_iou, CAMERA_NAME_MAP
from RMMBench.vlm_evaluation.evaluator.debug_visualizer import DebugVisualizer
from RMMBench.vlm_evaluation.evaluator.metrics import (
    build_rs_tracker, get_objective_score, compute_rr_act, compute_crnp,save_results
)
from RMMBench.tasks.condition import get_leaf_met_progress
from RMMBench.utils.entity_mapping import get_entity_from_bbox, BODY_LEVEL_TARGETS
from RMMBench.utils.navigation_utils import get_head_camera_navigation_view, draw_rays_on_image

os.environ["MUJOCO_GL"] = "egl"
from multiprocessing import Pool
import time




def get_args():
    parser = argparse.ArgumentParser(description='Interactive VLM evaluation with topology-based assessment')
    # VLM related
    parser.add_argument('--vlm-url', default="https://api.deepseek.com/chat/completions", type=str,
                        help='VLM server URL (OpenAI-compatible chat/completions endpoint)')
    parser.add_argument('--api-key', default="sk-f3666b93bf824c38ad7c083f2f4a9506", type=str,
                        help='API key for VLM service (e.g. DeepSeek). Falls back to env DEEPSEEK_API_KEY if not provided')
    parser.add_argument('--vlm-backend', default="deepseek-flash", type=str)
    # prompt_format is fixed to "openai" (vlm_client sends requests via the official OpenAI-compatible protocol).
    # The legacy prompt_image_map (custom gateway format) is kept inside the handler but no longer exposed as a CLI argument
    parser.add_argument('--history-maxlen', default=2, type=int, help='History maxlen for VLM handler')
    # Task and episode config
    parser.add_argument('--episode-config',
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                                             'RMMBench', 'vlm_evaluation', 'configs', 'episodes',
                                             'nav_episodes.json'),
                        type=str, help='Path to episode config JSON file (e.g. test_eval_pick.json)')
    parser.add_argument('--task-name', default="navigate_composite_to_wash_fruit", type=str,
                        help='Task name to evaluate directly (when episode-config is None, loads task via random init)')
    parser.add_argument('--tasks', nargs='+', default=None,
                        help='Specific tasks to evaluate. If None, evaluate all tasks in episode config')
    parser.add_argument('--config-idx', default=None, type=int,
                        help='Specific config index to evaluate for each task. If None, evaluate all configs')
    # Evaluation settings
    parser.add_argument('--max-skills-num', default=15, type=int, help='Max number of skills per episode')
    parser.add_argument('--robot', default="pandaomron", type=str, help='Robot name')
    parser.add_argument('--select-grasp', default=False, help='Whether to perform a grab selection when picking')
    parser.add_argument('--use-graspnet', action='store_true', default=False,
                        help='Enable GraspNet candidate grasping during pick (passed into SkillLib.pick)')
    parser.add_argument('--early-stop', action="store_true", default=False, help='Early stop when skill fails')
    parser.add_argument('--visual-rays', action="store_true", default=False,
                        help='Enable visual ray assistance: only takes effect for composite_navigation tasks; '
                             'when the head camera faces forward, navigation rays are overlaid on its '
                             'observation image; when disabled, all tasks use ordinary observations')
    parser.add_argument('--seed', default=42, type=int,
                        help='Random seed for reproducibility when not using episode_config. If None and episode_config is not provided, random initialization will be used.')
    parser.add_argument('--debug', action="store_true", default=False, help='Debug mode')
    # Save settings
    parser.add_argument('--save-dir',
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                                             'RMMBench', 'vlm_evaluation', 'results'),
                        type=str, help='Directory to save evaluation results')
    parser.add_argument('--save-interval', type=int, default=1, help='Interval for saving results')
    parser.add_argument('--save-video', action="store_true", default=True,
                        help='Save video for the first episode of each task')
    parser.add_argument('--num-workers', type=int, default=1,
                        help='Number of parallel workers for episode evaluation. 1 = serial, >1 = multiprocessing pool')
    args = parser.parse_args()

    # Fall back to the environment variable when api-key is not explicitly provided
    if args.api_key is None:
        args.api_key = os.environ.get("DEEPSEEK_API_KEY")

    return args


def load_episode_config(episode_config_path):
    """Load episode config from JSON file with category structure"""
    with open(episode_config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config


def _as_config_list(task_data):
    """Normalize a task config into a list: wrap a dict in a list, return a list as-is, treat any other type as invalid and return None."""
    if isinstance(task_data, dict):
        return [task_data]
    if isinstance(task_data, list):
        return task_data
    return None


# task_name -> (task_category, task_type): the single source of truth for classification is VLABench/configs/task_class.py
# manipulation is subdivided into strict / flexible; other categories (e.g. composite_navigation) have no subtype and are recorded as None
TASK_CLASS_INDEX = {}
for _task_type, _families in (TASK_CLASS.get("manipulation") or {}).items():
    for _task_names in _families.values():
        TASK_CLASS_INDEX.update({_name: ("manipulation", _task_type) for _name in _task_names})
for _category, _families in TASK_CLASS.items():
    if _category == "manipulation":
        continue
    for _task_names in _families.values():
        TASK_CLASS_INDEX.update({_name: (_category, None) for _name in _task_names})


def get_task_configs(episode_config, task_name):
    """Get the config list and classification for a given task.

    The episode config is in flat format: top-level keys are task names, values are configs (a list or a single dict).
    Classification is no longer inferred from the episode config; it is always provided by TASK_CLASS_INDEX;
    tasks not indexed return (None, None) for the caller to handle.

    Returns: (task_data, task_category, task_type)
    """
    return (_as_config_list(episode_config.get(task_name)), *TASK_CLASS_INDEX.get(task_name, (None, None)))


def get_all_tasks_from_config(episode_config):
    """Get all task names from the flat episode config (i.e. the top-level keys)"""
    return list(episode_config.keys())


def get_task_type(task_name, episode_config):
    """
    Get the task's category, always provided by TASK_CLASS_INDEX.

    Returns:
        tuple: (task_category, task_type)
            - task_category: 'manipulation', 'composite_navigation', or None
            - task_type: 'strict', 'flexible', or None
    """
    _, task_category, task_type = get_task_configs(episode_config, task_name)
    return task_category, task_type


def extract_skill_sequence_from_partials(partial_list):
    """
    Extract skill sequence from a list of partial functions
    Returns list of dict with 'skill_name' and 'params'
    """
    skill_seq = []
    for p in partial_list:
        skill_name = p.func.__name__ if hasattr(p, 'func') else str(p)
        params = p.keywords if hasattr(p, 'keywords') else {}
        # Remove env and vlm_handler from params for cleaner output
        # Also remove non-serializable objects like target_object_info
        clean_params = {}
        for k, v in params.items():
            if k in ['env', 'vlm_handler', 'target_object_info']:
                continue
            # Convert numpy arrays to lists
            if isinstance(v, np.ndarray):
                clean_params[k] = v.tolist()
            # Convert numpy scalars to Python types
            elif isinstance(v, (np.integer, np.floating)):
                clean_params[k] = v.item()
            # Only keep basic serializable types
            elif isinstance(v, (str, int, float, bool, list, dict, type(None))):
                clean_params[k] = v
            else:
                # Skip other non-serializable objects
                clean_params[k] = str(v)
        skill_seq.append({
            'skill_name': skill_name,
            'params': clean_params
        })
    return skill_seq


def set_random_seed(seed):
    """Set random seed for reproducibility"""
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)


def save_skill_images(img_dir, step_idx, action, vlm_img_input, bbox=None, pts=None, viz=None, cam_view=None):
    """Save skill-related images; only called when action is pick or place"""
    if img_dir is None:
        return

    if action == "pick":
        # Select the viewpoint image based on cam_view, defaulting to wrist
        cam_idx = CAMERA_NAME_MAP.get(cam_view, CAMERA_NAME_MAP["wrist"])
        cam_label = cam_view if cam_view in ("wrist", "head") else "wrist"
        if bbox and pts is not None:
            img_cam = vlm_img_input[cam_idx].copy()
            cv2.polylines(img_cam, [pts.astype(int)], isClosed=True, color=(0, 0, 255), thickness=2)
            cv2.imwrite(os.path.join(img_dir, f"img_{step_idx}_{cam_label}_{action}_pts.png"), img_cam)
        if viz is not None:
            for index, v in enumerate(viz):
                cv2.imwrite(os.path.join(img_dir, f"grasp_viz_{index}_action{step_idx}.png"), v)

    elif action == "place":
        # Head view; draw the box when pts is available
        if pts is not None:
            img_head = vlm_img_input[5].copy()
            cv2.polylines(img_head, [pts.astype(int)], isClosed=True, color=(0, 0, 255), thickness=2)
            cv2.imwrite(os.path.join(img_dir, f"img_{step_idx}_head_{action}.png"), img_head)


def save_mask(img_dir, step_idx, action, img_input, pts=None, mask=None):
    if mask is not None:
        vis_mask = (mask * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(img_dir, f"img_{step_idx}_wrist_mask_{action}.png"), vis_mask)


def get_ray_assisted_observation(env):
    """Get multi-camera observations with ray assistance for composite navigation tasks.

    Process:
    1. Make the robot fully transparent and compute unobstructed navigation rays;
    2. When the head camera faces forward, restore visibility and overlay the rays on its observation image;
       otherwise fall back to a transparent observation where only the base + link6 remain visible
       (the original moveforward observation approach);
    3. Finally restore full rendering so the env returned to the caller is in a normal state.

    Args:
        env: The simulation environment

    Returns:
        list[np.ndarray]: RGB observation images from the 6 camera views (same format as regular
        observations, can be fed directly to the VLM)
    """
    # 1. Fully transparent so point cloud computation is unobstructed by the robot
    env.robot.set_transparent_geom(physics=env.physics, alpha=0.0)
    env.step()

    head_body = env.robot.get_body_info("cam_head", env.physics)
    base_info = env.robot.get_link_base_info(env.physics)
    if_forward = head_body["euler"][2] - base_info["euler"][2]  # Check whether the head faces forward
    if abs(if_forward) < 0.01:
        # 2. Compute rays from the robot-free image, restore full visibility, take the observation,
        # and overlay the rays on the head camera image
        _, ray_info = get_head_camera_navigation_view(env)
        env.robot.restore_opacity_geom(env.physics)
        env.step()

        rgb_data = env.get_observation()
        vlm_img_input = [cv2.cvtColor(rgb_data["rgb"][i], cv2.COLOR_BGR2RGB) for i in range(6)]
        vlm_img_input[5] = draw_rays_on_image(vlm_img_input[5], ray_info)
    else:
        # head not facing forward: fall back to a transparent observation where only the base + link6 are visible
        env.robot.set_transparent_geom(physics=env.physics, alpha=0.0, exclude_bodies=["wheeled_base", "link6"])
        env.step()
        rgb_data = env.get_observation()
        vlm_img_input = [cv2.cvtColor(rgb_data["rgb"][i], cv2.COLOR_BGR2RGB) for i in range(6)]

    # 3. Restore full visibility so subsequent rendering/video saving works normally
    env.robot.restore_opacity_geom(env.physics)
    env.step()
    return vlm_img_input


def save_episode_video(observations, video_dir, task_name, task_success):
    """Save the episode replay video (skips directly when video_dir is empty or no valid observations exist).

    - Main video: each step stitches 4 views into a 2x2 grid (cameras 2/4 on the top row, 3/5 on the bottom row),
      filename includes task name / success-failure / timestamp;
    - forward.mp4: single forward view from camera 2 (note the filename is fixed; multiple evaluations in the
      same directory will overwrite each other).
    """
    if video_dir is None or len(observations) == 0:
        return

    frames = []
    fs = []
    for o in observations:
        frame = np.vstack([
            np.hstack([o["rgb"][2], o["rgb"][4]]),
            np.hstack([o["rgb"][3], o["rgb"][5]])
        ])
        frames.append(frame)
        fs.append(o["rgb"][2])

    timename = time.strftime("%H.%M.%S", time.localtime())
    task_success_str = "success" if task_success else "fail"

    os.makedirs(video_dir, exist_ok=True)
    mediapy.write_video(
        os.path.join(video_dir, f"{task_name}_{task_success_str}_{timename}.mp4"),
        frames, fps=7
    )
    mediapy.write_video(
        os.path.join(video_dir, f"forward.mp4"),
        fs, fps=7
    )
    print(f"  Video saved to: {video_dir}")


def evaluate_single_episode(args, task_name, episode_config, config_idx, logger, task_type, seed=None, video_dir=None,
                            img_dir=None, task_category='manipulation'):
    """
    Evaluate a single episode with interactive VLM

    Args:
        seed: Random seed for this episode. If None and episode_config is None, will use random init.
        video_dir: Directory to save video (only for first episode). If None, no video is saved.
        img_dir: Directory to save images. If None, no images are saved.
        task_category: Task category (manipulation/composite_navigation) for selecting appropriate prompt

    Returns:
        dict: Evaluation result containing:
            - vlm_skill_sequence: List of skills executed by VLM
            - task_success: Whether the task was completed successfully
            - expert_skill_sequence: Expert skill sequence for comparison
    """
    result = {
        'task_name': task_name,
        'config_idx': config_idx,
        'vlm_skill_sequence': [],
        'expert_skill_sequence': [],
        'task_success': False,
        'error': None
    }

    # Initialize variables that need to persist for video saving (outside try block)
    observations = []
    instruction = None

    # Set random seed if provided and no episode_config
    if seed is not None and episode_config is None:
        set_random_seed(seed)

    # Load environment
    if episode_config is not None:
        # Use deterministic episode config
        # task_name="navigate_to_wash_fruit_cook_steak_unordered",
        env = load_env(
            task_name,
            episode_config=episode_config,
            robot=args.robot
        )
    else:
        # Use random initialization (seed-controlled if seed is set)
        env = load_env(
            task_name,
            robot=args.robot,
        )
    env.reset()

    # RS metric: build the tracker at episode start, feed on_skill per skill, settle at episode end
    rs_tracker = build_rs_tracker(env)

    # === Diagnostic: print geom index range baseline (for cross-checking the index in IndexError) ===
    # diagnose_render_index_range(env)
    # ============================================================
    # Note: task_category is passed in by the caller (manipulation / composite_navigation), no longer hardcoded

    # Initialize VLM handler

    handler = VLAMessageHandler(
        env=env,
        history_maxlen=args.history_maxlen,
        img_size=(480, 480),
        vlm_url=args.vlm_url,
        api_key=args.api_key,
        vlm_backend=args.vlm_backend,
        prompt_format="openai",
        task_category=task_category
    )

    if img_dir is not None:
        debug_full_dir = os.path.join(os.path.dirname(img_dir), "debug_full")
        handler.enable_debug_full_save(debug_full_dir)

    # Get task information
    iou_list = []
    grasp_task = env.task.config_manager.grasp_task
    all_entities = env.task.component_name_list
    print("all entities:", all_entities)
    all_cleaned_entities = env.task.cleaned_component_name_list

    # all_cleaned_entities.append("fridge_handle")
    # all_cleaned_entities.append("knob")

    instruction = env.task.get_instruction()

    target_entity = env.task.config_manager.target_entity
    target_container = env.task.config_manager.target_container

    robot_pos = env.task.robot.get_link_base_info(env.physics)["position"]
    # Get expert skill sequence for topology comparison

    # Initialize robot info
    # init_ee_pos, init_ee_quat = list(env.robot.get_end_effector_pos(env.physics)), env.robot.get_end_effector_quat(
    #     env.physics)
    if env.task.robot_params is not None:
        robot_init_info = env.task.robot_params
    else:
        robot_init_info = env.robot.robot_init_info

    # Setup skill library
    allowed_skills = {
        'pick': lambda **kwargs: partial(SkillLib.pick, env=env, **kwargs),
        'place': lambda **kwargs: partial(SkillLib.place, env=env, **kwargs),
        'lift': lambda **kwargs: partial(SkillLib.lift, env=env, gripper_state=np.zeros(2), **kwargs),
        'pull': lambda **kwargs: partial(SkillLib.pull, env=env, gripper_state=np.zeros(2), **kwargs),
        'push': lambda **kwargs: partial(SkillLib.push, env=env, gripper_state=np.zeros(2), **kwargs),
        'close_gripper': lambda **kwargs: partial(SkillLib.close_gripper, env=env, **kwargs),
        'open_gripper': lambda **kwargs: partial(SkillLib.open_gripper, env=env, **kwargs),
        'observe': lambda **kwargs: partial(SkillLib.observe, env=env, **kwargs),
        'open_door': lambda **kwargs: partial(SkillLib.open_door, env=env, **kwargs),
        'close_door': lambda **kwargs: partial(SkillLib.close_door, env=env, **kwargs),
        'moveforward': lambda **kwargs: partial(SkillLib.moveforward, env=env, **kwargs),
        'rotate_right': lambda **kwargs: partial(SkillLib.rotate_right, env=env, **kwargs),
        'rotate_left': lambda **kwargs: partial(SkillLib.rotate_left, env=env, **kwargs),
        'look_right': lambda **kwargs: partial(SkillLib.look_right, env=env, **kwargs),
        'look_left': lambda **kwargs: partial(SkillLib.look_left, env=env, **kwargs),
        'look_forward': lambda **kwargs: partial(SkillLib.look_forward, env=env, **kwargs),
        'rotate_knob': lambda **kwargs: partial(SkillLib.rotate_knob, env=env, **kwargs),
        'open_sink': lambda **kwargs: partial(SkillLib.open_sink, env=env, **kwargs),
        'close_sink': lambda **kwargs: partial(SkillLib.close_sink, env=env, **kwargs),
        'end': lambda **kwargs: partial(SkillLib.end, env=env, **kwargs),
        'recall': lambda **kwargs: partial(SkillLib.recall, handler=handler, **kwargs),
    }

    # Interactive VLM evaluation loop
    vlm_skill_seq = []
    last_action = None
    n = args.max_skills_num

    viz = None
    pts = None
    target_out_of_reach = False
    movedistance = 0
    height = 0.08

    # [RECALL] Maintain full per-step trajectory observations indexed by step for the recall skill to look back on.
    # Structure: {"step_0": [obs0, obs1, ...], "step_1": [...], ...}
    # Only written after non-recall steps execute (recall itself produces no real trajectory; see the main loop below).
    total_observations = {}
    start_time = time.time()

    while n > 0:
        step_idx = args.max_skills_num - n

        if args.visual_rays and task_category == "composite_navigation":
            vlm_img_input = get_ray_assisted_observation(env)

        else:
            vlm_img_input = [cv2.cvtColor(env.get_observation()["rgb"][i], cv2.COLOR_BGR2RGB) for i in range(6)]

        # Save observation images if img_dir is provided
        if img_dir is not None:
            frame = np.vstack(
                [np.hstack([vlm_img_input[4], vlm_img_input[2]]), np.hstack([vlm_img_input[3], vlm_img_input[5]])])
            cv2.imwrite(os.path.join(img_dir, f"step{step_idx}_four_frame.png"), frame)

        vlm_result = handler.request_vlm(
            img=vlm_img_input,
            n=step_idx,
            instruction=instruction,
            action=last_action,
            robot_init_info=robot_init_info,
            allow_entities=all_cleaned_entities,
            target_out_of_reach=target_out_of_reach,
            movedistance=movedistance
        )
        movedistance = 0
        # Only possibly True when picking
        target_out_of_reach = False

        if not vlm_result:
            logger.error("VLM response failed")
            result['error'] = "VLM response failed"
            break

        bbox = vlm_result["bbox_2d"]
        action = vlm_result["action"]
        target_object = vlm_result["target_object"]
        content = vlm_result["content"]
        pts = vlm_result["pts"]
        # Note: fridge_handle is now handled via the body_name mechanism and does not need conversion here
        last_action = action
        cam_view = vlm_result["cam_view"]
        cam_pick = CAMERA_NAME_MAP.get(cam_view, CAMERA_NAME_MAP["wrist"])  # default wrist

        # Record the skill
        skill_record = {
            'step': step_idx,
            'skill_name': action,
            'target': target_object,
            'bbox': bbox,
        }
        vlm_skill_seq.append(skill_record)

        # Execute skill
        if action not in allowed_skills:
            logger.warning(f"Unknown action: {action}")
            break

        skill_factory = allowed_skills[action]

        if action == "pick":
            # [DEBUG-BBOX-MAP] Original logic: match entity by target_object prefix (kept for debugging comparison)
            # matches = [item for item in all_entities if item.startswith(target_object)]
            # if not matches:
            #     logger.warning(f"No matching entity found for {target_object}")
            #     if args.early_stop:
            #         break
            #     n -= 1
            #     continue
            # match = matches[0]

            # [DEBUG-BBOX-MAP] New logic: map the bbox to a specific entity_name or body_name
            match = None
            if bbox is not None:
                match = get_entity_from_bbox(
                    env,
                    bbox=bbox,
                    target_entity_name=target_object,
                    cam_id=cam_pick,
                    min_ratio=0
                )
                print(f"[DEBUG-BBOX-MAP] bbox={bbox} -> match={match}")

            # [DEBUG-BBOX-MAP] fallback: if bbox mapping fails, fall back to the original prefix matching
            if match is None:
                matches = [item for item in all_entities if item.startswith(target_object)]
                if not matches:
                    logger.warning(f"No matching entity found for {target_object}")
                    if args.early_stop:
                        break
                    n -= 1
                    continue
                match = matches[0]
                print(f"[DEBUG-BBOX-MAP] fallback to prefix match: {match}")

            # === Resolve match (may be in "entity_name/body_name" format) ===
            match_entity = match
            match_body = None
            if "/" in match:
                match_entity, match_body = match.split("/", 1)

            # Write the entity name resolved from the bbox back into the skill record; preferred over the VLM's raw target when matching RRact params
            vlm_skill_seq[-1]['matched_target'] = match

            # 1. IoU computation
            iou, mask = calculate_mask_bbox_iou(env, match_entity, bbox, cam_pick, body_name=match_body)
            iou_list.append(iou)
            save_mask(img_dir, step_idx, action, vlm_img_input[cam_pick], pts, mask)

            # Reachability check
            target_out_of_reach = is_target_out_of_reach(env, match_entity, body_name=match_body)
            # handler.target_out_of_reach = target_out_of_reach
            if target_out_of_reach:
                obs, waypoints, stage_success, task_success = [None], [None], False, False
            else:
                # 5. Compute lift height: look up entities by entity name
                height = 0.08
                if match_entity in env.task.entities:
                    height = 2 * env.task.entities[match_entity].get_placement_height()

                pick_bbox = bbox if bbox else [0, 0, 0, 0]
                if bbox:
                    # pick_bbox = [bbox[0] - 10, bbox[1] - 5, bbox[2] + 5, bbox[3] + 5]
                    pick_bbox = bbox

                print("pick_bbox:", pick_bbox)
                print("match:", match)

                # 6. Decide whether this is an interactive part, determining whether to pass body_name or target_entity_name
                is_body_level = target_object is not None and any(
                    name in target_object.lower() for name in BODY_LEVEL_TARGETS
                )
                if is_body_level:
                    obs, waypoint, stage_success, task_success = skill_factory(
                        body_name=match,
                        use_graspnet=args.use_graspnet,
                    )()
                else:
                    obs, waypoint, stage_success, task_success = skill_factory(
                        target_entity_name=match_entity,
                        use_graspnet=args.use_graspnet,
                    )()

        elif action == "place":
            # [DEBUG-BBOX-MAP] Original logic: use target_object directly as target_container_name
            # place_target = target_object

            # [DEBUG-BBOX-MAP] New logic: map the bbox to a specific entity_name (sink is skipped, not in entities)
            place_target = target_object
            if bbox is not None and target_object != "sink":
                # No valid object inside the bbox returns None
                place_target = get_entity_from_bbox(
                    env,
                    bbox=bbox,
                    target_entity_name=target_object,
                    cam_id=5,  # head camera
                    min_ratio=0.05
                )

                print(f"[DEBUG-BBOX-MAP] place bbox={bbox} -> entity={place_target}")

            # [DEBUG-BBOX-MAP] fallback: if bbox mapping fails, fall back to the original target_object
            # if place_target is None:
            #     place_target = target_object
            #     print(f"[DEBUG-BBOX-MAP] fallback to target_object: {place_target}")
            # When bbox resolution succeeds, write the matched entity back into the skill record; preferred over the VLM's raw target when matching RRact params
            if place_target is not None:
                vlm_skill_seq[-1]['matched_target'] = place_target

            if place_target is not None:
                iou, mask = calculate_mask_bbox_iou(env, target_object, bbox, 5)
                save_mask(img_dir, step_idx, action, vlm_img_input[5], pts, mask)
            else:
                iou = 0
            iou_list.append(iou)

            # Skip skill execution when the target is out of reach
            target_out_of_reach = is_target_out_of_reach(env, place_target)
            # handler.target_out_of_reach = target_out_of_reach  # Sync to the handler so history backfill is correct
            if place_target is None:
                obs, waypoints, stage_success, task_success = [None], [None], False, False
            elif target_out_of_reach:
                obs, waypoints, stage_success, task_success = [None], [None], False, False
            else:
                obs, waypoint, stage_success, task_success = skill_factory(
                    target_container_name=place_target,
                    placement_th=height,
                    bbox=bbox)()


        elif action in ["moveforward", "rotate_right", "rotate_left"]:
            if action == "moveforward":
                obs, waypoint, stage_success, task_success, movedistance = skill_factory()()
            else:
                obs, waypoint, stage_success, task_success = skill_factory()()

            if bbox is not None:
                iou, mask = calculate_mask_bbox_iou(env, target_entity, bbox, 5)
                iou_list.append(iou)

        elif action in ["look_right", "look_left"]:
            print("target_object:", target_object)
            obs, waypoint, stage_success, task_success = skill_factory(
                yaw=int(target_object))()

        # =========================================================
        # [RECALL] recall skill branch:
        #   - target_object is the "step_x" emitted by the VLM, specifying which historical step to revisit
        #   - recall does not call env.step and produces no real observations; obs is always the [None] placeholder
        #   - stage_success indicates whether recall itself completed normally (whether it retrieved the
        #     trajectory and called the VLM successfully)
        #   - task_success is always False and does not affect task completion determination
        #   - keyframes_info holds keyframe info, written into handler.recall_results so that the next
        #     build_prompt_components round can display it among the historical steps
        # =========================================================
        elif action == "recall":
            obs, waypoint, stage_success, task_success, keyframes_info = skill_factory(
                total_observations=total_observations,
                step_idx=target_object,
            )()
            if keyframes_info is not None:
                handler.recall_results[target_object] = keyframes_info
            else:
                logger.warning(f"[RECALL] step={step_idx}: failed to recall target={target_object}, no keyframe info obtained")

        elif action == "rotate_knob":
            # target_object is the rotation angle in degrees, default 90
            angle = float(target_object) if target_object is not None else 90
            obs, waypoint, stage_success, task_success = skill_factory(angle=angle)()

        elif action in ("open_sink", "close_sink"):
            # Find the Sink entity in the scene
            sink_entity_name = None
            for name, entity in env.task.entities.items():
                if hasattr(entity, 'open_direction'):
                    sink_entity_name = name
                    break
            if sink_entity_name is None:
                logger.warning(f"No Sink entity found for {action}")
                obs, waypoint, stage_success, task_success = [None], [None], False, False
            else:
                obs, waypoint, stage_success, task_success = skill_factory(target_entity_name=sink_entity_name)()

        elif action == "end":
            obs, waypoint, stage_success, task_success = skill_factory()()
            if bbox is not None:
                iou, mask = calculate_mask_bbox_iou(env, target_entity, bbox, 3)
                iou_list.append(iou)
            rs_tracker.on_skill("end", None)
            break

        else:
            obs, waypoint, stage_success, task_success = skill_factory()()

        # RS metric: feed once after the skill actually executes; pick passes the bbox-mapped match (supports the raw "entity/body" string).
        # Not fed when: not actually executed (obs is the [None] placeholder), pure-query recall, or pick that failed to map to an entity
        if action != "recall" and not (isinstance(obs, list) and len(obs) == 1 and obs[0] is None):
            if action != "pick" or match is not None:
                rs_tracker.on_skill(action, match if action == "pick" else None)

        # Save skill images uniformly
        if action in ("pick", "place"):
            save_skill_images(img_dir, step_idx, action, vlm_img_input, bbox=bbox, pts=pts, viz=viz, cam_view=cam_view)

        # [RECALL] Maintain total_observations: only non-recall steps have real trajectory observations;
        # recall's own obs is the [None] placeholder and must not overwrite/pollute total_observations.
        if action != "recall":
            total_observations[f"step_{step_idx}"] = obs

        # [RECALL] The original logic only accumulated observations when video_dir was not None;
        # now always executed: recall steps need to insert the [None] placeholder into observations
        # to keep the step counter n aligned with the observations list (whether or not video is saved).
        # The recall branch's obs is always [None]; other branches produce real observation lists;
        # None placeholders are filtered out uniformly before saving the video (see the video saving logic below).
        observations.extend(obs)

        # Check early stop
        if args.early_stop and not stage_success:
            logger.info(f"Skill {action} failed, early stopping")
            break

        n -= 1

    result['vlm_skill_sequence'] = vlm_skill_seq
    result['task_success'] = env.task.task_success if hasattr(env.task, 'task_success') else False
    result['expert_skill_sequence'] = extract_skill_sequence_from_partials(
        env.task.get_expert_skill_sequence(env.physics))
    result['task_category'] = task_category if task_category else 'manipulation'
    result['task_type'] = task_type

    # Compute completion_rate: actual completion based on conditions
    # score: combined condition score over all leaves; details: per-leaf breakdown (process data stored in result
    # for auditing the CRm/CRn split)
    score, details = get_leaf_met_progress(env.task.conditions, env.physics)

    # score is not stored separately in result: it duplicates metrics['cr_manipulation'] and can be recomputed from details
    result['condition_leaf_details'] = details

    # Compute IoU (iou_list kept in result as process data)
    iou_score = get_objective_score(iou_list)
    result['iou_list'] = iou_list

    # RS metric settlement: called once at episode end (internally closes out if end was missed); segments kept in result as process data
    rs_value, rs_segments = rs_tracker.result()
    result['rs_segments'] = rs_segments
    print(f"[RS] value: {rs_value}")
    for seg in rs_segments:
        print(f"[RS] segment: {seg}")

    # ===== metrics: fixed seven keys (sr / path_completion / cr_manipulation / cr_navigation /
    # ee / rr_act / iou / rs); inapplicable metrics are set to None so a failed episode's schema has no missing keys =====
    vlm_seq = result['vlm_skill_sequence']
    expert_seq = result['expert_skill_sequence']
    task_success = result['task_success']
    result['condition_leaf_score'] = score

    has_seqs = bool(vlm_seq) and bool(expert_seq)

    # EE: expert skill count / VLM skill count on success, 0 on failure
    ee = (len(expert_seq) / len(vlm_seq)) if (task_success and vlm_seq and expert_seq) else None

    metrics = {
        'sr': 1 if task_success else 0,
        'path_completion': None,
        'cr_manipulation': float(score),
        'cr_navigation': None,
        'ee': ee,
        'rr_act': 0.0,
        'iou': iou_score,
        'rs': rs_value,
    }

    if result['task_category'] == 'composite_navigation':
        # CRm / CRn / NP: read purely from the env side, independent of skill sequences; cr_manipulation is overwritten by the stratified values
        metrics.update(compute_crnp(env))
        # RRact: filter out navigation skills first, then match
        if has_seqs:
            metrics['rr_act'] = compute_rr_act(vlm_seq, expert_seq, task_type, filter_nav=True)
    elif has_seqs:
        # manipulation (strict matches directly; flexible enumerates combinations without observe and takes the max, distinguished internally)
        metrics['rr_act'] = compute_rr_act(vlm_seq, expert_seq, task_type)

    result['metrics'] = metrics
    print("metrics:", metrics)

    env.close()
    end_time = time.time()
    elapsed = end_time - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = elapsed % 60
    print(f"Elapsed: {hours}h {minutes}m {seconds:.6f}s")
    # [RECALL] Before saving the video, first remove the None placeholders inserted by recall steps
    # to avoid errors from o being None when indexing o["rgb"][...].
    observations = [o for o in observations if o is not None]

    # Save video if video_dir is provided (guard is inside the function)
    save_episode_video(observations, video_dir, task_name, result['task_success'])

    # Record final results

    return result


def resolve_evaluation_tasks(args):
    """Resolve the list of tasks to evaluate, determining the mode and scope of this evaluation run.

    Two mutually exclusive paths:
    - With --episode-config: load the deterministic config, optionally filter a task subset with --tasks;
    - Without --episode-config: single-task direct evaluation (--task-name) with random initialization.

    Args:
        args: Command-line argument object, must contain the episode_config / tasks / task_name fields

    Returns:
        tuple: (episode_config, task_list)
            - episode_config: dict or None (None means random-initialization mode)
            - task_list: list of task names to evaluate in this run
    """
    if args.episode_config is not None:
        episode_config = load_episode_config(args.episode_config)

        # Determine which tasks to evaluate
        if args.tasks is None:
            task_list = get_all_tasks_from_config(episode_config)
        else:
            task_list = []
            for task in args.tasks:
                if task in episode_config:
                    task_list.append(task)
                else:
                    task_series = None
                    for series, tasks in name2config.items():
                        if task in tasks:
                            task_series = series
                            break
                    if task_series is not None:
                        task_list.append(task)
                    else:
                        print(f"Warning: Task '{task}' not found in episode_config or name2config, skipping")
    else:
        # No episode config: use --task-name directly with random init
        if args.task_name is None:
            print("Error: --task-name is required when --episode-config is not provided")
            sys.exit(1)
        episode_config = None
        task_list = [args.task_name]
        print(f"No episode config provided, using task_name='{args.task_name}' with random init")

    print(f"Tasks to evaluate: {task_list}")
    print(f"Total tasks: {len(task_list)}")
    return episode_config, task_list


# ---- Multiprocessing support: flatten the work list + execution/aggregation ----

def build_work_list(args, episode_config, task_list):
    """
    Flatten the two-level (task x config) loop into a one-dimensional work list.
    Precompute the save_media flag for each element: only the first episode of each task saves
    images/video/HTML; the remaining episodes save only JSON evaluation results to save disk.
    """
    work_list = []
    for task_name in task_list:
        if episode_config is not None:
            task_configs, task_category, task_type = get_task_configs(episode_config, task_name)
            if task_configs is None:
                print(f"  Warning: No configs found for task {task_name}")
                continue
        else:
            task_configs = [None]
            task_category, task_type = TASK_CLASS_INDEX.get(task_name, (None, None))

        if args.config_idx is not None:
            if args.config_idx < len(task_configs):
                config_indices = [args.config_idx]
            else:
                print(f"  Warning: Config index {args.config_idx} out of range for {task_name}")
                continue
        else:
            config_indices = range(len(task_configs))

        for config_idx in config_indices:
            save_media = (config_idx == config_indices[0])
            work_list.append({
                'task_name': task_name,
                'config_idx': config_idx,
                'task_config': task_configs[config_idx],
                'task_category': task_category,
                'task_type': task_type,
                'save_media': save_media,
            })
    return work_list


def run_single_episode_worker(work_item, args, base_dir):
    """
    Process-pool worker: run a single episode and return the evaluation result dict.
    Note: each worker rebuilds its own logger (the root logger cannot be serialized across processes).
    """
    logger = get_logger()

    task_name = work_item['task_name']
    config_idx = work_item['config_idx']
    save_media = work_item['save_media']

    task_dir = os.path.join(base_dir, task_name)
    os.makedirs(task_dir, exist_ok=True)

    episode_name = f"{task_name}_episode_{config_idx + 1}"
    task_i_dir = os.path.join(task_dir, episode_name)

    # Only the first episode saves images/video/HTML
    if save_media:
        img_dir = os.path.join(task_i_dir, "img")
        os.makedirs(img_dir, exist_ok=True)
        video_dir = task_dir if args.save_video else None
    else:
        img_dir = None
        video_dir = None

    episode_seed = args.seed + config_idx if args.seed is not None else None

    episode_result = evaluate_single_episode(
        args, task_name, work_item['task_config'], config_idx, logger,
        seed=episode_seed, video_dir=video_dir, img_dir=img_dir,
        task_category=work_item['task_category'], task_type=work_item['task_type']
    )

    # Only the first episode generates the debug HTML
    if save_media:
        html_path = DebugVisualizer().generate(task_i_dir, task_name)
        if html_path:
            print(f"  Debug HTML saved to: {html_path}")

    return episode_result


def run_evaluation(work_list, args, base_dir):
    """
    Execute all episodes (serial or parallel) and aggregate results.
    Returns (task_results, total_episodes, successful_episodes).
    """
    if not work_list:
        return {}, 0, 0

    num_workers = args.num_workers
    if num_workers <= 1:
        episode_results = [run_single_episode_worker(item, args, base_dir) for item in work_list]
    else:
        with Pool(num_workers) as pool:
            episode_results = pool.starmap(
                run_single_episode_worker,
                [(item, args, base_dir) for item in work_list]
            )

    task_results = {}
    total = 0
    successful = 0
    for item, ep_result in zip(work_list, episode_results):
        task_name = item['task_name']
        if task_name not in task_results:
            task_results[task_name] = {
                'total_configs': 0,
                'evaluated_configs': [],
                'success_count': 0,
            }
        task_results[task_name]['evaluated_configs'].append(ep_result)
        task_results[task_name]['total_configs'] += 1
        total += 1
        if ep_result.get('task_success', False):
            successful += 1
            task_results[task_name]['success_count'] += 1

    return task_results, total, successful


def main():
    args = get_args()
    logger = get_logger()

    # 1. Create base_dir: {save_dir}/{vlm_backend}/exp_{MMDD}_{NNN}/
    #    NNN = the largest existing sequence number for the day + 1, so multiple runs on the same day do not overwrite each other
    today = datetime.now()
    month_day = f"{today.month:02d}{today.day:02d}"

    model_dir = os.path.join(args.save_dir, args.vlm_backend)
    os.makedirs(model_dir, exist_ok=True)

    prefix = f"exp_{month_day}_"
    existing_nums = [int(d[-3:]) for d in os.listdir(model_dir) if d.startswith(prefix) and d[-3:].isdigit()]
    exp_str = f"{max(existing_nums, default=0) + 1:03d}"

    base_dir = os.path.join(model_dir, f"exp_{month_day}_{exp_str}")
    os.makedirs(base_dir, exist_ok=True)

    # 2. Build the flat work list
    episode_config, task_list = resolve_evaluation_tasks(args)
    work_list = build_work_list(args, episode_config, task_list)

    # 3. Run the evaluation (serial or parallel)
    task_results, total_episodes, successful_episodes = run_evaluation(work_list, args, base_dir)

    # 4. Assemble results & save to disk
    results = {
        'vlm_backend': args.vlm_backend,
        'evaluation_time': datetime.now().isoformat(),
        'episode_config_path': args.episode_config,
        'tasks': task_results,
        'summary': {
            'total_episodes': total_episodes,
            'successful_episodes': successful_episodes,
            'success_rate': successful_episodes / total_episodes if total_episodes > 0 else 0,
        },
    }

    print(f"\n{'=' * 60}")
    print(f"Evaluation Complete")
    print(f"{'=' * 60}")
    print(f"Total Episodes: {total_episodes}")
    print(f"Successful: {successful_episodes}")
    print(f"Success Rate: {results['summary']['success_rate']:.2%}")

    final_path = save_results(results, base_dir, args.vlm_backend)
    print(f"\nFinal results saved to: {final_path}")


if __name__ == "__main__":
    main()
