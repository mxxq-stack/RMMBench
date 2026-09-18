import base64
import copy
import json
import os
import re

import cv2
import numpy as np

from RMMBench.utils.utils import get_direction_from_euler
from RMMBench.vlm_evaluation.clients import OpenAIFormatClient, PromptImageMapFormatClient

# [RECALL] Skill library descriptions (kept consistent with the "Available Action Library"
# section of new_prompt/manipulation_prompt.txt), maintained directly as a global constant
# dict to avoid parsing them out of the prompt text with regex every time.
# Used by VLAMessageHandler._build_recall_target_summary for direct lookup when assembling
# the keyframe extraction summary.
# Note: if the skill descriptions in manipulation_prompt.txt change, update here accordingly.
ACTION_LIBRARY_DESC = {
    "pick": "Move from the current position to a suitable location for grasping the target object and close the gripper to perform the grasp.",
    "place": "Move from the current position to a suitable location for placing the target object to target container and open the gripper to perform the place.",
    "lift": "Lift your end effector vertically.",
    "pull": "Pull your hand out parallel while maintaining the gripper state.",
    "push": "Push your end effector in while maintaining the gripper state.",
    "observe": "Reset the end-effector to the default position to gain a clear, wide-angle view or to stabilize the camera orientation before executing navigation skills.",
    "open_door": "Given that the gripper has already grasped the handle, open the door, then release the handle.",
    "close_door": "Push or pull the door along its path until fully closed.",
    "recall": "Review the key frames of a previous step if you are unsure whether it succeeded.",
    "end": "If you think you have completed the task, please output 'end'.",
}

# ---- Visualization helper (currently not enabled) ----
def save_vlm_input_image(img, save_path, step_num):
    """
    Save the stitched VLM input image (wrist camera + head camera).
    Enable manually at the handler call site when needed, e.g.:
        save_vlm_input_image(img, img_input_dir, n)
    where img_input_dir = os.path.join(task_i_dir, "img_input")
    """
    os.makedirs(save_path, exist_ok=True)
    vlm_input = np.hstack([img[3], img[5]])
    cv2.imwrite(os.path.join(save_path, f"step{step_num}.png"), vlm_input)


class VLAMessageHandler:
    def __init__(self, env, history_maxlen=8, img_size=(480, 480), vlm_url=None, vlm_backend="gemini",
                 prompt_format="openai", task_category="manipulation", collision_feedback_open=False):
        """
        Initialize the robot instruction handler
        :param history_maxlen: maximum length of the history queue
        :param img_size: actual image size (width, height), used for coordinate restoration
        :param vlm_backend: backend type name, for identification only (e.g. directory naming), such as "gemini", "seed", etc.
        :param prompt_format: prompt format, "openai" or "prompt_image_map"
        :param task_category: task category used to select different prompts; one of "manipulation", "navigation", "composite_navigation"
        """
        assert vlm_url is not None
        self.vlm_url = vlm_url
        self.vlm_backend = vlm_backend
        self.task_category = task_category
        self.collision_feedback_open=collision_feedback_open


        self.open_top_down_cam = False
        self.include_reasoning_in_history = True  # whether to include reasoning in history steps

        self.history_maxlen = history_maxlen - 1
        self.grasp_history = dict()
        # Record pick failure state (no point cloud)
        self.last_pick_no_cloud = False  # True means the previous pick failed due to no point cloud
        self.last_pick_failed_target = None  # record the failed target name

        # [RECALL] Stores keyframe info for each recalled history step
        # Structure: {"step_0": {"keyframe_indices": [...], "reasoning": "...",
        #                   "display_cam_wrist": np.ndarray, "display_cam_head": np.ndarray, ...}, ...}
        # Produced by SkillLib.recall, written by codelab_eval.py,
        # read by build_prompt_components when rendering recall history steps.
        self.recall_results = dict()
        self._recall_step_count = 0  # [RECALL] recall request counter, used as the step identifier for debug_full data saving

        # Create the corresponding client based on prompt_format
        if prompt_format == "openai":
            self.client = OpenAIFormatClient(url=vlm_url)
        elif prompt_format == "prompt_image_map":
            self.client = PromptImageMapFormatClient(url=vlm_url)
        else:
            raise ValueError(f"不支持的 prompt_format: {prompt_format}, 请选择 'openai' 或 'prompt_image_map'")
        self.allow_entities = None
        self.robot_init_info = None
        self.img_list = []
        self.vlm_response = ""
        self.img_width, self.img_height = img_size

        # Load prompt templates for different task categories
        self._load_prompt_templates()

        # Store the system prompt and accumulated user content (referring to the overall_prompt structure in vlm_prompt_mac.py)
        self.system_prompt_content = None
        self.overall_messages = None

        # List for storing full debug info (prompt_data and response for each step)
        self.debug_full_data = []
        self.save_debug_full = False  # switch: whether to save full debug info
        self.debug_save_dir = None  # save directory
        self._action_step_count = 0  # action step counter
        self._grasp_step_count = 0  # grasp step counter

    def _load_prompt_templates(self):
        """
        Load prompt templates for different task categories
        """
        prompt_dir = os.path.join(os.path.dirname(__file__), "new_prompt")

        self.prompt_templates = {
            'manipulation': '',
            'navigation': '',
            'composite_navigation': ''
        }

        # Load manipulation prompt
        manipulation_path = os.path.join(prompt_dir, "manipulation_prompt.txt")
        if os.path.exists(manipulation_path):
            with open(manipulation_path, 'r', encoding='utf-8') as f:
                self.prompt_templates['manipulation'] = f.read()
        else:
            print(f"Warning: manipulation_prompt.txt not found at {manipulation_path}")

        # Load navigation prompt
        navigation_path = os.path.join(prompt_dir, "navigation_prompt.txt")
        if os.path.exists(navigation_path):
            with open(navigation_path, 'r', encoding='utf-8') as f:
                self.prompt_templates['navigation'] = f.read()
        else:
            print(f"Warning: navigation_prompt.txt not found at {navigation_path}")

        # composite_navigation temporarily uses the navigation template
        # composite_navigation_prompt_no_bev
        if self.open_top_down_cam is False:
            composite_navigation = os.path.join(prompt_dir, "composite_navigation_prompt_no_bev.txt")
            if os.path.exists(composite_navigation):
                with open(composite_navigation, 'r', encoding='utf-8') as f:
                    self.prompt_templates['composite_navigation'] = f.read()
            else:
                print(f"Warning: composite_navigation_prompt_no_bev.txt not found at {composite_navigation}")
        else:
            composite_navigation = os.path.join(prompt_dir, "composite_navigation_prompt.txt")
            if os.path.exists(composite_navigation):
                with open(composite_navigation, 'r', encoding='utf-8') as f:
                    self.prompt_templates['composite_navigation'] = f.read()
            else:
                print(f"Warning: composite_navigation.txt not found at {composite_navigation}")

    def _encode_b64(self, rgb_array):
        """
        Internal method: convert an RGB numpy array to base64 encoding
        """
        if len(rgb_array.shape) == 3 and rgb_array.shape[2] == 3:
            rgb = rgb_array
        else:
            rgb = cv2.cvtColor(rgb_array, cv2.COLOR_BGR2RGB)

        success, encoded_image = cv2.imencode('.jpg', rgb)
        if not success:
            raise ValueError("图片编码失败")

        image_base64 = base64.b64encode(encoded_image.tobytes()).decode('utf-8')
        return image_base64

    def _get_system_content(self, instruction):
        """
        Get the corresponding system prompt based on task_category
        Uses the prompt template loaded from the txt file and replaces placeholders
        """
        robot_init_orient = get_direction_from_euler(self.robot_init_info["euler"][2])

        # Get the template for the current task category
        template = self.prompt_templates.get(self.task_category, '')

        if not template:
            print(f"Warning: No template found for task_category '{self.task_category}', using fallback")
            template = self.prompt_templates.get('manipulation', '')

        # Replace placeholders in the template
        try:
            system_content = template.format(
                allow_entities=self.allow_entities,
                instruction=instruction,
                robot_init_orient=robot_init_orient
            )
        except KeyError as e:
            print(f"Warning: Template placeholder {e} not found, trying alternative format")
            system_content = template
            system_content = system_content.replace('{allow_entities}', str(self.allow_entities))
            system_content = system_content.replace('{instruction}', instruction)
            system_content = system_content.replace('{robot_init_orient}', robot_init_orient)

        return system_content

    def parse_vlm_response(self, content):
        """
        Parse the text returned by the VLM, extracting the action and coordinates/parameters
        Compatible format 1 (plain text): action: pick apple \n box_2d: [393, 278, 461, 354]
        Compatible format 2 (JSON): {"action": "pick apple", "box_2d": [393, 278, 461, 354]}
        """
        bbox_2d_n = None
        action_type = None
        target_object = None
        pts = None
        reasoning = None
        cam_view = None  # new: camera view for the pick action

        # 1. First match the complex "place to" format with priority
        place_to_match = re.search(r'^action:\s*place\s+to\s+(\w+)', content, re.IGNORECASE | re.MULTILINE)
        if place_to_match:
            action_type = 'place'
            target_object = place_to_match.group(1)
        else:
            # Keep the original regex matching
            action_match = re.search(r'^action:\s*([\w_]+)(?:\((\d+)\))?(?:\s+(\w+))?', content,
                                     re.IGNORECASE | re.MULTILINE)
            if action_match:
                action_type = action_match.group(1).lower()

                # 1. Handle degree extraction for look-type skills
                if action_type in ['look_left', 'look_right']:
                    if action_match.group(2):
                        target_object = action_match.group(2)
                    else:
                        target_object = '30'  # default value
                # [RECALL] 2. The target of the recall skill is a standalone "target: step_x" line,
                #            not on the same line as the action like pick/place, so extract it separately.
                elif action_type == 'recall':
                    recall_target_match = re.search(r'^target:\s*(step_\d+)', content,
                                                     re.IGNORECASE | re.MULTILINE)
                    target_object = recall_target_match.group(1) if recall_target_match else None
                # 3. Handle regular manipulation skills, and add blacklist restrictions here
                else:
                    potential_target = action_match.group(3)

                    # Key change: if the extracted target is reasoning or box_2d, intercept it and set it to None
                    if potential_target and potential_target.lower() not in ['reasoning', 'box_2d']:
                        target_object = potential_target
                    else:
                        target_object = None

        # 2. Extract cam_view (only the pick action has this field, defaults to wrist)
        if action_type == 'pick':
            cam_view_match = re.search(r'cam_view["\s:]+(\w+)', content, re.IGNORECASE)
            if cam_view_match:
                cam_view = cam_view_match.group(1).lower()
            if cam_view not in ('wrist', 'head'):
                cam_view = 'wrist'  # default to wrist

        # 3. Extract the 2D bounding box (only pick and place will extract successfully; other actions going to the else branch with a print is expected)
        bbox_content_match = re.search(r'box_2d["\s:]+\[([^\]]+)\]', content)

        if bbox_content_match:
            try:
                raw_numbers = re.findall(r'\d+', bbox_content_match.group(1))
                bbox_raw = [int(n) for n in raw_numbers]

                if len(bbox_raw) >= 4:
                    y_min = int(bbox_raw[0] / 1000 * self.img_height)
                    x_min = int(bbox_raw[1] / 1000 * self.img_width)
                    y_max = int(bbox_raw[2] / 1000 * self.img_height)
                    x_max = int(bbox_raw[3] / 1000 * self.img_width)

                    bbox_2d_n = [y_min, x_min, y_max, x_max]

                    pts = np.array([
                        [x_min, y_min], [x_max, y_min],
                        [x_max, y_max], [x_min, y_max]
                    ], dtype=np.float32)
                else:
                    print(f"BBox坐标数量不足: 期望4个,实际得到{len(bbox_raw)}个")
            except Exception as e:
                print(f"BBox解析逻辑出错: {e}")
        else:
            # Navigation and look skills never had a bbox; keep the existing logic here, or add an action_type
            # check to avoid printing the warning for look skills
            # [RECALL] The recall skill likewise has no box_2d; add it to the blacklist to avoid false warnings
            if action_type not in ['moveforward', 'rotate_left', 'rotate_right', 'look_left', 'look_right',
                                   'look_forward', 'end', 'recall']:
                print(f"未能提取到 box_2d 数组内容。Content: {content}")

        # 4. New: extract reasoning
        # Compatible with plain text "reasoning: xxx" and JSON format '"reasoning": "xxx"'
        reasoning_match = re.search(r'reasoning["\s:]+(.*)', content, re.IGNORECASE | re.MULTILINE)
        if reasoning_match:
            # strip() removes surrounding spaces, newlines, or JSON quotes/commas
            reasoning = reasoning_match.group(1).strip('" \t\n\r,}')

        return {
            "bbox_2d": bbox_2d_n,
            "action": action_type,
            "target_object": target_object,
            "pts": pts,
            "reasoning": reasoning,
            "cam_view": cam_view,  # new
        }

    def request_vlm(self,
                    img,
                    n,
                    instruction,
                    action=None,
                    robot_init_info=None,
                    allow_entities=None,
                    target_out_of_reach=False,
                    movedistance=0
                    ):
        """Execute the network request and process the result, sending via the client

        Returns:
            (bbox_2d, action_type, target_object, content, pts)
        """
        self.robot_init_info = robot_init_info
        self.allow_entities = allow_entities
        self.target_out_of_reach = target_out_of_reach
        self.movedistance = movedistance
        print("movedistance:", movedistance)
        # Build the standardized prompt components
        system_prompt, images, user_contents = self.build_prompt_components(
            n, img, instruction, action=action, vlm_response=self.vlm_response
        )
        try:
            print(f"VLM正在响应 (第{n + 1}轮).... [后端: {self.vlm_backend}]")
            # Send the request via the client (the client decides how to assemble the format)
            # To save full debug data, prompt_data must be obtained
            if self.save_debug_full:
                content, prompt_data = self.client.send(
                    system_prompt=system_prompt,
                    images=images,
                    user_contents=user_contents,
                    max_new_tokens=4096,
                    return_prompt_data=True
                )
                # Save the debug data for this step
                self.add_debug_step_data(n, prompt_data, content)
            else:
                content = self.client.send(
                    system_prompt=system_prompt,
                    images=images,
                    user_contents=user_contents,
                    max_new_tokens=4096
                )

            # =========================================================
            # [DEBUG test mode]: skip the real VLM request and return a fixed skill sequence
            # To restore: comment out the DEBUG block below and uncomment the original try block
            # =========================================================

            # ---------- DEBUG block (start) ----------
            # Predefined skill sequence: pick steak -> moveforward -> pick steak -> end
            # fixed_skill_sequence = [
            #     {"action": "pick", "target": "steak", "bbox": [100, 200, 150, 250],
            #      "content": "action: pick steak\nbox_2d: [200, 300, 250, 350]\nreasoning: first pick, distance too far"},
            #     {"action": "moveforward", "target": None, "bbox": None,
            #      "content": "action: moveforward\nreasoning: move closer to target"},
            #     {"action": "pick", "target": "steak", "bbox": [300, 400, 350, 450],
            #      "content": "action: pick steak\nbox_2d: [400, 500, 450, 550]\nreasoning: second pick, should be reachable now"},
            #     {"action": "end", "target": None, "bbox": None, "content": "action: end\nreasoning: task done"},
            # ]
            #
            # if n < len(fixed_skill_sequence):
            #     skill_def = fixed_skill_sequence[n]
            #     action = skill_def["action"]
            #     target_object = skill_def["target"]
            #     bbox = skill_def["bbox"]
            #     content = skill_def["content"]
            #     pts = None
            #     print(f"[DEBUG FIXED SEQUENCE] Step {n}: action={action}, target={target_object}")
            # else:
            #     print(f"[DEBUG FIXED SEQUENCE] Step {n}: sequence exhausted, returning None")
            #     return None
            #
            # # Construct vlm_response and the parse result
            # self.vlm_response = content
            # bbox_2d = skill_def["bbox"]
            # action_type = skill_def["action"]
            # target_object = skill_def["target"]
            # pts = None
            # reasoning = "debug fixed sequence"
            # self.bbox = bbox_2d
            # self.target_object = target_object
            # self.reasoning = reasoning
            #
            # print(f"[DEBUG] Parsed result: action={action_type}, target={target_object}, bbox={bbox_2d}")
            # return bbox_2d, action_type, target_object, content, pts
                # ---------- DEBUG block (end) ----------

            # Simplify vlm_response: keep only the content text, removing the JSON wrapper
            self.vlm_response = content
            print("content:", content)

            parsed = self.parse_vlm_response(content)
            self.bbox = parsed["bbox_2d"]
            self.target_object = parsed["target_object"]
            self.reasoning = parsed["reasoning"]
            self.cam_view = parsed["cam_view"]  # new

            print(f"解析结果: action={parsed['action']}, target={parsed['target_object']},bbox={parsed['bbox_2d']},cam_view={parsed['cam_view']},reasoning:{parsed['reasoning']}")

            # Dict returned by request_vlm externally: no reasoning, includes content (consistent with the original tuple behavior)
            return {
                "bbox_2d": parsed["bbox_2d"],
                "action": parsed["action"],
                "target_object": parsed["target_object"],
                "content": content,
                "pts": parsed["pts"],
                "cam_view": parsed["cam_view"],  # new
            }

        except Exception as e:
            print(f"请求异常: {e}")
            return None

    def _make_image_ref(self, img_id, img_key, img_idx=None, cam_img=None):
        """Build a single image reference dict, reading from the history img_list or the current cam_img."""
        if img_idx is not None:
            data = f"data:image/jpeg;base64,{self.img_list[img_idx][img_key]}"
        else:
            data = f"data:image/jpeg;base64,{cam_img[img_key]}"
        return {"id": f"<image_{img_id}>", "data": data}

    def _append_step_images(self, all_images, img_id, img_idx, use_3_cams):
        """Append image references for a history step to all_images, returning the updated img_id."""
        if use_3_cams:
            all_images.append(self._make_image_ref(img_id, 'cam_top', img_idx=img_idx))
            all_images.append(self._make_image_ref(img_id + 1, 'cam_head', img_idx=img_idx))
            all_images.append(self._make_image_ref(img_id + 2, 'cam_wrist', img_idx=img_idx))
            return img_id + 3
        else:
            all_images.append(self._make_image_ref(img_id, 'cam_head', img_idx=img_idx))
            all_images.append(self._make_image_ref(img_id + 1, 'cam_wrist', img_idx=img_idx))
            return img_id + 2

    def _append_current_images(self, all_images, img_id, cam_img, use_3_cams):
        """Append image references for the current round to all_images, returning the updated img_id."""
        if use_3_cams:
            all_images.append(self._make_image_ref(img_id, 'cam_top', cam_img=cam_img))
            all_images.append(self._make_image_ref(img_id + 1, 'cam_head', cam_img=cam_img))
            all_images.append(self._make_image_ref(img_id + 2, 'cam_wrist', cam_img=cam_img))
            return img_id + 3
        else:
            all_images.append(self._make_image_ref(img_id, 'cam_head', cam_img=cam_img))
            all_images.append(self._make_image_ref(img_id + 1, 'cam_wrist', cam_img=cam_img))
            return img_id + 2

    def _build_user_contents(self, text_content, all_images):
        """Assemble the final user_contents list sent to the large model."""
        user_contents = [{"type": "text", "text": text_content}]
        for img_obj in all_images:
            user_contents.append({"type": "image_url", "image_url": {"url": img_obj["data"]}})
        return user_contents

    def build_prompt_components(self, n, img, instruction, action=None, vlm_response=None):
        """
        Build the refactored standardized prompt components, supporting navigation, manipulation and composite_navigation branches
        Returns: (system_prompt, images, all_user_contents)
        """
        # 1. Encode and store the current round's images (head_joint)
        cam_img = {
            "cam_right": self._encode_b64(img[0]),
            "cam_left": self._encode_b64(img[1]),
            "cam_top": self._encode_b64(img[2]),
            "cam_wrist": self._encode_b64(img[3]),
            "cam_opposite": self._encode_b64(img[4]),
            "cam_head": self._encode_b64(img[5]),
        }
        self.img_list.append(cam_img)
        current_img_idx = len(self.img_list) - 1


        # Get the current round's state
        current_bbox = getattr(self, 'bbox', None)
        current_target_object = getattr(self, "target_object", None)
        current_pick_no_cloud = getattr(self, 'last_pick_no_cloud', False)
        current_reasoning = getattr(self, 'reasoning', False)

        # Compute in real time whether the current step is "attempting manipulation skills before navigation completes"
        manipulation_skills = ['pick', 'place',  'open_door','close_door']
        is_premature_manipulation = False

        if n > 0 and action:
            action_name = action.split()[0] if isinstance(action, str) else action
            if action_name in manipulation_skills:
                is_premature_manipulation = True

        # 2. Record or update structured data
        if not hasattr(self, 'history_steps'):
            self.history_steps = []

        # For subsequent rounds, the action passed in now is actually "the decision result the model made at the previous step (n-1)"
        if n > 0 and len(self.history_steps) > 0:
            self.history_steps[-1]["action"] = action
            self.history_steps[-1]["bbox"] = current_bbox  # backfill the latest bbox to the previous step
            self.history_steps[-1]["reasoning"] = current_reasoning  # backfill the latest reasoning to the previous step
            self.history_steps[-1]["target"] = current_target_object  # backfill the latest target to the previous step
            self.history_steps[-1]["is_premature_manipulation"] = is_premature_manipulation
            self.history_steps[-1]["movedistance"] = self.movedistance  # backfill the latest movedistance to the previous step
            old_tor = self.history_steps[-1].get("target_out_of_reach", "NOT_SET")
            new_tor = getattr(self, "target_out_of_reach", False)
            self.history_steps[-1]["target_out_of_reach"] = new_tor
            print(f"[DEBUG PROMPT] Backfill step {self.history_steps[-1]['step_idx']}: target_out_of_reach {old_tor} -> {new_tor}")
            # Fix: assign the is_premature_manipulation flag to the previous step (the step that actually performed manipulation)
            # if is_premature_manipulation:
            #     self.history_steps[-1]["is_premature_manipulation"] = True

            # If there is a vlm_response from the previous round, update it too

        # Create a placeholder for the current step (its action has not happened yet, set to None)
        step_data = {
            "step_idx": n,
            "action": None,  # the current step's action will be backfilled at the next call (n+1)
            "bbox": None,
            "img_idx": current_img_idx,
            "target": current_target_object,
            "reasoning": current_reasoning,
            "is_pick_no_cloud": current_pick_no_cloud,
            "is_premature_manipulation": False,  # the current step has not executed an action yet, initialize to False
            "movedistance": self.movedistance,  # record this step's movedistance to avoid being overwritten globally later
            "target_out_of_reach": getattr(self, "target_out_of_reach", False),  # snapshot the state at this moment to avoid the global value polluting historical step prompts
        }

        self.history_steps.append(step_data)
        # 3. Get the system prompt
        system_prompt = self._get_system_content(instruction)

        # =========================================================
        # Core branching point: for composite navigation tasks, both n=0 and n>0 are handled
        # by a dedicated sub-method
        # =========================================================
        if self.task_category == "composite_navigation":
            return self._build_composite_navigation_prompt(n, cam_img, system_prompt)

        # =========================================================
        # 4. Original logic: initial round (n == 0) of plain navigation or manipulation tasks
        # =========================================================
        if n == 0:
            if self.task_category == "navigation":
                user_text = "These are your initial observation. Please analyze the scene and select a navigation action.\n"
                user_text += "1. TOP-DOWN CAMERA:<image_1>\n2. HEAD CAMERA:<image_2>\n3. WRIST CAMERA:<image_3>\n"
                all_images = [
                    {"id": "<image_1>", "data": f"data:image/jpeg;base64,{cam_img['cam_top']}"},
                    {"id": "<image_2>", "data": f"data:image/jpeg;base64,{cam_img['cam_head']}"},
                    {"id": "<image_3>", "data": f"data:image/jpeg;base64,{cam_img['cam_wrist']}"},
                ]
            else:  # manipulation initial round (usually uses 2 images)
                user_text = "These are your initial observation. Please analyze the scene and select a manipulation action.\n"
                user_text += "1. HEAD CAMERA:<image_1>\n2. WRIST CAMERA:<image_2>\n"
                all_images = [
                    {"id": "<image_1>", "data": f"data:image/jpeg;base64,{cam_img['cam_head']}"},
                    {"id": "<image_2>", "data": f"data:image/jpeg;base64,{cam_img['cam_wrist']}"},
                ]

            all_user_contents = self._build_user_contents(user_text, all_images)
            return system_prompt, all_images, all_user_contents

        # =========================================================
        # 5. Original logic: subsequent rounds (n > 0) of plain navigation or manipulation tasks
        # =========================================================
        all_images = []
        img_id = 1
        history_text = "Your historical actions and recent observations are as follows:\n"

        history_steps_to_process = self.history_steps[:-1]
        total_history_count = len(history_steps_to_process)

        use_3_cams = self.task_category == "navigation"

        # Iterate over all history steps
        for idx, step in enumerate(history_steps_to_process):
            step_num = step["step_idx"]
            act = step["action"]
            box = step["bbox"]
            img_idx = step["img_idx"]
            target = step["target"]
            is_no_cloud = step["is_pick_no_cloud"]
            reasoning = step["reasoning"]

            # =====================================================
            # [RECALL] When a history step is recall, use a dedicated display branch:
            #   - Do not use the current step_data's img_idx (it points to the next round's
            #     observation after the recall, not the keyframes of the recalled trajectory);
            #     instead, fetch the display frames (display_cam_wrist/head) chosen by
            #     SkillLib.recall from self.recall_results[target].
            #   - The number of image placeholders stays consistent with regular steps
            #     (2 images, or without top in 3cams mode), so they naturally slide out of
            #     / get displayed with the history_maxlen window without extra counting logic.
            #   - The text carries keyframes indices and reasoning so the VLM can understand
            #     the recall result.
            # =====================================================
            if act == "recall":
                recall_info = self.recall_results.get(target, None)
                step_prefix = (
                    f"step{step_num}:action:recall,target:{target},"
                    f"keyframes:{recall_info['keyframe_indices'] if recall_info else None},"
                    f"reasoning:{recall_info['reasoning'] if recall_info else None}"
                )
                if recall_info is not None and (total_history_count - idx) <= self.history_maxlen:
                    p_head = f"<image_{img_id}>"
                    p_wrist = f"<image_{img_id + 1}>"
                    history_text += (
                        f"{step_prefix},key frame observation:HEAD CAMERA:{p_head} WRIST CAMERA:{p_wrist};\n\n"
                    )
                    recall_cam_img = {
                        "cam_head": self._encode_b64(recall_info["display_cam_head"]),
                        "cam_wrist": self._encode_b64(recall_info["display_cam_wrist"]),
                    }
                    all_images.append(self._make_image_ref(img_id, 'cam_head', cam_img=recall_cam_img))
                    all_images.append(self._make_image_ref(img_id + 1, 'cam_wrist', cam_img=recall_cam_img))
                    img_id += 2
                else:
                    history_text += f"{step_prefix}, (key frame observation omitted)；\n\n"
                continue
            # =====================================================

            # Build the basic text format
            step_prefix = f"step{step_num}:action:{act},target:{target},bbox:{box},reasoning:{reasoning}"
            # -----
            if self.task_category == "manipulation":
                # 1. Highest priority: first check whether the target is beyond the arm's physical reach
                if step.get("target_out_of_reach", False) is True:
                    step_prefix += ",attention: The target object is out of your reachable range. The current distance exceeds the maximum arm reach."
                # 2. Second priority: if the distance is fine (within range), check whether a visual point-cloud anomaly was triggered
                elif is_no_cloud:
                    step_prefix += ",attention: Your action was not executed. No point cloud was detected within the predicted bounding box. Please re-verify the object's visual position and output a more precise bbox."
            # -----
            # Check whether this history step is within the recent history_maxlen window
            if (total_history_count - idx) <= self.history_maxlen:
                if use_3_cams:
                    p_top = f"<image_{img_id}>"
                    p_head = f"<image_{img_id + 1}>"
                    p_wrist = f"<image_{img_id + 2}>"
                    history_text += f"{step_prefix},observation at this time:{p_top} {p_head} {p_wrist};\n\n"
                else:
                    p_head = f"<image_{img_id}>"
                    p_wrist = f"<image_{img_id + 1}>"
                    history_text += f"{step_prefix},observation at this time:HEAD CAMERA:{p_head} WRIST CAMERA:{p_wrist};\n\n"
                img_id = self._append_step_images(all_images, img_id, img_idx, use_3_cams)
            else:
                # For steps far beyond the window, add explicit semantic marker protection text to prevent the Client from filtering them out
                history_text += f"{step_prefix}, (observation omitted)；\n\n"

        # 6. Build current round observations (Current Observations)
        if use_3_cams:
            cur_top = f"<image_{img_id}>"
            cur_head = f"<image_{img_id + 1}>"
            cur_wrist = f"<image_{img_id + 2}>"
            current_text = f"Your current observations are as follows:{cur_top} {cur_head} {cur_wrist};\n"
            current_text += "Please continue navigating or output 'action: end' if you have reached the target."
        else:
            # Manipulation current observation (2 images)
            cur_head = f"<image_{img_id}>"
            cur_wrist = f"<image_{img_id + 1}>"
            current_text = f"Your current observations are as follows:HEAD CAMERA:{cur_head} WRIST CAMERA:{cur_wrist};\n"
            current_text += "Please select a manipulation action or output 'action: end' if you have finished the task."

        img_id = self._append_current_images(all_images, img_id, cam_img, use_3_cams)

        # 7. Assemble the final all_user_contents sent to the large model
        all_user_contents = self._build_user_contents(history_text + current_text, all_images)

        print(f"Task: {self.task_category}, History steps: {len(self.history_steps)}, Total images: {len(all_images)}")
        return system_prompt, all_images, all_user_contents

    def _get_grasp_selection_system_content(self, target_entity):
        """
        Get the grasp selection system prompt

        Args:
            target_entity: name of the target entity

        Returns:
            str: system prompt text
        """
        return f"""Role: You are an expert in robotic arm grasp planning.

Task: pick {target_entity}

Rule:   1
- Select the grasping object that you think is most suitable from the provided candidates based on the wrist camera perspective (slightly tilted downward).
- Indicator: Green open-ended rectangular wireframes.The opening direction of the wireframe indicates the gripper's Approach Axis.

Output Format Rules:
- Output the choice in the format: choice: 1/2/3/4...
- You should optionally output your reasoning process in the format: reasoning: <your step-by-step analysis>

Examples:
- Case 1 (With reasoning):
choice: 1
reasoning: your reasoning

"""

    def select_grasp_prompt(self, viz_list, target_entity):
        """
        Build prompt components for grasp selection, adapting to different client formats

        Return format is the same as build_prompt_components:
        (system_prompt, images, user_contents)

        Args:
            viz_list: list of grasp candidate visualization images
            target_entity: name of the target entity

        Returns:
            tuple: (system_prompt, images, user_contents)
                - system_prompt: system prompt text
                - images: image list [{"id": "<image_1>", "data": "base64..."}, ...]
                - user_contents: user content list [{"type": "text", ...}, {"type": "image_url", ...}, ...]
        """
        # Encode all candidate images
        viz_g_list = []
        for i in viz_list:
            viz_g_list.append(self._encode_b64(i))

        # Build the system prompt
        system_prompt = self._get_grasp_selection_system_content(target_entity)

        # Build the user content and image lists
        user_contents = []
        images = []
        img_id = 1

        # Add failed historical selections (if any)
        history_text = ""
        if target_entity in self.grasp_history and self.grasp_history[target_entity]:
            history_text = "Your previous grasp selections are as follows; none of them were successful. However, if you still find a similar grasp appropriate, you may continue to select it:\n"
            for index, img_base64 in enumerate(self.grasp_history[target_entity]):
                history_text += f"Round {index + 1} selection:<image_{img_id}>\n"
                images.append({"id": f"<image_{img_id}>", "data": f"data:image/jpeg;base64,{img_base64}"})
                img_id += 1
            history_text += "\n"

        # Build the text and images for the current selection section
        current_text = "This is your current stage of choice:\nYou are presented with multiple grasp candidates. Please examine the following visualizations and select the best grasp orientation.\n"
        for i, img_base64 in enumerate(viz_g_list):
            current_text += f"{i + 1}. OPTION {i + 1}: <image_{img_id}>\n"
            images.append({"id": f"<image_{img_id}>", "data": f"data:image/jpeg;base64,{img_base64}"})
            img_id += 1
        current_text += "Based on these visualizations and environmental constraints, output your choice in the format: choice: 1/2/3/4.."

        # Combine the full user text
        full_user_text = history_text + current_text

        # Build user_contents (text and images alternating, same format as build_prompt_components)
        user_contents.append({"type": "text", "text": full_user_text})

        # Add an image_url reference for each image
        for i, img in enumerate(images):
            user_contents.append({
                "type": "image_url",
                "image_url": {"url": img["data"]}
            })

        return system_prompt, images, user_contents

    def request_grasp(self, viz_list, target_entity, retry_limit=10, step_idx=None):
        """
        Execute the grasp selection network request, sending via the client

        Args:
            viz_list: list of grasp candidate visualization images (list of numpy arrays)
            target_entity: name of the target entity
            retry_limit: retry limit
            step_idx: current step index, used to save debug data (optional)

        Returns:
            tuple: (option, content)
                - option: parsed choice (int or None)
                - content: raw VLM response content
        """
        # Build the standardized prompt components
        system_prompt, images, user_contents = self.select_grasp_prompt(
            viz_list=viz_list,
            target_entity=target_entity
        )

        try:
            print(f"VLM正在响应 (抓取选择).... [后端: {self.vlm_backend}]")
            # Send the request via the client (the client decides how to assemble the format)
            # To save full debug data, prompt_data must be obtained
            if self.save_debug_full:
                content, prompt_data = self.client.send(
                    system_prompt=system_prompt,
                    images=images,
                    user_contents=user_contents,
                    max_new_tokens=4096,
                    return_prompt_data=True
                )
                # Save the debug data for this grasp request with a special step identifier
                grasp_step_idx = step_idx if step_idx is not None else f"grasp_{len(self.debug_full_data)}"
                self.add_debug_step_data(grasp_step_idx, prompt_data, content, step_type="grasp")
            else:
                content = self.client.send(
                    system_prompt=system_prompt,
                    images=images,
                    user_contents=user_contents,
                    max_new_tokens=4096
                )

            print("content:", content)

            option = self.parse_grasp_response(content)

            print(f"解析结果: grasp option: {option}")

            # Save the selected image to grasp_history
            if option is not None and 1 <= option <= len(viz_list):
                # Initialize history for this target_entity (if not present)
                if target_entity not in self.grasp_history:
                    self.grasp_history[target_entity] = []

                # Get the selected image (option is 1-based, convert to a 0-based index)
                selected_img = viz_list[option - 1]

                # Encode to base64 and save
                selected_img_b64 = self._encode_b64(selected_img)
                self.grasp_history[target_entity].append(selected_img_b64)

                print(
                    f"已保存选择到 grasp_history[{target_entity}], 当前历史数: {len(self.grasp_history[target_entity])}")
            else:
                if option is None:
                    print(f"警告: 无法解析有效的 option，跳过保存 grasp_history")
                else:
                    print(f"警告: option={option} 超出范围 [1, {len(viz_list)}]，跳过保存 grasp_history")

            return option, content

        except Exception as e:
            print(f"请求异常: {e}")
            return None, str(e)

    def parse_grasp_response(self, content):
        """
        Supports extraction in the following formats:
        - choice: 3
        - choice: [3]
        - -choice: [3]
        - Choice:3
        """
        # Regex explanation:
        # choice\s*:\s* matches choice and the colon, allowing spaces in between
        # \[?            optionally matches the left bracket [
        # (\d+)          core capture group: matches one or more digits
        # \]?            optionally matches the right bracket ]
        pattern = r"choice\s*:\s*\[?(\d+)\]?"

        match = re.search(pattern, content, re.IGNORECASE)

        if match:
            choice_value = int(match.group(1))
            return choice_value
        else:
            # Fallback logic: if the formats above did not match, try to find the first number directly
            # Useful when the model does not follow the format at all
            print(f"Warning: Standard format not found, attempting fallback...")
            fallback_match = re.search(r"(\d+)", content)
            if fallback_match:
                return int(fallback_match.group(1))

            print(f"Error: Could not parse any choice from content: {content}")
            return None

    def set_pick_no_cloud(self, target_entity=None):
        """
        Set the state of a pick that failed due to no point cloud
        :param target_entity: name of the failed target entity
        """
        self.last_pick_no_cloud = True
        self.last_pick_failed_target = target_entity
        # print(f"[VLM Handler] Pick failed due to no point cloud for target: {target_entity}")

    # =========================================================================
    # === [RECALL] Keyframe extraction methods ================================
    # -------------------------------------------------------------------------
    # This group of methods serves SkillLib.recall exclusively: from the sampled frames
    # of a history step (usually 16 frames, each a wrist+head horizontal concatenation),
    # it makes a separate VLM call asking it to pick the keyframes truly worth attention
    # (e.g. the grasp moment, the moment the object slips away, place success/failure, etc.).
    #
    # These methods are fully independent of the main-flow prompt building in
    # build_prompt_components, similar to how select_grasp_prompt / request_grasp
    # handle grasp selection independently.
    # =========================================================================
    def _build_recall_target_summary(self, step_idx):
        """
        [RECALL] Build the VLM decision summary of the recalled target step in the main flow,
        used by the keyframe extraction request (request_keyframe), replacing the previous
        overly simple action_desc (a single action word).

        The summary contains the step's action / target / reasoning at the time, plus the
        official skill description for that action (from the Available Action Library),
        helping the frame-sampling model truly understand "what happened at that step and
        what the VLM intended to do".

        Args:
            step_idx: str, the recalled step identifier, e.g. "step_0"

        Returns:
            str: the assembled summary text; if no history record is found for the step,
            returns a fallback explanation.
        """
        step_num = None
        if "_" in step_idx:
            try:
                step_num = int(step_idx.split("_")[-1])
            except ValueError:
                step_num = None

        target_step_data = None
        if hasattr(self, "history_steps") and step_num is not None:
            for s in self.history_steps:
                if s.get("step_idx") == step_num:
                    target_step_data = s
                    break

        if target_step_data is None:
            return f"No recorded VLM decision found for step \"{step_idx}\"."

        action = target_step_data.get("action")
        target = target_step_data.get("target")
        reasoning = target_step_data.get("reasoning")

        summary_lines = [f"The VLM's output and reasoning at step \"{step_idx}\" were as follows:"]
        fields = [f"action: {action}"]
        if target:  # not shown when target is None (target-less skills such as lift/pull)
            fields.append(f"target: {target}")
        fields.append(f"reasoning: {reasoning}")
        summary_lines.append(", ".join(fields))

        # Append the official skill description for this action (directly look up the global
        # constant dict ACTION_LIBRARY_DESC); skip if not found, which should not normally happen.
        action_desc = ACTION_LIBRARY_DESC.get(action) if action else None
        if action_desc:
            summary_lines.append(f"{action} skill description: {action_desc}")

        return "\n".join(summary_lines)

    def _build_keyframe_prompt(self, step_idx, target_summary, n_frames):
        """
        [RECALL] Build the system prompt for the keyframe extraction request.

        Note: this deliberately uses a "unified attention list" instead of customizing large
        prompt sections per action type, because the core keyframe judgment logic (before/after
        state changes, abnormal moments, first/last frames) is common to all skills; the
        specific action semantics are instead provided dynamically by target_summary (the
        step's action/target/reasoning + official skill description), so no per-action
        hardcoded prompt is needed.

        Args:
            step_idx: str, the recalled step, e.g. "step_0"
            target_summary: str, the step's VLM decision summary in the main flow (see _build_recall_target_summary)
            n_frames: int, total number of sampled frames

        Returns:
            str: the system prompt
        """
        prompt = f"""Role: You are an expert at reviewing robot manipulation/navigation trajectories.

Task: You are reviewing the full trajectory of a previously executed step "{step_idx}".
This trajectory has been uniformly sampled into {n_frames} frames (frame index starts from 0, in chronological order).
Each frame image is a horizontal concatenation of two camera views: the LEFT half is the WRIST camera (close-up view near the gripper), the RIGHT half is the HEAD camera (wider overview of the scene).

Context about this step (what the robot intended to do at the time):
{target_summary}

Your job:
1. Carefully look through ALL {n_frames} frames in order, and select the SINGLE frame that best represents the final/most decisive state of this trajectory (usually the frame that most clearly shows whether this step succeeded or failed), so that another model can quickly judge the outcome just by looking at this one frame, based on the context above.
2. In your reasoning, briefly describe what happens ACROSS THE WHOLE trajectory (not just the chosen frame) — e.g., how the state evolves over time, and call out any specific frames where something noteworthy or abnormal happens (state changes, contact/release moments, anything that looks wrong or unexpected). This gives full context even though only one frame is returned.

Attention:
- The frame right BEFORE/AFTER an important state change (e.g., before/after contact/release).
- Any frame where something looks WRONG or unexpected (e.g., object slipping, missing target, collision).
- The final state of the trajectory, which usually best reflects success/failure.
- If action is "pick": via the WRIST camera, check whether the object ends up firmly grasped between the gripper fingers.
- If action is "place": check whether the object ends up inside the target container, or falls/drops midway instead.

Output Format Rules:
keyframes: [<exactly ONE frame index, 0-based>]
reasoning: <a summary of the key information across the whole trajectory, referencing specific frame indices when describing noteworthy moments, and explaining why the chosen frame best represents the outcome>

Example:
keyframes: [15]
reasoning: The trajectory starts with the gripper approaching the object; around frame 7 the gripper closes and lifts it; however around frame 11 the object appears to slip, and by frame 15 the object is missing from the target container, indicating the step likely failed. Frame 15 is chosen as it most clearly shows this final outcome.
"""
        return prompt

    def request_keyframe(self, frames, step_idx, step_action=None):
        """
        [RECALL] Make an independent request to the VLM to extract keyframes from the sampled
        trajectory frames.

        Fully independent of the main-flow request_vlm: not appended to history_steps / img_list,
        and not involved in the main prompt building of build_prompt_components.

        Args:
            frames: List[np.ndarray], the list of sampled images, each already a wrist+head
                    horizontal concatenation.
            step_idx: str, the recalled step identifier, e.g. "step_0".
            step_action: str or None, kept for backward compatibility with old call signatures,
                         no longer used (the summary is now generated internally by looking up
                         history_steps via step_idx, see _build_recall_target_summary).

        Returns:
            (keyframe_indices, reasoning):
                keyframe_indices: List[int] or None, keyframe indices chosen by the VLM (local
                                  indices into the frames list)
                reasoning: str, the VLM's reasoning explanation
        """
        # [RECALL] Build the step's action/target/reasoning + official skill description,
        # replacing the previous simple approach of passing just an action word (step_action),
        # helping the frame-sampling model understand the context.
        target_summary = self._build_recall_target_summary(step_idx)
        system_prompt = self._build_keyframe_prompt(step_idx, target_summary, len(frames))

        # Encode all sampled frames
        images = []
        img_id = 1
        frame_refs = []
        for frame in frames:
            img_b64 = self._encode_b64(frame)
            images.append({"id": f"<image_{img_id}>", "data": f"data:image/jpeg;base64,{img_b64}"})
            frame_refs.append(f"<image_{img_id}>")
            img_id += 1

        user_text = f"Here are the {len(frames)} sampled frames of step \"{step_idx}\" in chronological order:\n"
        for i, ref in enumerate(frame_refs):
            user_text += f"frame {i}: {ref}\n"
        user_text += "\nPlease select the key frame(s) following the output format described above."

        user_contents = [{"type": "text", "text": user_text}]
        for img_obj in images:
            user_contents.append({"type": "image_url", "image_url": {"url": img_obj["data"]}})

        try:
            print(f"VLM正在响应 (recall关键帧提取, target={step_idx}).... [后端: {self.vlm_backend}]")
            if self.save_debug_full:
                content, prompt_data = self.client.send(
                    system_prompt=system_prompt,
                    images=images,
                    user_contents=user_contents,
                    max_new_tokens=2048,
                    return_prompt_data=True
                )
                recall_step_idx = f"recall_{self._recall_step_count}_{step_idx}"
                self._recall_step_count += 1
                self.add_debug_step_data(recall_step_idx, prompt_data, content, step_type="recall")
            else:
                content = self.client.send(
                    system_prompt=system_prompt,
                    images=images,
                    user_contents=user_contents,
                    max_new_tokens=2048
                )

            print("recall keyframe content:", content)
            keyframe_indices, reasoning = self.parse_keyframe_response(content)
            print(f"[RECALL] 解析结果: keyframes={keyframe_indices}, reasoning={reasoning}")
            return keyframe_indices, reasoning

        except Exception as e:
            print(f"[RECALL] 关键帧请求异常: {e}")
            return None, None

    def parse_keyframe_response(self, content):
        """
        [RECALL] Parse the VLM response to the keyframe extraction request.

        Expected format (only two items, no action/target, see the output format requirements
        in _build_keyframe_prompt):
            keyframes: [15]
            reasoning: ...
        Although the VLM is asked to return only one keyframe index, the parsing here still
        handles a list for compatibility (if the VLM occasionally returns a few extra indices,
        reading them as a list does not error; the caller takes only the first one).

        Returns:
            (keyframe_indices, reasoning):
                keyframe_indices: List[int] or None
                reasoning: str or None
        """
        keyframe_indices = None
        reasoning = None

        keyframes_match = re.search(r'keyframes["\s:]+\[([^\]]+)\]', content, re.IGNORECASE)
        if keyframes_match:
            try:
                keyframe_indices = [int(x.strip()) for x in re.findall(r'\d+', keyframes_match.group(1))]
            except Exception as e:
                print(f"[RECALL] keyframes解析出错: {e}")
                keyframe_indices = None
        else:
            print(f"[RECALL] 未能提取到 keyframes 数组内容。Content: {content}")

        reasoning_match = re.search(r'reasoning["\s:]+(.*)', content, re.IGNORECASE | re.MULTILINE)
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip('" \t\n\r,}')

        return keyframe_indices, reasoning

    def enable_debug_full_save(self, save_dir: str):
        """
        Enable saving full debug info

        Args:
            save_dir: save directory, e.g. /home/sankuai/work/RMMBench/RMMBench/vlm_evaluation/results/Qwen3d5/episode_001_0325/cluster_vegetables_vs_fruits/task_1
        """
        self.save_debug_full = True
        if self.save_debug_full:
            print("开启保存完整vlm_data")
        self.debug_save_dir = save_dir
        self.debug_full_data = []  # clear previous data
        os.makedirs(save_dir, exist_ok=True)
        print(f"[Debug] Full debug data saving enabled. Will save to: {save_dir}")

    def _simplify_image_refs(self, data):
        """
        Recursively simplify image base64 references in the data, replacing base64 strings with variable names
        e.g.: f"data:image/jpeg;base64,{head_img}" -> "<head_img>"
        """
        if isinstance(data, dict):
            result = {}
            for k, v in data.items():
                if k == "url" and isinstance(v, str) and v.startswith("data:image"):
                    # Detected an image URL, extract the variable name
                    result[k] = "<image_placeholder>"
                elif isinstance(v, str) and v.startswith("data:image/jpeg;base64,"):
                    # Simplify the image data reference
                    result[k] = "<base64_image_data>"
                else:
                    result[k] = self._simplify_image_refs(v)
            return result
        elif isinstance(data, list):
            return [self._simplify_image_refs(item) for item in data]
        elif isinstance(data, str):
            # If the string contains base64 image data, simplify it
            if data.startswith("data:image/jpeg;base64,") and len(data) > 100:
                return "<base64_image_data>"
            return data
        else:
            return data

    def add_debug_step_data(self, step_idx, prompt_data: dict, response_data: dict, step_type: str = "action"):
        """
        Add full debug data for one step and write it to file immediately

        Args:
            step_idx: step index (int or str)
            prompt_data: prompt data sent to the VLM
            response_data: full response data returned by the VLM
            step_type: step type, either "action" (regular action) or "grasp" (grasp selection)
        """
        if not self.save_debug_full or not self.debug_save_dir:
            return

        # Check whether this is the first save (action and grasp are counted independently)
        is_first_step = False
        if step_type == "action":
            is_first_step = (self._action_step_count == 0)
            self._action_step_count += 1
        elif step_type == "grasp":
            is_first_step = (self._grasp_step_count == 0)
            self._grasp_step_count += 1
        elif step_type == "recall":
            # [RECALL] Each recall request is an independent, one-off complete prompt (unlike
            # action, whose history grows), so always keep the full system prompt to make it
            # easy to inspect each recall's sampled frames and prompt content when debugging.
            is_first_step = True

        # If not the first, save only the user part (remove the system prompt)
        if not is_first_step and isinstance(prompt_data, dict):
            prompt_data_copy = copy.deepcopy(prompt_data)
            # Remove the system prompt key-value pairs
            if "system" in prompt_data_copy:
                del prompt_data_copy["system"]
            if "system_prompt" in prompt_data_copy:
                del prompt_data_copy["system_prompt"]
            # For OpenAI-format messages, remove role=system messages
            if "messages" in prompt_data_copy:
                prompt_data_copy["messages"] = [
                    msg for msg in prompt_data_copy["messages"]
                    if msg.get("role") != "system"
                ]

            # =========================================================
            # [Core fix]: precise slicing, never cutting off any distant pure-text history
            # =========================================================
            if "prompt" in prompt_data_copy and isinstance(prompt_data_copy["prompt"], str):
                full_prompt = prompt_data_copy["prompt"]

                # Define the absolute starting sentences of the user content (for n=0 and n>0 cases)
                start_keywords = [
                    "Your historical actions and recent observations are as follows:",
                    "These are your initial observation."
                ]

                user_start_idx = -1
                # Find which starting sentence appears first in the text
                for kw in start_keywords:
                    idx = full_prompt.find(kw)
                    if idx != -1:
                        user_start_idx = idx
                        break

                if user_start_idx != -1:
                    # Precisely keep everything after the starting sentence (including the full STEP history text)
                    prompt_data_copy["prompt"] = "[System prompt omitted for brevity]\n" + full_prompt[user_start_idx:]
                else:
                    # Fallback: if none is found, the structure may have changed; keep as-is without truncating to ensure data safety
                    pass
        else:
            prompt_data_copy = prompt_data

        # Simplify image data in prompt_data
        simplified_prompt = self._simplify_image_refs(prompt_data_copy)

        step_data = {
            "step": step_idx,
            "step_type": step_type,
            # "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "prompt_data": simplified_prompt,
            "response_data": response_data
        }
        self.debug_full_data.append(step_data)

        # Write to file immediately (real-time saving, updated at every step)
        filepath = os.path.join(self.debug_save_dir, f"{getattr(self, '_current_task_name', 'task')}_vlm_feedback.json")
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.debug_full_data, f, indent=2, ensure_ascii=False)
            print(f"[Debug] Step {step_idx} data saved to: {filepath}")
        except Exception as e:
            print(f"[Debug] Failed to save step {step_idx} data: {e}")

    # def save_debug_full_data(self, task_name: str = "task"):
    #     """
    #     Save all debug data to a JSON file
    #
    #     Args:
    #         task_name: task name, used for the file name
    #     """
    #     if not self.save_debug_full or not self.debug_save_dir:
    #         return
    #
    #     if not self.debug_full_data:
    #         print("[Debug] No debug data to save")
    #         return
    #
    #     filepath = os.path.join(self.debug_save_dir, f"{task_name}_vlm_feedback.json")
    #
    #     try:
    #         with open(filepath, 'w', encoding='utf-8') as f:
    #             json.dump(self.debug_full_data, f, indent=2, ensure_ascii=False)
    #         print(f"[Debug] Full debug data saved to: {filepath}")
    #     except Exception as e:
    #         print(f"[Debug] Failed to save debug data: {e}")

    def _build_composite_navigation_prompt(self, n, cam_img, system_prompt):
        """
        Dedicated sub-method for building the prompt of composite navigation tasks
        """
        # Detect the number of cameras in use
        use_3_cams = getattr(self, 'open_top_down_cam', None) is True

        # =========================================================
        # Branch 1: initial round (n == 0)
        # =========================================================
        if n == 0:
            user_text = (
                "These are your initial observation. This is a composite task requiring navigation followed by manipulation.\n"
                "Phase 1 - Navigation: First navigate to the target location and adjust your orientation.\n"
                "Phase 2 - Manipulation: Only after reaching the target position, execute manipulation skills.\n"
            )
            if use_3_cams:
                user_text += "1. TOP-DOWN CAMERA:<image_1>\n2. HEAD CAMERA:<image_2>\n3. WRIST CAMERA:<image_3>\n"
                all_images = [
                    {"id": "<image_1>", "data": f"data:image/jpeg;base64,{cam_img['cam_top']}"},
                    {"id": "<image_2>", "data": f"data:image/jpeg;base64,{cam_img['cam_head']}"},
                    {"id": "<image_3>", "data": f"data:image/jpeg;base64,{cam_img['cam_wrist']}"},
                ]
            else:
                user_text += "1. HEAD CAMERA:<image_1>\n2. WRIST CAMERA:<image_2>\n"
                all_images = [
                    {"id": "<image_1>", "data": f"data:image/jpeg;base64,{cam_img['cam_head']}"},
                    {"id": "<image_2>", "data": f"data:image/jpeg;base64,{cam_img['cam_wrist']}"},
                ]

            all_user_contents = [{"type": "text", "text": user_text}]
            for img_obj in all_images:
                all_user_contents.append({"type": "image_url", "image_url": {"url": img_obj["data"]}})
            return system_prompt, all_images, all_user_contents

        # =========================================================
        # Subsequent rounds (n > 0): dynamically build history and current observations
        # =========================================================
        all_images = []
        img_id = 1
        history_text = "Your historical actions and recent observations are as follows:\n"

        # Dynamic slicing. If the last step is the placeholder of the not-yet-occurred current
        # step (action is None), cut it off; if the last step's action has been backfilled,
        # or the list is empty, keep the full list.
        if len(self.history_steps) > 0 and self.history_steps[-1]["action"] is None:
            history_steps_to_process = self.history_steps[:-1]
        else:
            history_steps_to_process = self.history_steps

        total_history_count = len(history_steps_to_process)

        for idx, step in enumerate(history_steps_to_process):
            step_num = step["step_idx"]
            act = step["action"]
            box = step["bbox"]
            img_idx = step["img_idx"]
            target = step["target"]
            # Read the snapshot state flags
            is_premature = step["is_premature_manipulation"]
            is_no_cloud = step["is_pick_no_cloud"]
            reasoning = step["reasoning"]

            # =====================================================
            # [RECALL] The composite_navigation branch also needs to support displaying recall
            #          history steps, consistent with the manipulation/navigation branch
            #          (see the comments in the branch above for details).
            # =====================================================
            if act == "recall":
                recall_info = self.recall_results.get(target, None)
                step_str = (
                    f"STEP{step_num}:action:recall,target:{target},"
                    f"keyframes:{recall_info['keyframe_indices'] if recall_info else None},"
                    f"reasoning:{recall_info['reasoning'] if recall_info else None}"
                )
                if recall_info is not None and (total_history_count - idx) <= self.history_maxlen:
                    p_head = f"<image_{img_id}>"
                    p_wrist = f"<image_{img_id + 1}>"
                    step_str += f",key frame observation:HEAD CAMERA:{p_head} WRIST CAMERA:{p_wrist};\n\n"
                    recall_cam_img = {
                        "cam_head": self._encode_b64(recall_info["display_cam_head"]),
                        "cam_wrist": self._encode_b64(recall_info["display_cam_wrist"]),
                    }
                    all_images.append(self._make_image_ref(img_id, 'cam_head', cam_img=recall_cam_img))
                    all_images.append(self._make_image_ref(img_id + 1, 'cam_wrist', cam_img=recall_cam_img))
                    img_id += 2
                    history_text += step_str
                else:
                    history_text += f"{step_str}, (key frame observation omitted);\n\n"
                continue
            # =====================================================

            print("is_premature:", is_premature)

            # no reasoning
            if act in ["look_left", "look_right"]:
                step_str = f"STEP{step_num}:action:{act},target_degree:{target}"
            elif act in ["pick", "place"]:
                step_str = f"STEP{step_num}:action:{act},bbox:{box}"
            elif act in ["moveforward"]:#,move_distance:{step.get('movedistance', self.movedistance)}
                step_str = f"STEP{step_num}:action:{act}"
            else:
                step_str = f"STEP{step_num}:action:{act}"
            
            # Decide whether to append reasoning based on the switch
            if self.include_reasoning_in_history and reasoning:
                step_str += f",reasoning:{reasoning}"

            print(f"第{n}轮step内容：{step}")

            # Priority branches: branch1 (is_premature) -> branch2 (target_out_of_reach) -> branch3 (moveforward blocked) -> branch4 (is_no_cloud) -> branch5 (normal)
            # if is_premature:
            #     # branch1: manipulation skills attempted too early during navigation, target position not reached yet
            #     step_str += "Attention: The target object is out of your reachable range. The current distance exceeds the maximum arm reach."
            if is_premature and step.get("target_out_of_reach", False):
                # branch2: target position reached, but the target's physical distance exceeds the arm's limit
                step_str += "Attention: The target object is out of your reachable range. The current distance exceeds the maximum arm reach."
            elif act == "moveforward" and step.get("movedistance", self.movedistance) < 0.1 and self.collision_feedback_open:
                # branch3: forward movement blocked by an obstacle
                step_str += "Attention: Forward movement blocked. The robot has detected an obstacle directly ahead and cannot proceed forward."
            elif is_no_cloud:
                # branch4: no point cloud during manipulation; out-of-range already excluded, meaning the arm is not correctly aligned with the object
                step_str += "Attention:Your pick action for target object was not executed because the target object is not in the current field of wrist view. Please re-verify the object's visual position and output a more precise bbox."
            else:
                # branch5: normal round, keep as-is (no attention needed)
                pass

            # Check whether the step is within the history sliding window (history_maxlen) to decide whether to load image placeholders
            if (total_history_count - idx) <= self.history_maxlen:
                if use_3_cams:
                    p_top = f"<image_{img_id}>"
                    p_head = f"<image_{img_id + 1}>"
                    p_wrist = f"<image_{img_id + 2}>"
                    step_str += f"Observation at this time:TOP-DOWN CAMERA:{p_top} HEAD CAMERA:{p_head} WRIST CAMERA:{p_wrist};\n\n"
                else:
                    p_head = f"<image_{img_id}>"
                    p_wrist = f"<image_{img_id + 1}>"
                    step_str += f"Observation at this time:HEAD CAMERA:{p_head} WRIST CAMERA:{p_wrist};\n\n"
                img_id = self._append_step_images(all_images, img_id, img_idx, use_3_cams)
                history_text += step_str
            else:
                # Composite navigation likewise adds semantic marker protection so distant text is not filtered out even without images
                history_text += f"{step_str}, (observation omitted);\n\n"

        # Build current round observations (Current Observations)
        if use_3_cams:
            cur_top = f"<image_{img_id}>"
            cur_head = f"<image_{img_id + 1}>"
            cur_wrist = f"<image_{img_id + 2}>"
            current_text = f"Your current observations are as follows:TOP-DOWN CAMERA:{cur_top} HEAD CAMERA:{cur_head} WRIST CAMERA:{cur_wrist};\n"
        else:
            cur_head = f"<image_{img_id}>"
            cur_wrist = f"<image_{img_id + 1}>"
            current_text = f"Your current observations are as follows:HEAD CAMERA:{cur_head} WRIST CAMERA:{cur_wrist};\n"

        current_text += "Please continue navigating or output 'action: end' if you have reached the target."
        img_id = self._append_current_images(all_images, img_id, cam_img, use_3_cams)

        all_user_contents = self._build_user_contents(history_text + current_text, all_images)

        print(f"[Composite] Cams count: {'3' if use_3_cams else '2'}, Total images: {len(all_images)}")
        return system_prompt, all_images, all_user_contents