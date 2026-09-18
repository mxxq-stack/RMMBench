import base64
import json
import re
import time
import requests
import cv2
import numpy as np
import copy
from collections import deque
from RMMBench.utils.utils import get_direction_from_euler


class VLAMessageHandler:
    def __init__(self, history_maxlen=8, img_size=(480, 480),vlm_url=None):
        """
        Initialize the robot instruction handler
        :param history_maxlen: maximum length of the history queue
        :param img_size: actual image size (width, height), used for coordinate restoration
        """
        assert vlm_url is not None
        self.vlm_url = vlm_url
        self.allow_entities = None
        self.robot_init_info = None
        self.img_list = []
        self.deque_prompts = deque(maxlen=history_maxlen)
        self.m = 1
        self.vlm_response = ""
        self.img_width, self.img_height = img_size

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

    def _get_system_content(self, instruction, static_cam):

        return f"""You are a robotic arm composed of seven links with a white movable chassis in given room. You need to complete the tasks according to human instructions. 
We provide an Available_Actions set and the corresponding explanations for each action. Each step, you should select one action from Available_Actions.

I will provide you with images from two different perspectives:
1.STATIONARY CAMERA: This perspective automatically toggles based on task progress.
-Navigation Phase: Provides a Top-down View of the entire scene for global pathfinding and positioning.
-Interaction Phase: As you approach the target, the view automatically switches to a Head View (human-like perspective).
-Important Note: The switch in perspective is a clear sitgnal that you have arrived in the vicinity of the arget, but it does not necessarily mean you are directly facing it.

2.WRIST CAMERA: Mounted on the robotic arm's end effector, this camera moves in real-time with the arm and provides a detailed close-up perspective.

Current STATIONARY CAMERA VIEW: {static_cam}
Use the exact object names from the list: {self.allow_entities}
This is your task: {instruction}

After executing the action, you will receive a new observation image showing the updated state. To complete your task, you should output your action from the Available Action Library.

Available Action Library:
- pick <object>: Move from the current position to a suitable location for grasping the target object and close the gripper to perform the grasp.
- place to <container>: Move from the current position to a suitable location for placing the target object to target container and  open the gripper to perform the place.
- lift: Lift your end effector vertically.
- pull: Pull your hand out parallel while maintaining the gripper state.
- push: Push your end effector in while maintaining the gripper state.
- close_gripper: Close your gripper.
- open_gripper: Open your gripper.
- observe: Reset the end-effector to the default position to gain a clear, wide-angle view or to stabilize the camera orientation before executing navigation skills.
- open_door: Given that the gripper has already grasped the handle, open the door, then release the handle.
- close_door: Push or pull the door along its path until fully closed.
- moveforward: The robot moves forward a specific distance in the viewing direction of the wrist camera.Before you feel you have reached the right position, you can use it multiple times in one direction.
- rotate_right: 90° clockwise base rotation: Centers objects from the right periphery of the wrist camera for a clearer view, while shifting left-side objects out of frame.
- rotate_left: 90° counter-clockwise base rotation: Centers objects from the left periphery of the wrist camera for a clearer view, while shifting right-side objects out of frame.
- end: If you think you have completed the task, please output 'end'.

Please follow these output rules:
- If your action is 'pick', output the bounding box coordinates of the target object in the wrist camera image.
- If your action is 'place', output the bounding box coordinates of the target container in the stationary camera image.
- For all other actions(except to navigation actions), do not output any bounding box coordinates.


Navigation Task Specific Rules (Ignore if the task is not navigation):
- Navigation skills include moveforward, rotate_right, and rotate_left; when executing these skills, output the bounding box coordinates of the target object in the Top-down view and the robot's current orientation based on the Top-down view image space, with the orientation output format strictly restricted to "[top/bottom/right/left]".
- During navigation tasks, the stationary camera will switch to a Top-down view. Use this perspective to track the robot's global position and treat it as the primary guidance for navigation.
- Long-distance Navigation Strategy: If the robot is far from the target, first determine the target's orientation via the Top-down view, use rotate skills to select the direction based on the Top-down view, and then output moveforward to approach. It is strictly forbidden to frequently fine-tune rotation at long distances just to see the object clearly.
- When far away, you can combine the Top-down view with the local perspective of the WRIST CAMERA. The Wrist Camera is primarily used to observe whether there are obstacles directly ahead, do not use the Wrist Camera perspective as a tool for searching for the target. As long as the Top-down view shows the target is in front of the robot, execute moveforward decisively even if the target is not perfectly centered in the Wrist Camera.
- Only when the perspective switches from the Top-down view to the Head view should you use rotation for final pose fine-tuning to ensure the target remains clear in the Wrist Camera."

Your action must strictly follow the format: action:Your Action (for example: action:pick apple ).
Output its bbox coordinates using JSON format. The box_2d should be [y_min, x_min, y_max, x_max] normalized to 0-1000,do not include any image data, base64 strings, or patches in the output"""



    def build_prompt_messages(self, n, img, instruction, down_to_head, action=None, vlm_response=None):
        """
        Build the message body sent to the VLM
        :param n: current round index (0 is the initial round)
        :param img: list containing two images [stationary_img, wrist_img]
        :param instruction: task instruction
        :param action: action executed in the previous step (required for n>=1)
        :param vlm_response: VLM reply from the previous step (required for n>=1)
        """
        robot_init_orient = get_direction_from_euler(self.robot_init_info["euler"][2])
        static_cam = "Head"
        # 1. Convert and store the current images
        cam_img = {
            "cam_right": self._encode_b64(img[0]),
            "cam_left": self._encode_b64(img[1]),
            "cam_top": self._encode_b64(img[2]),
            "cam_wrist": self._encode_b64(img[3]),
            "cam_head": self._encode_b64(img[4]),
            "cam_opposite": self._encode_b64(img[5]),
        }

        self.img_list.append(cam_img)

        if down_to_head is not False:
            wrist_img = self.img_list[n]["cam_wrist"]
            stationary_img = self.img_list[n]["cam_head"]
        else:
            wrist_img = self.img_list[n]["cam_wrist"]
            stationary_img = self.img_list[n]["cam_top"]
            static_cam = "Top-down"

        print("robot_init_orient:", robot_init_orient)
        print("static_cam:", static_cam)
        print("down_to_head", down_to_head)
        print("allow_objects:", self.allow_entities)

        # 2. Build the base System Message
        messages_template = [
            {
                'role': 'system',
                'content': [{'type': 'text', 'text': self._get_system_content(instruction, static_cam)}]
            },
            {
                'role': 'user',
                'content': []
            }
        ]

        # 3. Handle Content for different phases
        if n == 0:
            # Initial observation
            content = [
                {'type': 'text',
                 'text': "These are your initial observation. Please select an action from the Available Action Library and output bbox coordinates."},
                {'type': 'text', 'text': "1. STATIONARY CAMERA:<image_1>"},
                {'type': 'image_url', 'image_url': {'url': f"data:image/jpeg;base64,{stationary_img}"}},
                {'type': 'text', 'text': "2. WRIST CAMERA:<image_2>"},
                {'type': 'image_url', 'image_url': {'url': f"data:image/jpeg;base64,{wrist_img}"}},
                {'type': 'text',
                 'text': "Each action will cause the image to change. You can only output one primitive operation at a time."}
            ]
            # Initial orientation
            if down_to_head is False:
                content.append({'type': 'text',
                                'text': f"The initial orientation of the robot is {robot_init_orient}. "})
            self.deque_prompts.append(content)
        else:
            # Subsequent feedback rounds
            new_content = [
                {"type": "text",
                 "text": f"Your history action, corresponding feedback and observation are as follows: Your response: {vlm_response}"},
                {'type': 'text',
                 'text': f"feedback：These are the state after you executing {action}.Please select an action from the Available Action Library and output bbox coordinates.For navigation task, the static camera provides a top-down view of the scene, from which you can see the robot and the entire scene layout in an overhead perspective."},
                {'type': 'text', 'text': f"1. STATIONARY CAMERA:<image_{self.m}>"},
                {'type': 'image_url', 'image_url': {'url': f"data:image/jpeg;base64,{stationary_img}"}},
                {'type': 'text', 'text': f"2. WRIST CAMERA:<image_{self.m + 1}>"},
                {'type': 'image_url', 'image_url': {'url': f"data:image/jpeg;base64,{wrist_img}"}}
            ]
            self.m += 2
            self.deque_prompts.append(new_content)

        for item in self.deque_prompts:
            messages_template[1]["content"].extend(item)

        print(f"History rounds: {len(self.deque_prompts)}")

        return {
            "messages": messages_template,
            "max_new_tokens": 4096
        }


    def parse_vlm_response(self, content):
        """
        Parse the text returned by the VLM, extracting the action and coordinates
        Compatible format 1 (plain text): action: pick apple \n box_2d: [393, 278, 461, 354]
        Compatible format 2 (JSON): {"action": "pick apple", "box_2d": [393, 278, 461, 354]}
        """
        bbox_2d_n = None
        action_type = None
        target_object = None
        pts = None

        place_to_match = re.search(r'action:\s*place\s+to\s+(\w+)', content, re.IGNORECASE)
        if place_to_match:
            action_type = 'place'
            target_object = place_to_match.group(1)
        else:
            action_match = re.search(r'action["\s:]+(\w+)(?:\s+(\w+))?', content, re.IGNORECASE)
            if action_match:
                action_type = action_match.group(1).lower()
                target_object = action_match.group(2)

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
                    print(f"BBox坐标数量不足: 期望4个，实际得到{len(bbox_raw)}个")
            except Exception as e:
                print(f"BBox解析逻辑出错: {e}")
        else:
            print(f"未能提取到 box_2d 数组内容。Content: {content}")

        return bbox_2d_n, action_type, target_object, pts

    def parse_grasp_response(self,content):
        return 1


    def request_vlm(self,
                    img,
                    n,
                    instruction,
                    action=None,
                    retry_limit=10,
                    down_to_head=None,
                    robot_init_info=None,
                    allow_entities=None,
                    ):
        """Execute the network request and process the result"""
        self.robot_init_info = robot_init_info
        self.allow_entities = allow_entities
        messages = self.build_prompt_messages(n, img, instruction, action=action, vlm_response=self.vlm_response,
                                              down_to_head=down_to_head)
        payload = json.dumps(messages)
        headers = {'Content-Type': 'application/json'}
        retry_interval = 4

        for attempt in range(1, retry_limit + 1):
            try:
                print(f"VLM正在响应 (第{n + 1}轮, 尝试 {attempt})....")
                response = requests.post(self.vlm_url, data=payload, headers=headers, timeout=120)
                response.raise_for_status()

                data_dict = response.json()
                content = data_dict["choices"][0]["message"]["content"]


                self.vlm_response = content
                print("content:", content)

                bbox_2d, action_type, target_object, pts = self.parse_vlm_response(content)

                print(f"解析结果: action={action_type}, target={target_object},bbox={bbox_2d}")
                return bbox_2d, action_type, target_object, content, pts

            except Exception as e:
                print(f"请求异常: {e}")
                if attempt < retry_limit:
                    time.sleep(retry_interval)
                    retry_interval = min(retry_interval * 1.5, 60)

        return None



    def _get_grasp_selection_system_content(self,target_entity):
        return f"""Role: You are an expert in robotic arm grasp planning, specializing in assessing the safety of grasp paths within constrained spaces (e.g., refrigerators, deep cabinets, trays).

Task: The target object and grasp position are fixed. Based on the WRIST CAMERA perspective(Slightly tilted downward), select the Grasp Orientation that best fits the environmental constraints from the provided candidates.

Target object: {target_entity}

Visual Guide:
-Indicator: Green open-ended rectangular wireframes.
-Physical Meaning: The opening direction of the wireframe indicates the gripper's Approach Axis.

Please evaluate whether the robotic end-effector (wrist) will collide with environmental boundaries if grasping according to that orientation.

Output in the following format: 
-choice: [1/2/3]
-reason: <Specific visual derivation of the conflict between environmental boundaries (e.g., top cover, side wall distance) and the wireframe's opening direction>"""

    def select_grasp_prompt(self,viz_list,target_entity):
        viz_dict = dict()
        viz_g_list = []
        for i in viz_list:
            viz_g_list.append(self._encode_b64(i))
        for i,viz in enumerate(viz_g_list):
            if i ==0:
                viz_dict['g_graspnet'] = viz_g_list[0]
            if i ==1:
                viz_dict['g_horizontal'] = viz_g_list[1]
            if i ==2:
                viz_dict['g_vertical'] = viz_g_list[2]


        grasp_content = [
                {'type': 'text',
                 'text': "Please examine the following images to select the best grasp orientation."},
                {'type': 'text', 'text': "OPTION 1 <image_1>"},
                {'type': 'image_url', 'image_url': {'url': f"data:image/jpeg;base64,{viz_dict['g_graspnet']}"}},
                {'type': 'text', 'text': "OPTION 2 <image_2>"},
                {'type': 'image_url', 'image_url': {'url': f"data:image/jpeg;base64,{viz_dict['g_horizontal']}"}},
                {'type': 'text', 'text': "OPTION 3 <image_3>"},
                {'type': 'image_url', 'image_url': {'url': f"data:image/jpeg;base64,{viz_dict['g_vertical']}"}},
                {'type': 'text',
                 'text': "Based on the visual guide in the system prompt, compare the green grasping posture with the environment and output your choice."}
            ]


        # 3. Assemble the final message
        messages_grasp = [
            {
                'role': 'system',
                'content': [{'type': 'text', 'text': self._get_grasp_selection_system_content(target_entity)}]
            },
            {
                'role': 'user',
                'content': grasp_content
            }
        ]

        return {
            "messages": messages_grasp,
            "max_new_tokens": 4096
        }

    def request_grasp(self,viz_list,target_entity,retry_limit=10):
        messages = self.select_grasp_prompt(
            viz_list=viz_list,
            target_entity = target_entity
        )
        payload = json.dumps(messages)
        headers = {'Content-Type': 'application/json'}
        retry_interval = 4

        for attempt in range(1, retry_limit + 1):
            try:
                print(f"VLM正在响应 ,( 尝试 {attempt})....")
                response = requests.post(self.vlm_url, data=payload, headers=headers, timeout=120)
                response.raise_for_status()

                data_dict = response.json()
                content = data_dict["choices"][0]["message"]["content"]

                print("content:", content)

                option = self.parse_grasp_response(content)

                print(f"解析结果: grasp option: {option}")
                return option

            except Exception as e:
                print(f"请求异常: {e}")
                if attempt < retry_limit:
                    time.sleep(retry_interval)
                    retry_interval = min(retry_interval * 1.5, 60)

        return option

    def reset_session(self):
        """Reset all state, used for a new task"""
        self.img_list = []
        self.deque_prompts.clear()
        self.m = 1