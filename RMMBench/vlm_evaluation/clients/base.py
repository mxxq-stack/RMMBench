import json
import os
import time
import requests
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple


class BaseVLMClient(ABC):
    """Base class for VLM clients, defining a unified interface.

    Design principles:
    - build_prompt: convert the standardized input (system prompt, image list, text content)
                    into the backend-specific prompt format
    - send: send the request and process the response
    - Each backend decides on its own how to assemble the prompt (string or messages array)
    """

    def __init__(self, url: str, retry_limit: int = 10, retry_interval: float = 4.0):
        self.url = url
        self.retry_limit = retry_limit
        self.retry_interval = retry_interval

    @abstractmethod
    def build_prompt(self, system_prompt: str, images: List[Dict[str, Any]], user_contents: List[Dict[str, str]]) -> Any:
        """
        Build the backend-specific prompt format.
        :param system_prompt: system prompt text
        :param images: list of images, each item containing {"id": "<image_1>", "data": "base64..."}
        :param user_contents: list of user contents, each item containing {"type": "text", "text": "..."}
        :return: backend-specific request body (Gemini: messages array, Qwen: prompt string, etc.)
        """
        pass

    @abstractmethod
    def extract_content(self, response_data: dict) -> str:
        """
        Extract the text content from the response data.
        """
        pass

    def send(self, system_prompt: str, images: List[Dict[str, Any]], user_contents: List[Dict[str, str]], return_prompt_data: bool = False, **kwargs) -> Any:
        """
        Send the request and return the content.
        :param system_prompt: system prompt
        :param images: list of images
        :param user_contents: user contents
        :param return_prompt_data: whether to return the built prompt_data
        :return: returns the model-generated text (str) by default;
                 if return_prompt_data=True: returns (content, prompt_data)
        """
        prompt_data = self.build_prompt(system_prompt, images, user_contents)
        # print(f"[DEBUG] Actual payload: {json.dumps(prompt_data, indent=2)[:20000]}")
        # exit()
        # =========================================================
        # [DEBUG TEST MODE]: skip the real HTTP request and return a placeholder directly
        # To restore: comment out the DEBUG block below and uncomment the original code block
        # =========================================================

        # ---------- DEBUG block (start) ----------
        # content = "action: placeholder\nreasoning: test mode, vlm request skipped"
        # print(f"[DEBUG TEST MODE] VLM request skipped, returning placeholder content")
        #
        # if return_prompt_data:
        #     return content, prompt_data
        # return content
        # ---------- DEBUG block (end) ----------

        # =========================================================
        # [Original code]: real VLM HTTP request
        # =========================================================
        headers = {'Content-Type': 'application/json'}
        retry_interval = self.retry_interval

        for attempt in range(1, self.retry_limit + 1):
            try:
                print(f"[尝试 {attempt}/{self.retry_limit}] 请求 {self.url}")
                response = self._post_request(prompt_data, headers)
                response.raise_for_status()

                data = response.json()
                content = self.extract_content(data)

                if return_prompt_data:
                    return content, prompt_data
                return content

            except requests.exceptions.RequestException as e:
                print(f"[尝试 {attempt}/{self.retry_limit}] 请求失败: {e}")
                if attempt < self.retry_limit:
                    print(f"将在 {retry_interval}s 后重试...")
                    time.sleep(retry_interval)
                    retry_interval = min(retry_interval * 1.5, 60)
            except (KeyError, json.JSONDecodeError) as e:
                print(f"[尝试 {attempt}/{self.retry_limit}] 响应解析失败: {e}")
                if attempt < self.retry_limit:
                    time.sleep(retry_interval)
                    retry_interval = min(retry_interval * 1.5, 60)

        raise Exception(f"达到重试上限 ({self.retry_limit})，请求失败")

    def _post_request(self, prompt_data: Any, headers: dict):
        """
        Actually send the HTTP request; subclasses may override.
        """
        return requests.post(self.url, json=prompt_data, headers=headers, timeout=120)
