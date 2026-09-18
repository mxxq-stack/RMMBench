import json
import os
import time

import requests
from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseVLMClient(ABC):
    """Base class for VLM clients (new vlm_client version), defining a unified interface.

    Keeps calling-interface compatibility with the legacy clients/base.py:
    - send(system_prompt=..., images=..., user_contents=..., max_new_tokens=..., return_prompt_data=...)
    - Returns the model text as str by default; when return_prompt_data=True, returns (content, prompt_data)

    Differences from the legacy version:
    - Added api_key: automatically attaches the Authorization: Bearer auth header
    - Supports falling back to reading api_key from an environment variable (injected by the registry)
    """

    def __init__(self, url: str, api_key: str = None,
                 retry_limit: int = 10, retry_interval: float = 4.0):
        self.url = url
        self.api_key = api_key
        self.retry_limit = retry_limit
        self.retry_interval = retry_interval

    @abstractmethod
    def build_prompt(self, system_prompt: str, images: List[Dict[str, Any]],
                     user_contents: List[Dict[str, Any]], **kwargs) -> Any:
        """
        Build the backend-specific request body.
        :param system_prompt: system prompt text
        :param images: list of images, each item containing {"id": "<image_1>", "data": "data:image/jpeg;base64,..."}
                       (in the OpenAI-compatible format images are already embedded in user_contents;
                       this parameter is only for debugging/subclass extension)
        :param user_contents: list of user contents in OpenAI multimodal format,
                              e.g. [{"type": "text", "text": "..."}, {"type": "image_url", "image_url": {...}}, ...]
        :return: request body dict
        """
        pass

    @abstractmethod
    def extract_content(self, response_data: dict) -> str:
        """Extract the text content from the response data"""
        pass

    def _build_headers(self) -> Dict[str, str]:
        """Build the request headers, attaching Bearer auth when api_key is present"""
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        return headers

    def send(self, system_prompt: str, images: List[Dict[str, Any]],
             user_contents: List[Dict[str, Any]], return_prompt_data: bool = False,
             **kwargs) -> Any:
        """
        Send the request and return the content (with exponential backoff retries).
        :param system_prompt: system prompt
        :param images: list of images
        :param user_contents: user contents
        :param return_prompt_data: whether to return the built prompt_data
        :param kwargs: forwarded to build_prompt (e.g. max_new_tokens)
        :return: returns the model-generated text (str) by default;
                 if return_prompt_data=True: returns (content, prompt_data)
        """
        prompt_data = self.build_prompt(system_prompt, images, user_contents, **kwargs)
        headers = self._build_headers()
        retry_interval = self.retry_interval

        for attempt in range(1, self.retry_limit + 1):
            try:
                print(f"[attempt {attempt}/{self.retry_limit}] request {self.url}")
                response = self._post_request(prompt_data, headers)
                response.raise_for_status()

                data = response.json()
                content = self.extract_content(data)

                # Fallback for empty-content responses: finish_reason=length means the output cap was
                # entirely consumed by the reasoning process; finish_reason=stop with empty content is an
                # occasional empty response from the server. Both cases are worth retrying.
                if not content:
                    finish_reason = None
                    try:
                        finish_reason = data["choices"][0].get("finish_reason")
                    except (KeyError, IndexError, TypeError):
                        pass
                    has_reasoning = bool(
                        (data.get("choices") or [{}])[0].get("message", {}).get("reasoning_content")
                    )
                    print(f"[警告] 第 {attempt} 次请求返回空 content "
                          f"(finish_reason={finish_reason}, reasoning_content={has_reasoning})")
                    if attempt < self.retry_limit:
                        print(f"将在 {retry_interval}s 后重试...")
                        time.sleep(retry_interval)
                        retry_interval = min(retry_interval * 1.5, 60)
                        continue
                    print("[警告] 达到重试上限仍为空，按空响应继续（上游按 action=None 处理）")

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
        return requests.post(self.url, json=prompt_data, headers=headers, timeout=300)
