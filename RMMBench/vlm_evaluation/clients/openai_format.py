import json
import requests
from typing import List, Dict, Any
import re
from .base import BaseVLMClient


class OpenAIFormatClient(BaseVLMClient):
    """OpenAI standard-format client.

    Format: OpenAI-style messages array
    Request: {"messages": [{"role": "system", ...}, {"role": "user", ...}], "max_new_tokens": 4096}
    Response: {"choices": [{"message": {"content": "..."}}]}
    """

    def build_prompt(self, system_prompt: str, images: List[Dict[str, Any]], user_contents: List[Dict[str, str]],
                     max_new_tokens: int = 4096) -> dict:
        """
        Build the OpenAI-format messages array.
        Keeps the interleaved format of user_contents as-is, without cleaning or splitting.
        """
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            },
            {
                "role": "user",
                "content": user_contents
            }
        ]

        return {
            "model": "default",
            "messages": messages,
            "max_new_tokens": max_new_tokens
        }

    def extract_content(self, response_data: dict) -> str:
        """Extract the text from choices[0].message.content"""
        if "choices" not in response_data:
            raise KeyError(f"响应中没有 'choices' 字段: {response_data.keys()}")
        return response_data["choices"][0]["message"]["content"]

    def _post_request(self, prompt_data: dict, headers: dict):
        """OpenAI format is sent via data=json.dumps(payload)"""
        return requests.post(
            self.url,
            data=json.dumps(prompt_data),
            headers=headers,
            timeout=120
        )