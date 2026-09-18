from typing import Any, Dict, List

from .base import BaseVLMClient


class OpenAICompatibleClient(BaseVLMClient):
    """OpenAI-compatible protocol client (covers DeepSeek / Doubao-Seed(ARK) / Qwen(DashScope) / Gemini-compatible endpoints).

    Request format (matching the official API documentation):
    Request:
        {
            "model": "<real model name>",
            "messages": [
                {"role": "system", "content": "<system prompt text>"},
                {"role": "user", "content": [
                    {"type": "text", "text": "..."},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
                    ...
                ]}
            ],
            "max_tokens": 4096,
            ...extra_params (e.g. seed's reasoning_effort)
        }
    Response: {"choices": [{"message": {"content": "..."}}]}

    Notes:
    - user_contents is passed through as-is: vlm_prompt.py already builds the standard
      OpenAI multimodal format
    - api_key is injected by the base class as the Authorization: Bearer header
    """

    def __init__(self, url: str, model: str, api_key: str = None, extra_params: dict = None,
                 max_tokens: int = None, retry_limit: int = 10, retry_interval: float = 4.0):
        super().__init__(url=url, api_key=api_key, retry_limit=retry_limit, retry_interval=retry_interval)
        self.model = model
        self.extra_params = extra_params or {}
        self.max_tokens = max_tokens  # default output cap, from the registry; takes precedence unless the caller explicitly passes max_new_tokens

    def build_prompt(self, system_prompt: str, images: List[Dict[str, Any]],
                     user_contents: List[Dict[str, Any]], max_new_tokens: int = None, **kwargs) -> dict:
        """
        Build the OpenAI chat/completions request body.
        Note: the parameter name stays max_new_tokens for backward compatibility with the
        legacy caller (vlm_prompt.py); the field actually sent is named max_tokens.
        When max_new_tokens=None, uses the default cap configured in the registry for this
        backend (no longer a fixed 4096).
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_contents},
        ]

        effective_max_tokens = max_new_tokens if max_new_tokens is not None else (self.max_tokens or 4096)
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": effective_max_tokens,
        }
        # Merge backend-specific extension parameters (e.g. seed's reasoning_effort); caller kwargs take the highest priority
        if self.extra_params:
            payload.update(self.extra_params)
        for k, v in kwargs.items():
            if k.startswith("extra_"):
                payload[k[len("extra_"):]] = v
        return payload

    def extract_content(self, response_data: dict) -> str:
        """Extract the text from choices[0].message.content"""
        if "choices" not in response_data:
            raise KeyError(f"响应中没有 'choices' 字段: {list(response_data.keys())}")
        return response_data["choices"][0]["message"]["content"]
