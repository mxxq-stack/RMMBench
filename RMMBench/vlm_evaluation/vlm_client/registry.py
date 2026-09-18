import os
from typing import Optional

from .openai_compatible import OpenAICompatibleClient


# Backend registry: --vlm-backend name -> official API config
# base_url/model values are taken from each official API's documentation; api_key prefers an
# explicitly passed value, otherwise falls back to the corresponding environment variable
# When "max_tokens" is not configured in an entry, create_client defaults to DEFAULT_MAX_TOKENS
# Default output cap (each model allows a different maximum output; if you need the exact
# documented value for a provider, add a "max_tokens" key to the corresponding entry)
DEFAULT_MAX_TOKENS = 8192

BACKEND_CONFIGS = {
    # ==== Real backend names (valid --vlm-backend values for vlm_eval) ====

    # Gemini family (via the official OpenAI-compatible endpoint): https://ai.google.dev/gemini-api/docs/openai
    "gemini3flash": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "model": "gemini-3.8-flash",  # model name from the API docs' SDK example (the same doc's curl example uses gemini-3.5-flash; change here to switch)
        "api_key_env": "GEMINI_API_KEY",
        "extra_params": {},
    },
    "gemini3pro": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "model": "gemini-3-pro",  # not listed in the docs; named following the official naming convention
        "api_key_env": "GEMINI_API_KEY",
        "extra_params": {},
    },
    "gemini2_5pro": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "model": "gemini-2.5-pro",
        "api_key_env": "GEMINI_API_KEY",
        "extra_params": {},
    },
    # Volcano Ark Doubao-Seed: https://www.volcengine.com/docs/82379
    "seed2d0": {
        "base_url": "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
        "model": "doubao-seed-2-0-pro-260215",
        "api_key_env": "ARK_API_KEY",
        "extra_params": {"reasoning_effort": "medium"},
    },
    # Alibaba Cloud Model Studio DashScope (OpenAI-compatible mode): https://help.aliyun.com/zh/model-studio/
    "Qwen3d5": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        "model": "qwen3.8-max",  # name from the API docs
        "api_key_env": "DASHSCOPE_API_KEY",
        "extra_params": {},
    },
    "Qwen3": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        "model": "qwen3-max",  # not listed in the docs; named following the official naming convention
        "api_key_env": "DASHSCOPE_API_KEY",
        "extra_params": {},
    },

    # ==== Generic aliases (call directly by provider name) ====
    "deepseek": {
        "base_url": "https://api.deepseek.com/chat/completions",
        "model": "deepseek-flash",  # name from the API docs
        "api_key_env": "DEEPSEEK_API_KEY",
        "extra_params": {},
    },
    "seed": {
        "base_url": "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
        "model": "doubao-seed-2-0-pro-260215",
        "api_key_env": "ARK_API_KEY",
        "extra_params": {"reasoning_effort": "medium"},
    },
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        "model": "qwen3-max",
        "api_key_env": "DASHSCOPE_API_KEY",
        "extra_params": {},
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "model": "gemini-2.5-flash",
        "api_key_env": "GEMINI_API_KEY",
        "extra_params": {},
    },
    # ==== Model-name aliases (a bare model name from the API docs also routes to the corresponding provider) ====
    "deepseek-flash": {
        "base_url": "https://api.deepseek.com/chat/completions",
        "model": "deepseek-flash",
        "api_key_env": "DEEPSEEK_API_KEY",
        "extra_params": {},
    },
    "qwen3.8-max": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        "model": "qwen3.8-max",
        "api_key_env": "DASHSCOPE_API_KEY",
        "extra_params": {},
    },
    "gemini-3.8-flash": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "model": "gemini-3.8-flash",
        "api_key_env": "GEMINI_API_KEY",
        "extra_params": {},
    },
    "doubao-seed-2-0-pro-260215": {
        "base_url": "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
        "model": "doubao-seed-2-0-pro-260215",
        "api_key_env": "ARK_API_KEY",
        "extra_params": {"reasoning_effort": "medium"},
    },
}


def list_backends():
    """List all supported backend names"""
    return sorted(BACKEND_CONFIGS.keys())


def create_client(backend: str, url: Optional[str] = None, model: Optional[str] = None,
                  api_key: Optional[str] = None, retry_limit: int = 10,
                  retry_interval: float = 4.0) -> OpenAICompatibleClient:
    """
    Factory function: create the VLM client corresponding to the given backend name.
    :param backend: backend name (see BACKEND_CONFIGS, e.g. deepseek/seed/qwen/gemini)
    :param url: override the default base_url (e.g. when going through an internal gateway/proxy)
    :param model: override the default model name
    :param api_key: explicit API key; when not provided, falls back to the environment
                    variable corresponding to the backend
    :param retry_limit: maximum number of retries
    :param retry_interval: first retry interval in seconds, with exponential backoff afterwards
    """
    if backend not in BACKEND_CONFIGS:
        raise ValueError(
            f"不支持的 vlm_backend: {backend!r}, 当前支持: {list_backends()}\n"
            f"提示: --vlm-backend 接受后端别名（如 deepseek/seed2d0/gemini3flash）或 API 文档中的模型名（如 deepseek-flash）；\n"
            f"如需其他模型名，用 --vlm-backend 指后端 + registry.py 里改 model，或命令行 --vlm-url 覆盖端点"
        )
    cfg = BACKEND_CONFIGS[backend]

    resolved_url = url or cfg["base_url"]
    resolved_model = model or cfg["model"]
    resolved_api_key = api_key or os.environ.get(cfg["api_key_env"])

    return OpenAICompatibleClient(
        url=resolved_url,
        model=resolved_model,
        api_key=resolved_api_key,
        extra_params=dict(cfg.get("extra_params", {})),
        max_tokens=cfg.get("max_tokens", DEFAULT_MAX_TOKENS),
        retry_limit=retry_limit,
        retry_interval=retry_interval,
    )
