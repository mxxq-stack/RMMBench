from typing import List, Dict, Any
from .base import BaseVLMClient


class PromptImageMapFormatClient(BaseVLMClient):
    """Prompt + Image Map format client.

    Format:
    - Request: {"prompt": "...<image_1>...", "image_url": {"<image_1>": "data:image/..."}}
    - Send: json=payload (raw JSON body)
    - Response: returns the text content directly
    """

    def build_prompt(self, system_prompt: str, images: List[Dict[str, Any]], user_contents: List[Dict[str, str]], **kwargs) -> dict:
        """
        Build the Prompt + Image Map format.
        Converts the standardized input into a prompt string.
        """
        text_parts = []

        # 1. Add the system prompt
        if system_prompt:
            text_parts.append(system_prompt.strip())
            text_parts.append("")  # blank line separator

        # 2. Extract the user content (text parts)
        # At this point user_contents[0]["text"] contains: the lead-in text + all previous Step texts + the current observation text
        for item in user_contents:
            if item.get("type") == "text":
                text_content = item.get("text", "").strip()
                if text_content:
                    text_parts.append(text_content)

        # Combine into the final full-text prompt sent to the backend
        prompt = "\n".join(text_parts)

        # 3. Build the image_url dict, mapping <image_N> to the actual base64 data
        image_map = {}
        for img in images:
            img_id = img.get("id", f"<image_{images.index(img) + 1}>")
            image_map[img_id] = img["data"]

        return {
            "prompt": prompt,
            "image_url": image_map,
        }

    def extract_content(self, response_data: dict) -> str:
        """Extract the text content from the response data"""
        if isinstance(response_data, str):
            return response_data

        if isinstance(response_data, dict):
            if "completions" in response_data and isinstance(response_data["completions"], list):
                if len(response_data["completions"]) > 0 and "text" in response_data["completions"][0]:
                    return response_data["completions"][0]["text"]

            if "choices" in response_data and isinstance(response_data["choices"], list):
                if len(response_data["choices"]) > 0:
                    choice = response_data["choices"][0]
                    if "text" in choice:
                        return choice["text"]
                    if "message" in choice and "content" in choice["message"]:
                        return choice["message"]["content"]

            if "text" in response_data: return response_data["text"]
            if "content" in response_data: return response_data["content"]
            if "response" in response_data: return response_data["response"]
            if "result" in response_data: return response_data["result"]
            return str(response_data)

        return str(response_data)