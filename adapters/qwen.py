from .base import BaseAdapter
from ..utils import api_session, safe_process_image

class QwenAdapter(BaseAdapter):
    def call(self, ai_config, system_prompt, parts, temperature, seed):
        api_key = ai_config["api_key"]
        model_name = ai_config["model_name"]
        extra_params = ai_config.get("extra_params", {})
        base_url = ai_config.get("custom_base_url") or "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        user_content = []
        for part in parts:
            if part["type"] == "text":
                user_content.append({"type": "text", "text": part["data"]})
            elif part["type"] == "image":
                processed_url = safe_process_image(part["data"])
                if processed_url:
                    user_content.append({"type": "image_url", "image_url": {"url": processed_url}})

        messages.append({"role": "user", "content": user_content})
        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            **extra_params
        }

        resp = api_session.post(base_url, json=payload, headers=headers, timeout=120)
        if resp.status_code != 200:
            raise RuntimeError(f"Qwen API Error: {resp.text}")
        result = resp.json()
        message = result["choices"][0]["message"]
        if "audio" in message:
            return {"type": "audio", "content": message["audio"].get("data")}
        return {"type": "text", "content": message.get("content", "")}