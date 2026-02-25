import time
import base64
from .base import BaseAdapter
from ..utils import api_session, safe_process_image, extract_all_text
from .factory import AdapterFactory

@AdapterFactory.register("Doubao")
class DoubaoAdapter(BaseAdapter):
    provider_name = "Doubao"  # 也可用于自动发现

    def call(self, ai_config, system_prompt, parts, temperature, seed):
        api_key = ai_config["api_key"]
        model_name = ai_config["model_name"]
        extra_params = ai_config.get("extra_params", {})

        # 判断是否为图像生成模型
        if "[IMAGE]" in ai_config.get("model_name", ""):
            prompt = extract_all_text(parts)
            return self._image_gen_call(api_key, model_name, prompt, extra_params)
        else:
            return self._chat_call(ai_config, system_prompt, parts, temperature, extra_params)

    def _chat_call(self, ai_config, system_prompt, parts, temperature, extra_params):
        api_key = ai_config["api_key"]
        model_name = ai_config["model_name"]
        base_url = ai_config.get("custom_base_url") or "https://ark.cn-beijing.volces.com/api/v3/chat/completions"

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
        resp.raise_for_status()
        result = resp.json()
        message = result["choices"][0]["message"]
        if "audio" in message:
            return {"type": "audio", "content": message["audio"].get("data")}
        return {"type": "text", "content": message.get("content", "")}

    def _image_gen_call(self, api_key, model_name, prompt, extra_params):
        submit_url = "https://ark.cn-beijing.volces.com/api/v3/cv/generation_task"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {"model": model_name, "prompt": prompt, **extra_params}
        resp = api_session.post(submit_url, json=payload, headers=headers, timeout=30)
        if resp.status_code != 200:
            raise RuntimeError(f"Doubao Image Gen Submit Failed: {resp.text}")

        task_id = resp.json().get("task_id")
        check_url = f"https://ark.cn-beijing.volces.com/api/v3/cv/get_task_result?task_id={task_id}"

        start_time = time.time()
        while time.time() - start_time < 180:
            time.sleep(3)
            try:
                r_resp = api_session.get(check_url, headers=headers, timeout=10)
                if r_resp.status_code != 200:
                    continue
                r = r_resp.json()
                if r.get("status") == "success":
                    data_list = r.get("data", [])
                    if not data_list:
                        continue
                    img_item = data_list[0]
                    if "b64_json" in img_item:
                        return {"type": "image", "content": img_item["b64_json"]}
                    elif "url" in img_item:
                        img_resp = api_session.get(img_item["url"])
                        return {"type": "image", "content": base64.b64encode(img_resp.content).decode()}
                elif r.get("status") == "failed":
                    raise RuntimeError(f"Doubao Gen Failed: {r.get('reason')}")
            except Exception as e:
                if "failed" in str(e):
                    raise e
                continue
        raise RuntimeError("Doubao Image Generation Timeout")