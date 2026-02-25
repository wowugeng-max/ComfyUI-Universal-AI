import base64
import time
from urllib.parse import urlparse
from .base import BaseAdapter
from ..utils import api_session, extract_all_text, safe_process_image, ModelCapability
from .factory import AdapterFactory

@AdapterFactory.register("Qwen")
class QwenAdapter(BaseAdapter):
    provider_name = "Qwen"  # 也可用于自动发现

    def call(self, ai_config, system_prompt, parts, temperature, seed):
        api_key = ai_config["api_key"]
        model_name = ai_config["model_name"]
        extra_params = ai_config.get("extra_params", {})
        base_url = ai_config.get("custom_base_url") or "https://dashscope.aliyuncs.com/compatible-mode/v1"

        capability = self.get_capability(model_name, ai_config.get("provider"))

        if capability == ModelCapability.IMAGE_GEN:
            return self._image_gen_call(api_key, model_name, parts, base_url, extra_params)
        elif capability == ModelCapability.AUDIO_GEN:
            raise NotImplementedError("Audio generation not yet implemented for Qwen")
        elif capability == ModelCapability.VIDEO_GEN:
            raise NotImplementedError("Video generation not yet implemented for Qwen")
        elif capability == ModelCapability.VISION:
            return self._vision_call(ai_config, system_prompt, parts, temperature, base_url, extra_params)
        else:
            return self._chat_call(ai_config, system_prompt, parts, temperature, base_url, extra_params)

    def _chat_call(self, ai_config, system_prompt, parts, temperature, base_url, extra_params):
        api_key = ai_config["api_key"]
        model_name = ai_config["model_name"]
        chat_url = f"{base_url}/chat/completions"

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

        resp = api_session.post(chat_url, json=payload, headers=headers, timeout=120)
        if resp.status_code != 200:
            raise RuntimeError(f"Qwen Chat API Error: {resp.text}")
        result = resp.json()
        message = result["choices"][0]["message"]
        if "audio" in message:
            return {"type": "audio", "content": message["audio"].get("data")}
        return {"type": "text", "content": message.get("content", "")}

    def _vision_call(self, ai_config, system_prompt, parts, temperature, base_url, extra_params):
        """视觉理解模型调用，复用聊天逻辑"""
        return self._chat_call(ai_config, system_prompt, parts, temperature, base_url, extra_params)

    def _filter_params_for_model(self, model_name, params):
        """根据模型名称过滤不支持的参数"""
        supported_params = {}
        model_lower = model_name.lower()
        if "z-image-turbo" in model_lower:
            allowed = {"prompt_extend", "size", "seed"}
        elif "qwen-image" in model_lower:
            allowed = {"negative_prompt", "prompt_extend", "watermark", "size", "seed", "n"}
        else:
            allowed = set(params.keys())  # 默认全部支持

        for key, value in params.items():
            if key in allowed:
                supported_params[key] = value
            else:
                print(f"Warning: Parameter '{key}' is not supported by model '{model_name}' and will be ignored.")
        return supported_params

    def _image_gen_call(self, api_key, model_name, parts, base_url, extra_params):
        prompt = extract_all_text(parts)
        if not prompt:
            raise ValueError("Image generation requires a text prompt.")

        model_lower = model_name.lower()
        # 如果模型是 Qwen-Image 或 Z-Image，优先使用 DashScope 原生同步接口
        if "qwen-image" in model_lower or "z-image" in model_lower:
            # 构建原生 API 基础 URL
            if "dashscope.aliyuncs.com" in base_url:
                # 从 OpenAI 兼容地址转换为原生地址
                dashscope_url = base_url.replace("compatible-mode/v1", "api/v1")
            else:
                # 如果用户自定义了域名，使用其根域名加 /api/v1
                parsed = urlparse(base_url)
                dashscope_url = f"{parsed.scheme}://{parsed.netloc}/api/v1"

            gen_url = f"{dashscope_url}/services/aigc/multimodal-generation/generation"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": model_name,
                "input": {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"text": prompt}
                            ]
                        }
                    ]
                },
                "parameters": self._filter_params_for_model(model_name, extra_params)
            }

            resp = api_session.post(gen_url, json=payload, headers=headers, timeout=120)
            if resp.status_code == 200:
                data = resp.json()
                try:
                    image_url = data["output"]["choices"][0]["message"]["content"][0]["image"]
                    img_resp = api_session.get(image_url, timeout=60)
                    img_resp.raise_for_status()
                    img_b64 = base64.b64encode(img_resp.content).decode()
                    return {"type": "image", "content": img_b64}
                except (KeyError, IndexError) as e:
                    print(f"Unexpected response structure from DashScope sync API: {data}")
                    # 降级到其他方法
            else:
                print(f"DashScope sync image gen failed with status {resp.status_code}, response: {resp.text}")
                # 降级

        # 备选：尝试 OpenAI 兼容的 /images/generations 端点
        gen_url = f"{base_url}/images/generations"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model_name,
            "prompt": prompt,
            **extra_params
        }
        resp = api_session.post(gen_url, json=payload, headers=headers, timeout=120)
        if resp.status_code == 200:
            data = resp.json()
            if "data" in data and len(data["data"]) > 0:
                img_item = data["data"][0]
                if "b64_json" in img_item:
                    return {"type": "image", "content": img_item["b64_json"]}
                elif "url" in img_item:
                    img_resp = api_session.get(img_item["url"], timeout=60)
                    img_resp.raise_for_status()
                    img_b64 = base64.b64encode(img_resp.content).decode()
                    return {"type": "image", "content": img_b64}

        # 最后尝试异步任务（旧接口，作为保底）
        return self._image_gen_async_call(api_key, model_name, prompt, extra_params)

    def _image_gen_async_call(self, api_key, model_name, prompt, extra_params):
        """保留原有的异步调用作为降级方案（未修改）"""
        submit_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text2image/image-synthesis"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        task_payload = {
            "model": model_name,
            "input": {"prompt": prompt},
            "parameters": extra_params
        }
        resp = api_session.post(submit_url, json=task_payload, headers=headers, timeout=30)
        if resp.status_code != 200:
            print(f"Async submit failed: {resp.text}")
            raise RuntimeError(f"Qwen Async Submit Failed: {resp.text}")

        resp_data = resp.json()
        task_id = resp_data.get("output", {}).get("task_id")
        if not task_id:
            raise RuntimeError(f"No task_id in response: {resp_data}")

        check_url = f"https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}"
        start_time = time.time()
        while time.time() - start_time < 180:
            time.sleep(3)
            try:
                r_resp = api_session.get(check_url, headers=headers, timeout=10)
                if r_resp.status_code != 200:
                    continue
                r = r_resp.json()
                status = r.get("output", {}).get("task_status")
                if status == "SUCCEEDED":
                    results = r.get("output", {}).get("results", [])
                    if not results:
                        continue
                    img_item = results[0]
                    if "b64_json" in img_item:
                        return {"type": "image", "content": img_item["b64_json"]}
                    elif "url" in img_item:
                        img_resp = api_session.get(img_item["url"])
                        img_b64 = base64.b64encode(img_resp.content).decode()
                        return {"type": "image", "content": img_b64}
                elif status in ["FAILED", "CANCELED"]:
                    raise RuntimeError(f"Qwen Async Gen Failed: {r.get('output', {}).get('message')}")
            except Exception as e:
                if "failed" in str(e).lower():
                    raise e
                continue
        raise RuntimeError("Qwen Async Image Generation Timeout")