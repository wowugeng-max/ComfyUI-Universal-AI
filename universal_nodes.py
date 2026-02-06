import torch
import numpy as np
import comfy.utils
from .utils import *
from .api_adapters import call_universal_api


class UniversalAILoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "provider": (["Gemini", "OpenAI", "Grok", "Qwen", "Doubao", "Hailuo", "Luma"], {"default": "Gemini"}),
                "api_key": ("STRING", {"default": "", "multiline": True}),
                "model_selection": (get_combined_models(), {"default": "gemini-1.5-flash"}),
                "api_version": (["v1beta", "v1"], {"default": "v1beta"}),
                "refresh_list": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "custom_model_name": ("STRING", {"default": ""}),
                "custom_base_url": ("STRING", {"default": ""}),
                "custom_api_version": ("STRING", {"default": ""}),
                "extra_params": ("STRING", {"default": "{}", "multiline": True}),
            }
        }

    RETURN_TYPES = ("AI_CONFIG",)
    FUNCTION = "load"
    CATEGORY = "Universal_AI"

    def load(self, provider, api_key, model_selection, api_version, refresh_list, **kwargs):
        active_key = get_api_key(api_key)
        if refresh_list:
            sync_all_models(provider, active_key)
        return ({
            "provider": provider,
            "api_key": active_key,
            "model_name": kwargs.get("custom_model_name") or model_selection,
            "api_version": kwargs.get("custom_api_version") or api_version,
            "custom_base_url": kwargs.get("custom_base_url"),
            "extra_params": kwargs.get("extra_params")
        },)


class UniversalAIRunner:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ai_config": ("AI_CONFIG",),
                "system_prompt": ("STRING", {"default": "You are a helpful assistant.", "multiline": True}),
                "user_prompt": ("STRING", {"default": "", "multiline": True}),
                "max_image_size": ("INT", {"default": 1024, "min": 256, "max": 2048}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0, "max": 2.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "text": ("STRING", {"default": "", "multiline": True}),  # ← 新增独立文本模态
                "images": ("IMAGE",),
                "video": ("IMAGE",),
                "max_video_frames": ("INT", {"default": 10, "min": 1, "max": 100}),
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE", "IMAGE")
    RETURN_NAMES = ("text", "image", "video_frames")
    FUNCTION = "execute"
    CATEGORY = "Universal_AI"

    def execute(
        self,
        ai_config,
        system_prompt,
        user_prompt,
        max_image_size,
        temperature,
        seed,
        text="",
        images=None,
        video=None,
        max_video_frames=10
    ):
        try:
            safe_max_frames = int(''.join(filter(str.isdigit, str(max_video_frames))) or "10")
        except:
            safe_max_frames = 10

        pbar = comfy.utils.ProgressBar(100)
        pbar.update_absolute(10)

        empty_img = torch.zeros([1, 64, 64, 3])

        # === 组装统一多模态部件 ===
        parts = []

        if user_prompt.strip():
            parts.append({"type": "text", "data": user_prompt.strip()})

        if text.strip():
            clean_text = text.strip()
            if len(clean_text) > 10000:
                clean_text = clean_text[:10000]
                print("⚠️ [Universal AI] Extra 'text' input truncated to 10,000 characters.")
            parts.append({"type": "text", "data": clean_text})

        if images is not None:
            if images.ndim == 4:
                for i in range(images.shape[0]):
                    b64 = tensor_to_base64(images[i], max_image_size)
                    if b64:
                        parts.append({"type": "image", "data": b64})
            else:
                b64 = tensor_to_base64(images, max_image_size)
                if b64:
                    parts.append({"type": "image", "data": b64})

        if video is not None:
            num_frames = video.shape[0]
            indices = np.linspace(0, num_frames - 1, min(num_frames, safe_max_frames), dtype=int)
            for i in indices:
                b64 = tensor_to_base64(video[i], max_image_size)
                if b64:
                    parts.append({"type": "image", "data": b64})

        if not parts:
            raise ValueError("No input provided: connect at least user_prompt, text, image, or video.")

        pbar.update_absolute(40)

        try:
            res = call_universal_api(
                ai_config=ai_config,
                system_prompt=system_prompt.strip() if system_prompt.strip() else None,
                parts=parts,
                temperature=temperature,
                seed=seed
            )
            pbar.update_absolute(85)

            if res["type"] == "image":
                return (user_prompt, base64_to_tensor(res["content"]), empty_img)

            text_content = res["content"]
            video_tensor = empty_img

            if "http" in text_content and any(ext in text_content.lower() for ext in [".mp4", ".mov", "video"]):
                import re
                urls = re.findall(r'https?://[^\s]+', text_content)
                if urls:
                    pbar.update_absolute(90)
                    v_tensor = url_to_video_tensor(urls[0])
                    if v_tensor is not None:
                        video_tensor = v_tensor
                    pbar.update_absolute(100)

            return (text_content, empty_img, video_tensor)

        except Exception as e:
            return (f"❌ Error: {str(e)}", empty_img, empty_img)



# ==============================
# Set / Get Global AI Config Nodes
# ==============================

class UniversalAISetConfig:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ai_config": ("AI_CONFIG",),
                "key": ("STRING", {"default": "default"}),
            }
        }
    RETURN_TYPES = ()
    OUTPUT_NODE = True  # 表示该节点主要用于副作用（设置状态）
    FUNCTION = "set_config"
    CATEGORY = "Universal_AI/Utils"

    def set_config(self, ai_config, key="default"):
        from .utils import set_global_ai_config
        set_global_ai_config(key.strip() or "default", ai_config)
        return {}


class UniversalAIGetConfig:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "key": ("STRING", {"default": "default"}),
            }
        }
    RETURN_TYPES = ("AI_CONFIG",)
    FUNCTION = "get_config"
    CATEGORY = "Universal_AI/Utils"

    def get_config(self, key="default"):
        from .utils import get_global_ai_config
        config = get_global_ai_config(key.strip() or "default")
        if config is None:
            raise ValueError(f"No AI config found for key '{key}'. Please use 'UniversalAISetConfig' first.")
        return (config,)