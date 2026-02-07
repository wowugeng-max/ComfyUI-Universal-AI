import torch
import numpy as np
import comfy.utils
from .utils import *
from .api_adapters import call_universal_api
from server import PromptServer
from aiohttp import web
import time
from .utils import get_api_key, get_combined_models, set_global_ai_config, get_global_ai_config, _GLOBAL_AI_CONFIG


# ==============================
# 后端 API 路由注册
# ==============================

@PromptServer.instance.routes.get("/universal_ai/get_models")
async def get_models_endpoint(request):
    provider = request.query.get("provider", "")
    models = get_combined_models(provider=provider)
    return web.json_response(models)

@PromptServer.instance.routes.get("/universal_ai/get_all_keys")
async def get_all_keys_endpoint(request):
    from .utils import _GLOBAL_AI_CONFIG
    keys = list(_GLOBAL_AI_CONFIG.keys())
    if not keys:
        return web.json_response(["(Wait) Run Loader + Set Node first"])
    return web.json_response(keys)


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
            },
            "hidden": {"unique_id": "UNIQUE_ID"}, 
        }

    RETURN_TYPES = ("AI_CONFIG",)
    FUNCTION = "load"
    CATEGORY = "Universal_AI"

    def load(self, provider, api_key, model_selection, api_version, refresh_list, unique_id=None, **kwargs):
        active_key = get_api_key(api_key)
        return ({
            "provider": provider,
            "api_key": active_key,
            "model_name": kwargs.get("custom_model_name") or model_selection,
            "api_version": kwargs.get("custom_api_version") or api_version,
            "custom_base_url": kwargs.get("custom_base_url"),
            "extra_params": kwargs.get("extra_params"),
            "source_node": f"Loader_{unique_id}" if unique_id else "Direct_Loader",
            "_timestamp": time.time()
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
                "text": ("STRING", {"default": "", "multiline": True}),
                "images": ("IMAGE",),
                "video": ("IMAGE",),
                "max_video_frames": ("INT", {"default": 10, "min": 1, "max": 100}),
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE", "IMAGE")
    RETURN_NAMES = ("text", "image", "video_frames")
    FUNCTION = "execute"
    CATEGORY = "Universal_AI"

    def execute(self, ai_config, system_prompt, user_prompt, max_image_size, temperature, seed, **kwargs):
        # 💡 核心改动：使用 Comfy 内部的 ProgressBar
        # 它会自动处理节点内的绿色填充条
        pbar = comfy.utils.ProgressBar(100)
        
        source_info = ai_config.get("source_node", "Unknown_Source")
        provider = ai_config.get("provider")
        model = ai_config.get("model_name")
        
        print(f"🕵️ [Universal AI] Runner Starting...")
        pbar.update(10) # 节点亮起 10%

        # 组装消息
        parts = []
        combined_text = (kwargs.get("text", "") + "\n" + user_prompt).strip()
        if combined_text:
            parts.append({"type": "text", "data": combined_text})
        
        if "images" in kwargs:
            pbar.update(20) # 组装图片进度
            parts.append({"type": "image", "data": kwargs["images"]})

        try:
            # 💡 在调用 API 前，将进度条推到中段
            pbar.update(50) 
            
            res = call_universal_api(
                ai_config=ai_config,
                system_prompt=system_prompt.strip() if system_prompt.strip() else None,
                parts=parts,
                temperature=temperature,
                seed=seed
            )
            
            # 💡 任务结束，进度条拉满
            pbar.update(100)
            
            content = res.get("content", "") if isinstance(res, dict) else str(res)
            return (content, torch.zeros([1, 64, 64, 3]), torch.zeros([1, 64, 64, 3]))

        except Exception as e:
            # 出错重置进度条
            pbar.update(0)
            error_report = f"❌ Error [Source: {source_info}]: {str(e)}"
            print(error_report)
            return (error_report, torch.zeros([1, 64, 64, 3]), torch.zeros([1, 64, 64, 3]))


# ==============================
# Set / Get Global AI Config Nodes
# ==============================

class UniversalAISetConfig:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ai_config": ("AI_CONFIG",),
                "key": ("UNIVERSAL_KEY", {"default": "default"}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"}, 
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "set_config"
    CATEGORY = "Universal_AI/Utils"

    def set_config(self, ai_config, key="default", unique_id=None):
        config_to_store = ai_config.copy()
        original_source = config_to_store.get("source_node", "Unknown")
        config_to_store["source_node"] = f"{original_source} -> GlobalKey:{key}(Node_{unique_id})"
        config_to_store["_timestamp"] = time.time()
        set_global_ai_config(key.strip() or "default", config_to_store)
        print(f"💾 [Universal AI] Config saved to Key: {key}")
        return {}


class UniversalAIGetConfig:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "key": ("UNIVERSAL_KEY", {"default": "default"}),
            }
        }
        
    RETURN_TYPES = ("AI_CONFIG",)
    FUNCTION = "get_config"
    CATEGORY = "Universal_AI/Utils"

    @classmethod
    def IS_CHANGED(s, key):
        from .utils import _GLOBAL_AI_CONFIG
        config = _GLOBAL_AI_CONFIG.get(key, {})
        return config.get("_timestamp", 0)

    def get_config(self, key="default"):
        config = get_global_ai_config(key)
        if config is None:
            raise RuntimeError(f"❌ [Universal AI] Config Key '{key}' not found.")
        return (config,)