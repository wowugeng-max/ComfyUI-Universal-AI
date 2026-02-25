import torch
import numpy as np
import comfy.utils
from .utils import *
from .api_adapters import call_universal_api
from server import PromptServer
from aiohttp import web
import time

# ==============================
# 后端 API 路由
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
                "provider": (["Gemini", "OpenAI", "Grok", "Qwen", "Doubao", "Hailuo", "Luma","DeepSeek"], {"default": "Gemini"}),
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
        print(f"🔍 [DEBUG] get key, key={active_key}")
        if refresh_list and active_key:
            print(f"🔍 [DEBUG] load refresh model, key={active_key}")
            sync_all_models(provider, active_key)   # 刷新模型缓存

            # 清洗模型名称（去除UI标签）
            raw_model_name = kwargs.get("custom_model_name") or model_selection
            model_name = strip_model_label(raw_model_name)  # 使用 utils 中的函数

        return ({
            "provider": provider,
            "api_key": active_key,
            "model_name": model_name,
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
                "auto_resize": ("BOOLEAN", {"default": True}),
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

    def execute(self, ai_config, system_prompt, user_prompt, auto_resize, max_image_size, temperature, seed,
                text="", images=None, video=None, max_video_frames=10):
        source_info = ai_config.get("source_node", "Unknown_Source")
        provider = ai_config.get("provider")
        model = ai_config.get("model_name")
        print(f"🕵️ [Universal AI] Runner Starting... (Source: {source_info})")
        print(f"   - Config: {provider} / {model}")

        pbar = comfy.utils.ProgressBar(100)
        pbar.update_absolute(10)   # 初始化进度

        # 组装多模态输入
        parts = []
        combined_text = (text + "\n" + user_prompt).strip()
        if combined_text:
            parts.append({"type": "text", "data": combined_text})

        if images is not None:
            batch_size = images.shape[0]
            print(f"📸 [Universal AI] Processing {batch_size} image(s) in batch...")
            for i in range(batch_size):
                single_img = images[i:i+1]
                b64 = tensor_to_base64(single_img, auto_resize=auto_resize, max_size=max_image_size)
                if b64:
                    parts.append({"type": "image", "data": b64})
                # 更新进度：20% ~ 45%
                progress = 20 + int((i + 1) / batch_size * 25)
                pbar.update_absolute(progress)

        if video is not None:
            num_frames = video.shape[0]
            indices = np.linspace(0, num_frames - 1, min(num_frames, max_video_frames), dtype=int)
            for idx in indices:
                frame = video[idx:idx+1]
                b64 = tensor_to_base64(frame, auto_resize=auto_resize, max_size=max_image_size)
                if b64:
                    parts.append({"type": "image", "data": b64})
            # 视频帧进度合并到图片处理中（这里简单处理）

        if not parts:
            raise ValueError(f"No input provided. (Check {source_info})")

        pbar.update_absolute(50)   # 进入 API 调用阶段

        try:
            res = call_universal_api(
                ai_config=ai_config,
                system_prompt=system_prompt.strip() if system_prompt.strip() else None,
                parts=parts,
                temperature=temperature,
                seed=seed
            )
            pbar.update_absolute(85)

            empty_img = torch.zeros([1, 64, 64, 3])
            video_tensor = empty_img

            if res["type"] == "image":
                # 返回生成的图像
                generated_image = base64_to_tensor(res["content"])
                return ("Image generated successfully.", generated_image, empty_img)

            # 文本类型
            text_content = res["content"]

            # 检查文本中是否包含视频链接
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
            pbar.update_absolute(0)
            error_report = f"❌ Error [Source: {source_info}]: {str(e)}"
            print(error_report)
            import traceback
            traceback.print_exc()
            return (error_report, torch.zeros([1, 64, 64, 3]), torch.zeros([1, 64, 64, 3]))


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

    RETURN_TYPES = ("AI_CONFIG",)          # 增加输出
    RETURN_NAMES = ("ai_config",)           # 可选：命名输出
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
        return (ai_config,)                  # 返回传入的配置（原样）


class UniversalAIGetConfig:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "key": ("UNIVERSAL_KEY", {"default": "default"}),
                # 可选：增加一个开关，控制是否允许缺失时返回空配置
                "allow_missing": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("AI_CONFIG",)
    FUNCTION = "get_config"
    CATEGORY = "Universal_AI/Utils"

    @classmethod
    def IS_CHANGED(s, key, allow_missing=True):
        from .utils import _GLOBAL_AI_CONFIG
        config = _GLOBAL_AI_CONFIG.get(key, {})
        return config.get("_timestamp", 0)

    def get_config(self, key="default", allow_missing=True):
        config = get_global_ai_config(key)
        if config is None:
            if allow_missing:
                print(f"⚠️ [Universal AI] Config Key '{key}' not found. Returning empty config.")
                # 返回一个占位配置（可根据需要填充默认值）
                return ({"provider": "unknown", "model_name": "", "api_key": ""},)
            else:
                raise RuntimeError(f"❌ [Universal AI] Config Key '{key}' not found.单独运行Loader节点刷新模型")
        return (config,)