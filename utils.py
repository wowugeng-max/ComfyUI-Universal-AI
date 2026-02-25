import os
import json
import random
import requests
import base64
import io
import torch
import cv2
import tempfile
import numpy as np
import urllib3
import copy
import re
from PIL import Image
from enum import Enum
import logging

# 内置默认模型（硬编码，作为最后的后备和文件生成源）
_BUILTIN_DEFAULT_MODELS = {
    "Gemini": ["[CHAT] gemini-1.5-flash","[CHAT] gemini-1.5-pro","[VISION] gemini-2.0-flash-exp"],
    "OpenAI": ["[CHAT] gpt-4o","[CHAT] gpt-4o-mini","[IMAGE] dall-e-3"],
    "Grok": ["[CHAT] grok-2-latest","[CHAT] grok-beta"],
    "Qwen": ["[VISION] qwen-vl-max","[CHAT] qwen-turbo","[CHAT] qwen-plus"],
    "Doubao": ["[CHAT] doubao-pro-32k","[IMAGE] doubao-t2i-pro"],
    "Hailuo": ["[VIDEO] mini-max-v1"],
    "Luma": ["[VIDEO] luma-ray-v1"],
    "DeepSeek": ["[CHAT] deepseek-chat","[CHAT] deepseek-coder"]
}

# 配置日志（如果尚未配置）
logger = logging.getLogger(__name__)

# 默认模型配置文件路径
DEFAULT_MODELS_PATH = os.path.join(os.path.dirname(__file__), "default_models.json")


def ensure_default_models_file():
    """
    如果 default_models.json 不存在，则用内置默认模型创建。
    若文件已存在但内容损坏，可选择性修复（这里简单记录错误但不覆盖）。
    """
    if os.path.exists(DEFAULT_MODELS_PATH):
        # 可选：验证文件内容是否为有效 JSON 且为字典，若不是则记录警告
        try:
            with open(DEFAULT_MODELS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                logger.warning(f"{DEFAULT_MODELS_PATH} exists but is not a dictionary. "
                               "You may want to delete it to regenerate.")
        except Exception as e:
            logger.error(f"Failed to parse {DEFAULT_MODELS_PATH}: {e}")
        return  # 文件已存在，不覆盖

    # 文件不存在，创建
    try:
        with open(DEFAULT_MODELS_PATH, "w", encoding="utf-8") as f:
            json.dump(_BUILTIN_DEFAULT_MODELS, f, indent=4, ensure_ascii=False)
        logger.info(f"Created default models file: {DEFAULT_MODELS_PATH}")
    except Exception as e:
        logger.error(f"Failed to create default models file: {e}")

# 缓存默认模型字典，避免重复读取文件
_DEFAULT_MODELS_CACHE = None

def load_default_models():
    """加载默认模型配置文件，返回 {provider: [model_with_tag]} 字典"""
    global _DEFAULT_MODELS_CACHE
    if _DEFAULT_MODELS_CACHE is not None:
        return _DEFAULT_MODELS_CACHE

    # 确保文件存在（理论上已在模块加载时确保，但此处再次检查以防外部删除）
    ensure_default_models_file()

    try:
        with open(DEFAULT_MODELS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("default_models.json must be a dictionary")
            _DEFAULT_MODELS_CACHE = data
            return data
    except Exception as e:
        logger.error(f"Failed to load default models, using built-in defaults: {e}")
        _DEFAULT_MODELS_CACHE = _BUILTIN_DEFAULT_MODELS.copy()
        return _DEFAULT_MODELS_CACHE

# ====================== 全局配置 ======================
VERIFY_SSL = False
if not VERIFY_SSL:
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

api_session = requests.Session()
api_session.verify = VERIFY_SSL

CACHE_PATH = os.path.join(os.path.dirname(__file__), "universal_model_cache.json")
_GLOBAL_AI_CONFIG = {}

# ====================== 能力枚举 ======================
class ModelCapability(Enum):
    CHAT = "chat"
    VISION = "vision"
    IMAGE_GEN = "image_gen"
    VIDEO_GEN = "video_gen"
    AUDIO_GEN = "audio_gen"
    UNKNOWN = "unknown"

# ====================== 核心能力判断 ======================
def get_model_capability(model_name: str, provider: str = "") -> ModelCapability:
    """
    根据模型名称和提供商判断模型能力（优先级：视频 > 图像 > 音频 > 视觉 > 对话）
    """
    if not isinstance(model_name, str):
        return ModelCapability.UNKNOWN

    name_lower = model_name.lower()

    # 视频生成
    if any(kw in name_lower for kw in ["video", "sora", "cogvideo", "veo", "wan2", "t2v"]):
        return ModelCapability.VIDEO_GEN

    # 音频生成
    if any(kw in name_lower for kw in ["audio", "speech", "tts", "whisper", "cosyvoice"]):
        return ModelCapability.AUDIO_GEN

    # 图像生成（排除视觉理解模型）
    img_kws = ["image", "imagen", "wanx", "dall-e", "flux", "paint", "draw", "art", "gen", "style"]
    if any(kw in name_lower for kw in img_kws):
        # 如果同时包含视觉关键词（vl/vision），优先视为视觉理解
        if not any(vision_kw in name_lower for vision_kw in ["vl", "vision", "visual"]):
            return ModelCapability.IMAGE_GEN

    # 视觉理解（图文输入）
    if any(kw in name_lower for kw in ["vision", "vl", "visual"]):
        return ModelCapability.VISION

    # 默认对话
    return ModelCapability.CHAT

# ====================== 标签生成 ======================
def get_model_tag(model_name: str, provider: str = "") -> str:
    """返回带 UI 标签的模型名称，如 '[CHAT] qwen-max'"""
    capability = get_model_capability(model_name, provider)
    tag_map = {
        ModelCapability.CHAT: "[CHAT]",
        ModelCapability.VISION: "[VISION]",
        ModelCapability.IMAGE_GEN: "[IMAGE]",
        ModelCapability.VIDEO_GEN: "[VIDEO]",
        ModelCapability.AUDIO_GEN: "[AUDIO]",
        ModelCapability.UNKNOWN: "[UNKNOWN]",
    }
    prefix = tag_map.get(capability, "[UNKNOWN]")
    return f"{prefix} {model_name}"

def strip_model_label(model_name: str) -> str:
    """移除模型名称开头的标签前缀，例如 '[CHAT] '"""
    if not isinstance(model_name, str):
        return ""
    return re.sub(r'^\[\w+\]\s*', '', model_name)

# ====================== 原有工具函数保持不变 ======================
def parse_extra_params(extra_str):
    """解析 extra_params JSON 字符串"""
    try:
        return json.loads(extra_str) if extra_str.strip() else {}
    except:
        return {}

def get_api_key(api_key_str):
    """支持逗号分隔的多 Key 随机轮询"""
    if not api_key_str:
        return ""
    keys = [k.strip() for k in api_key_str.split(",") if k.strip()]
    return random.choice(keys) if keys else ""

def extract_all_text(parts):
    """从 parts 中提取所有文本"""
    texts = [p["data"] for p in parts if p["type"] == "text"]
    return "\n\n".join(texts)

def extract_all_images(parts):
    """从 parts 中提取所有图片数据（Base64）"""
    return [p["data"] for p in parts if p["type"] == "image"]

def safe_process_image(img_data):
    """安全处理图片数据，补全 Data URI 前缀"""
    if not isinstance(img_data, str):
        print(f"⚠️ [Universal AI] Warning: Expected Base64 string, but got {type(img_data)}.")
        return None
    clean_data = img_data.replace("\n", "").replace("\r", "").strip()
    if clean_data.startswith("data:image"):
        return clean_data
    return f"data:image/jpeg;base64,{clean_data}"

# utils.py（修改后的 sync_all_models 函数）
def sync_all_models(provider, api_key):
    """刷新模型列表，使用同步器工厂解耦"""
    if not api_key:
        return

    from .syncers.modelSyncerFactory import ModelSyncerFactory

    syncer = ModelSyncerFactory.get_syncer(provider, api_key)
    if not syncer:
        print(f"⚠️ [Universal AI] No syncer found for provider: {provider}")
        return

    collected_models = syncer.sync()
    if not collected_models:
        print(f"⚠️ [Universal AI] No models synced for {provider}")
        return

    # 缓存写入（与原逻辑相同）
    try:
        cache_data = {}
        if os.path.exists(CACHE_PATH):
            with open(CACHE_PATH, "r", encoding="utf-8") as f:
                try:
                    cache_data = json.load(f)
                except:
                    cache_data = {}
        unique_models = sorted(list(set(collected_models)))
        cache_data[provider] = unique_models
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, indent=4, ensure_ascii=False)
        print(f"💾 [Universal AI] {provider} cache updated with {len(unique_models)} items.")
    except Exception as e:
        print(f"❌ [Universal AI] Cache Write Error: {e}")

def get_combined_models(provider=None):
    """
    获取合并的模型列表（缓存 + 默认模型）
    如果 provider 为 None，则返回所有提供商的模型（用于下拉框全量展示）
    """
    default_models = load_default_models()
    # ... 其余逻辑不变，但后备可以设为空列表或省略
    # 如果没有任何模型，可返回 [] 或一个通用后备

    # 读取缓存
    cache = {}
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, "r", encoding="utf-8") as f:
                cache = json.load(f)
                if not isinstance(cache, dict):
                    cache = {}
        except Exception:
            cache = {}

    if provider:
        # 单个提供商：优先使用缓存，若无则使用默认，若无默认则用后备
        provider_models = cache.get(provider, [])
        if not provider_models:
            provider_models = default_models.get(provider, fallback_defaults)
        return sorted(set(provider_models))
    else:
        # 全量模型：合并所有缓存 + 所有默认
        all_models = set()
        for models in cache.values():
            all_models.update(models)
        for models in default_models.values():
            all_models.update(models)
        # 如果为空，至少返回后备列表
        if not all_models:
            return fallback_defaults
        return sorted(all_models)

def tensor_to_base64(tensor, max_size=1024, auto_resize=True):
    """将 ComfyUI Tensor 转换为 Base64 字符串，支持自动缩放"""
    if tensor.ndim == 4:
        tensor = tensor[0]
    img_np = (255. * tensor.cpu().numpy()).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(img_np)
    if auto_resize and max(img.size) > max_size:
        scale = max_size / max(img.size)
        img = img.resize((int(img.size[0] * scale), int(img.size[1] * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def base64_to_tensor(b64):
    """将 Base64 图片转换为 ComfyUI Tensor"""
    img_data = base64.decodebytes(b64.encode('utf-8'))
    img = Image.open(io.BytesIO(img_data)).convert("RGB")
    return torch.from_numpy(np.array(img).astype(np.float32) / 255.0)[None,]

def url_to_video_tensor(url):
    """从视频 URL 下载并解码为帧张量"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        try:
            with requests.get(url, stream=True, timeout=60, verify=False) as r:
                r.raise_for_status()
                for chunk in r.iter_content(8192):
                    if chunk:
                        tmp.write(chunk)
            tmp_path = tmp.name
        except:
            return None
    try:
        cap = cv2.VideoCapture(tmp_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
        cap.release()
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return torch.from_numpy(np.array(frames)) if frames else None
    except:
        return None

def set_global_ai_config(key: str, config):
    global _GLOBAL_AI_CONFIG
    if not key:
        return
    clean_key = key.strip()
    _GLOBAL_AI_CONFIG[clean_key] = config
    print(f"📡 [Universal AI] Config stored under key: {clean_key}")

def get_global_ai_config(key: str):
    global _GLOBAL_AI_CONFIG
    config = _GLOBAL_AI_CONFIG.get(key.strip())
    return copy.deepcopy(config) if config else None

def get_all_active_config_keys():
    global _GLOBAL_AI_CONFIG
    return list(_GLOBAL_AI_CONFIG.keys())



#加载默认模型列表，必须
ensure_default_models_file()