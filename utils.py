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

def sync_all_models(provider, api_key):
    """刷新模型列表（原函数不变，但内部调用 get_model_tag 已更新）"""
    if not api_key:
        return
    collected_models = []
    session = requests.Session()
    session.verify = False
    try:
        # --- 豆包 ---
        if provider == "Doubao":
            for reg in ["cn-beijing", "cn-shanghai"]:
                url = f"https://ark.{reg}.volces.com/api/v3/endpoints"
                resp = session.get(url, headers={"Authorization": f"Bearer {api_key}"}, timeout=15)
                if resp.status_code == 200:
                    for ep in resp.json().get("items", []):
                        m_name = str(ep.get("model", {}).get("name", "")).lower()
                        tag = get_model_tag(m_name, provider)   # 使用新函数
                        collected_models.append(f"{tag} {ep['endpoint_id']}")
        # --- OpenAI/Grok/Qwen ---
        elif provider in ["OpenAI", "Grok", "Qwen"]:
            ep_map = {"OpenAI": "https://api.openai.com/v1/models", "Grok": "https://api.x.ai/v1/models", "Qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1/models"}
            resp = session.get(ep_map[provider], headers={"Authorization": f"Bearer {api_key}"}, timeout=10)
            if resp.status_code == 200:
                for m in resp.json().get("data", []):
                    collected_models.append(get_model_tag(m['id'], provider))

                    # --- Gemini ---
        elif provider == "Gemini":
            url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
            resp = session.get(url, timeout=10)
            if resp.status_code == 200:
                for m in resp.json().get("models", []):
                    name = m["name"].replace("models/", "")
                    collected_models.append(get_model_tag(m['id'], provider))
    except Exception as e:
        print(f"❌ [Universal AI] {provider} Sync Error: {e}")

    if collected_models:
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
    """获取合并的模型列表（缓存 + 默认）"""
    default_map = {
        "Gemini": ["[CHAT] gemini-1.5-flash", "[CHAT] gemini-1.5-pro", "[VISION] gemini-2.0-flash-exp"],
        "OpenAI": ["[CHAT] gpt-4o", "[CHAT] gpt-4o-mini", "[IMAGE] dall-e-3"],
        "Grok": ["[CHAT] grok-2-latest", "[CHAT] grok-beta"],
        "Qwen": ["[VISION] qwen-vl-max", "[CHAT] qwen-turbo", "[CHAT] qwen-plus"],
        "Doubao": ["[CHAT] doubao-pro-32k", "[IMAGE] doubao-t2i-pro"],
        "Hailuo": ["[VIDEO] mini-max-v1"],
        "Luma": ["[VIDEO] luma-ray-v1"],
    }
    fallback_defaults = ["[CHAT] gpt-4o", "[CHAT] gemini-1.5-flash"]

    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, "r", encoding="utf-8") as f:
                cache = json.load(f)
                if not isinstance(cache, dict):
                    cache = {}
                if provider:
                    return sorted(cache.get(provider, default_map.get(provider, fallback_defaults)))
                # 未指定 provider：合并缓存 + 所有默认模型
                all_models = set()
                for models in cache.values():
                    all_models.update(models)
                for models in default_map.values():
                    all_models.update(models)
                return sorted(all_models)
        except Exception:
            pass

    if provider:
        return sorted(default_map.get(provider, fallback_defaults))
    all_defaults = set()
    for models in default_map.values():
        all_defaults.update(models)
    return sorted(all_defaults)

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