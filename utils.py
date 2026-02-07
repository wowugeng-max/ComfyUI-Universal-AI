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
from PIL import Image
import urllib3

# 屏蔽 SSL 警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

CACHE_PATH = os.path.join(os.path.dirname(__file__), "universal_model_cache.json")

def get_api_key(api_key_str):
    """支持逗号分隔的多 Key 随机轮询"""
    keys = [k.strip() for k in api_key_str.split(",") if k.strip()]
    return random.choice(keys) if keys else ""

def sync_all_models(provider, api_key):
    """
    同步模型列表并按 Provider 分类存入缓存
    """
    if not api_key: 
        print(f"ℹ️ [Universal AI] No API Key provided for {provider}, skipping sync.")
        return
    
    collected_models = [] 
    print(f"🔄 [Universal AI] Starting sync for {provider}...")

    session = requests.Session()
    session.verify = False # 解决便携版证书问题

    try:
        # ====================== 1. 豆包 (Doubao) 接入点抓取 ======================
        if provider == "Doubao":
            regions = ["cn-beijing", "cn-shanghai"]
            for reg in regions:
                try:
                    url = f"https://ark.{reg}.volces.com/api/v3/endpoints"
                    headers = {"Authorization": f"Bearer {api_key}"}
                    resp = session.get(url, headers=headers, timeout=15)
                    
                    if resp.status_code == 200:
                        res_json = resp.json()
                        data = res_json.get("items") or res_json.get("data") or []
                        for ep in data:
                            ep_id = ep.get("endpoint_id")
                            if not ep_id: continue
                            
                            m_info = ep.get("model", {})
                            m_name = str(m_info.get("name") or ep.get("name") or "").lower()
                            
                            tag = "[CHAT]"
                            if any(kw in m_name for kw in ["wan2", "video", "t2v", "v2v"]):
                                tag = "[VIDEO]"
                            elif any(kw in m_name for kw in ["vision", "vl"]): 
                                tag = "[VISION]"
                            elif any(kw in m_name for kw in ["audio", "speech", "tts", "voice", "asr"]): 
                                tag = "[AUDIO]"
                            elif any(kw in m_name for kw in ["art", "cv", "image"]): 
                                tag = "[IMAGE]"
                            
                            collected_models.append(f"{tag} {ep_id}")
                except Exception as e:
                    print(f"⚠️ [Universal AI] {reg} fetch error: {e}")

        # ====================== 2. Qwen/OpenAI/Grok 标签修正 ======================
        elif provider in ["OpenAI", "Grok", "Qwen"]:
            ep_map = {
                "OpenAI": "https://api.openai.com/v1/models",
                "Grok": "https://api.x.ai/v1/models",
                "Qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1/models",
            }
            headers = {"Authorization": f"Bearer {api_key}"}
            resp = session.get(ep_map[provider], headers=headers, timeout=10)
            
            if resp.status_code == 200:
                data = resp.json().get("data", [])
                for m in data:
                    m_id = m.get("id", "")
                    m_id_l = m_id.lower()
                    
                    tag = "[CHAT]"
                    if any(kw in m_id_l for kw in ["wan2", "video", "v2v", "t2v", "sora", "cogvideo"]):
                        tag = "[VIDEO]"
                    elif any(kw in m_id_l for kw in ["audio", "speech", "cosyvoice", "sambert", "tts", "whisper", "asr"]):
                        tag = "[AUDIO]"
                    elif any(kw in m_id_l for kw in ["vl", "vision", "gui"]): 
                        tag = "[VISION]"
                    elif any(kw in m_id_l for kw in ["dall-e", "flux", "wanx", "image-gen"]):
                        tag = "[IMAGE]"
                    elif "coder" in m_id_l:
                        tag = "[CODE]"
                    
                    collected_models.append(f"{tag} {m_id}")

        # ====================== 3. Gemini 标签修正 ======================
        elif provider == "Gemini":
            url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
            res = session.get(url, timeout=10).json()
            
            for m in res.get("models", []):
                name = m["name"].replace("models/", "")
                name_l = name.lower()
                
                tag = "[CHAT]"
                if any(kw in name_l for kw in ["image", "imagen", "drawing"]):
                    tag = "[IMAGE]"
                elif any(kw in name_l for kw in ["video", "veo"]):
                    tag = "[VIDEO]"
                elif any(kw in name_l for kw in ["audio", "speech", "tts"]):
                    tag = "[AUDIO]"
                elif any(kw in name_l for kw in ["vision", "vl", "pro"]):
                    tag = "[VISION]"
                
                collected_models.append(f"{tag} {name}")

    except Exception as e:
        print(f"❌ [Universal AI] {provider} Sync Error: {e}")

    # --- 整合后的缓存保存逻辑 (Provider 分类存储) ---
    if collected_models:
        try:
            cache_data = {}
            if os.path.exists(CACHE_PATH):
                with open(CACHE_PATH, "r", encoding="utf-8") as f:
                    try: 
                        cache_data = json.load(f)
                        if not isinstance(cache_data, dict): cache_data = {}
                    except: cache_data = {}
            
            unique_models = sorted(list(set(collected_models)))
            # 按 Provider 分类存储
            cache_data[provider] = unique_models
            
            with open(CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, indent=4, ensure_ascii=False)
            
            print(f"💾 [Universal AI] {provider} cache updated with {len(unique_models)} items.")
        except Exception as e:
            print(f"❌ [Universal AI] Cache Write Error: {e}")

def get_combined_models(provider=None):
    """
    💡 增强后的动态筛选逻辑：
    1. 为每个 Provider 提供内置保底模型列表。
    2. 如果有缓存，优先使用缓存。
    3. 如果没缓存且没适配，返回该 Provider 的默认模型，防止 UI 列表不变化。
    """
    # --- 1. 定义每个 Provider 的内置保底模型 ---
    default_map = {
        "Gemini": ["[CHAT] gemini-1.5-flash", "[CHAT] gemini-1.5-pro", "[VISION] gemini-2.0-flash-exp"],
        "OpenAI": ["[CHAT] gpt-4o", "[CHAT] gpt-4o-mini", "[IMAGE] dall-e-3"],
        "Grok": ["[CHAT] grok-2-latest", "[CHAT] grok-beta"],
        "Qwen": ["[VISION] qwen-vl-max", "[CHAT] qwen-turbo", "[CHAT] qwen-plus"],
        "Doubao": ["[CHAT] doubao-pro-32k", "[IMAGE] doubao-t2i-pro"],
        "Hailuo": ["[VIDEO] mini-max-v1"],
        "Luma": ["[VIDEO] luma-ray-v1"],
    }
    
    # 全局万能兜底（用于初始化或极端情况）
    fallback_defaults = ["[CHAT] gpt-4o", "[CHAT] gemini-1.5-flash"]

    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, "r", encoding="utf-8") as f:
                cache = json.load(f)
                if not isinstance(cache, dict): cache = {}

                # 如果指定了 Provider
                if provider:
                    cached_models = cache.get(provider, [])
                    if cached_models:
                        return sorted(cached_models)
                    # 💡 关键：如果没有缓存，返回预设的保底模型
                    return sorted(default_map.get(provider, fallback_defaults))
                
                # 未指定 Provider（初始化状态）：返回所有缓存 + 所有预设的并集
                all_models = set()
                for models in cache.values():
                    if isinstance(models, list): all_models.update(models)
                for models in default_map.values():
                    all_models.update(models)
                return sorted(list(all_models))
        except: pass
    
    # 彻底没缓存文件时
    if provider:
        return sorted(default_map.get(provider, fallback_defaults))
    
    # 彻底没文件也没传 Provider
    init_list = set()
    for models in default_map.values():
        init_list.update(models)
    return sorted(list(init_list))

# --- 图像与视频处理工具函数 ---

def tensor_to_base64(tensor, max_size=1024):
    if tensor.ndim == 4:
        tensor = tensor[0]
    img_np = (255. * tensor.cpu().numpy()).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(img_np)
    if max(img.size) > max_size:
        scale = max_size / max(img.size)
        img = img.resize((int(img.size[0] * scale), int(img.size[1] * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def base64_to_tensor(b64):
    img_data = base64.b64decode(b64)
    img = Image.open(io.BytesIO(img_data)).convert("RGB")
    return torch.from_numpy(np.array(img).astype(np.float32) / 255.0)[None,]

def url_to_video_tensor(url):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        try:
            with requests.get(url, stream=True, timeout=60, verify=False) as r:
                r.raise_for_status()
                for chunk in r.iter_content(8192):
                    if chunk: tmp.write(chunk)
            tmp_path = tmp.name
        except:
            return None
    try:
        cap = cv2.VideoCapture(tmp_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb.astype(np.float32) / 255.0)
        cap.release()
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return torch.from_numpy(np.array(frames)) if frames else None
    except Exception as e:
        print(f"Video Error: {e}")
        return None

# ====================== 
# 全局配置隔离存储机制
# ======================

_GLOBAL_AI_CONFIG = {}

def set_global_ai_config(key_or_id: str, config):
    if not key_or_id:
        return
    _GLOBAL_AI_CONFIG[key_or_id.strip()] = config
    print(f"📡 [Universal AI] Config stored under key: {key_or_id}")

def get_global_ai_config(key_or_id: str):
    config = _GLOBAL_AI_CONFIG.get(key_or_id.strip())
    if config:
        import copy
        return copy.deepcopy(config)
    return None