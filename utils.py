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

CACHE_PATH = os.path.join(os.path.dirname(__file__), "universal_model_cache.json")

def get_api_key(api_key_str):
    keys = [k.strip() for k in api_key_str.split(",") if k.strip()]
    return random.choice(keys) if keys else ""

def sync_all_models(provider, api_key):
    if not api_key: return
    new_models = []
    try:
        if provider == "Gemini":
            url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
            res = requests.get(url, timeout=10).json()
            new_models = [m["name"].replace("models/", "") for m in res.get("models", [])]
        elif provider in ["OpenAI", "Grok", "Qwen", "Doubao"]:
            endpoints = {
                "OpenAI": "https://api.openai.com/v1/models",
                "Grok": "https://api.x.ai/v1/models",
                "Qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1/models",
                "Doubao": "https://ark.cn-beijing.volces.com/api/v3/models"
            }
            headers = {"Authorization": f"Bearer {api_key}"}
            res = requests.get(endpoints[provider], headers=headers, timeout=10).json()
            new_models = [m["id"] for m in res.get("data", [])]
        
        if new_models:
            cache = {}
            if os.path.exists(CACHE_PATH):
                with open(CACHE_PATH, "r") as f: cache = json.load(f)
            cache[provider] = new_models
            with open(CACHE_PATH, "w") as f: json.dump(cache, f)
    except Exception as e:
        print(f"⚠️ [Universal AI] Sync models failed: {e}")

def get_combined_models():
    default_models = ["gemini-1.5-flash", "gpt-4o", "grok-2-vision-1212", "qwen-vl-max"]
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, "r") as f:
                cache = json.load(f)
                return list(set(default_models + [m for sublist in cache.values() for m in sublist]))
        except: pass
    return default_models

def tensor_to_base64(tensor, max_size=1024):
    """优化：支持 4D Tensor，并确保在 CPU 处理"""
    if tensor.ndim == 4:
        tensor = tensor[0] # 取 Batch 中的第一张
    
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
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            for chunk in r.iter_content(8192):
                if chunk: tmp.write(chunk)
        tmp_path = tmp.name

    try:
        cap = cv2.VideoCapture(tmp_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb.astype(np.float32) / 255.0)
        cap.release()
        os.remove(tmp_path)
        return torch.from_numpy(np.array(frames)) if frames else None
    except Exception as e:
        print(f"Video Error: {e}")
        return None



# ===== 全局 AI 配置存储（用于 Set/Get 节点）=====
_GLOBAL_AI_CONFIG = {}

def set_global_ai_config(key: str, config):
    _GLOBAL_AI_CONFIG[key] = config

def get_global_ai_config(key: str):
    return _GLOBAL_AI_CONFIG.get(key)