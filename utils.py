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
    keys = [k.strip() for k in api_key_str.split(",") if k.strip()]
    return random.choice(keys) if keys else ""

def sync_all_models(provider, api_key):
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
                    # 💡 改为请求 endpoints (接入点) 接口
                    url = f"https://ark.{reg}.volces.com/api/v3/endpoints"
                    headers = {"Authorization": f"Bearer {api_key}"}
                    resp = session.get(url, headers=headers, timeout=15)
                    
                    if resp.status_code == 200:
                        data = resp.json().get("data", [])
                        for ep in data:
                            ep_id = ep.get("endpoint_id")
                            if not ep_id: continue
                            
                            # 获取模型名称用于判断能力标签
                            m_info = ep.get("model", {})
                            m_name = str(m_info.get("name") or ep.get("name") or "").lower()
                            
                            # 🏷️ 标签识别逻辑
                            tag = "[CHAT]"
                            if any(kw in m_name for kw in ["vision", "vl"]): 
                                tag = "[VISION]"
                            elif any(kw in m_name for kw in ["audio", "speech", "tts", "voice"]): 
                                tag = "[AUDIO]"
                            elif any(kw in m_name for kw in ["art", "cv", "image"]): 
                                tag = "[IMAGE]"
                            
                            collected_models.append(f"{tag} {ep_id}")
                        print(f"✅ [Universal AI] Found {len(data)} endpoints in {reg}")
                except Exception as e:
                    print(f"⚠️ [Universal AI] {reg} fetch error: {e}")

        # ====================== 2. Qwen/OpenAI 标签修正 ======================
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
                    # 💡 针对 Qwen 增强识别能力
                    if provider == "Qwen":
                        if any(kw in m_id_l for kw in ["vl", "vision", "gui"]): 
                            tag = "[VISION]"
                        elif any(kw in m_id_l for kw in ["audio", "speech", "cosyvoice", "tts"]): 
                            tag = "[AUDIO]"
                        elif "coder" in m_id_l:
                            tag = "[CODE]"
                    
                    collected_models.append(f"{tag} {m_id}")

        elif provider == "Gemini":
            url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
            res = session.get(url, timeout=10).json()
            for m in res.get("models", []):
                name = m["name"].replace("models/", "")
                tag = "[VISION]" if "vision" in name.lower() else "[CHAT]"
                collected_models.append(f"{tag} {name}")

    except Exception as e:
        print(f"❌ [Universal AI] {provider} Sync Error: {e}")

    # --- 缓存逻辑 ---
    if collected_models:
        try:
            cache_data = {}
            if os.path.exists(CACHE_PATH):
                with open(CACHE_PATH, "r", encoding="utf-8") as f:
                    try: cache_data = json.load(f)
                    except: cache_data = {}
            
            unique_models = sorted(list(set(collected_models)))
            cache_data[provider] = unique_models
            
            with open(CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, indent=4, ensure_ascii=False)
            
            print(f"💾 [Universal AI] {provider} cache updated with {len(unique_models)} items.")
        except Exception as e:
            print(f"❌ [Universal AI] Cache Write Error: {e}")
    else:
        print(f"ℹ️ [Universal AI] No models collected for {provider}.")

def get_combined_models():
    # 默认展示列表
    default_models = ["[CHAT] gpt-4o", "[CHAT] gemini-1.5-flash", "[VISION] qwen-vl-max"]
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, "r", encoding="utf-8") as f:
                cache = json.load(f)
                cached_list = []
                for models in cache.values():
                    if isinstance(models, list): cached_list.extend(models)
                return sorted(list(set(default_models + cached_list)))
        except: pass
    return sorted(default_models)

# === 以下函数保持不变 ===
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
            with requests.get(url, stream=True, timeout=60) as r:
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

# ===== 全局 AI 配置存储 =====
_GLOBAL_AI_CONFIG = {}

def set_global_ai_config(key: str, config):
    _GLOBAL_AI_CONFIG[key] = config

def get_global_ai_config(key: str):
    return _GLOBAL_AI_CONFIG.get(key)