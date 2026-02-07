import os, json, random, requests, base64, io, torch, cv2, tempfile, numpy as np, urllib3, copy
from PIL import Image
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

CACHE_PATH = os.path.join(os.path.dirname(__file__), "universal_model_cache.json")
_GLOBAL_AI_CONFIG = {}

def get_api_key(api_key_str):
    """支持逗号分隔的多 Key 随机轮询"""
    if not api_key_str: return ""
    keys = [k.strip() for k in api_key_str.split(",") if k.strip()]
    return random.choice(keys) if keys else ""

def get_model_tag(m_id_l, provider):
    """💡 高优先级分类引擎：IMAGE > VIDEO > AUDIO > VISION > CHAT"""
    # 1. 视频
    if any(kw in m_id_l for kw in ["video", "sora", "cogvideo", "veo", "wan2", "t2v"]): return "[VIDEO]"
    # 2. 语音
    if any(kw in m_id_l for kw in ["audio", "speech", "tts", "whisper", "cosyvoice"]): return "[AUDIO]"
    # 3. 图像 (针对 Qwen/Wanx 和 Gemini/Imagen 强化)
    img_kws = ["image", "imagen", "wanx", "dall-e", "flux", "paint", "draw", "art", "gen", "style", "cosplay", "background"]
    if any(kw in m_id_l for kw in img_kws):
        if any(kw in m_id_l for kw in ["-vl", "vision-preview", "chat"]): return "[CHAT]" # 排除对话模型
        return "[IMAGE]"
    # 4. 视觉理解
    if any(kw in m_id_l for kw in ["vision", "vl", "pro"]): return "[VISION]"
    return "[CHAT]"

def sync_all_models(provider, api_key):
    if not api_key: return
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
                        tag = get_model_tag(m_name + ep["endpoint_id"].lower(), provider)
                        collected_models.append(f"{tag} {ep['endpoint_id']}")
        # --- OpenAI/Grok/Qwen ---
        elif provider in ["OpenAI", "Grok", "Qwen"]:
            ep_map = {"OpenAI": "https://api.openai.com/v1/models", "Grok": "https://api.x.ai/v1/models", "Qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1/models"}
            resp = session.get(ep_map[provider], headers={"Authorization": f"Bearer {api_key}"}, timeout=10)
            if resp.status_code == 200:
                for m in resp.json().get("data", []):
                    collected_models.append(f"{get_model_tag(m['id'].lower(), provider)} {m['id']}")
        # --- Gemini ---
        elif provider == "Gemini":
            url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
            resp = session.get(url, timeout=10)
            if resp.status_code == 200:
                for m in resp.json().get("models", []):
                    name = m["name"].replace("models/", "")
                    tag = get_model_tag(name.lower() + m.get("description", "").lower(), provider)
                    collected_models.append(f"{tag} {name}")
    except Exception as e: print(f"❌ [Universal AI] {provider} Sync Error: {e}")
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
        except Exception as e: print(f"❌ [Universal AI] Cache Write Error: {e}")

# ... (get_combined_models, tensor_to_base64 等保持不变) ...

def get_combined_models(provider=None):
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
                if provider:
                    return sorted(cache.get(provider, default_map.get(provider, fallback_defaults)))
                all_models = set()
                for models in cache.values(): all_models.update(models)
                for models in default_map.values(): all_models.update(models)
                return sorted(list(all_models))
        except: pass
    return sorted(default_map.get(provider, fallback_defaults)) if provider else sorted(fallback_defaults)

def tensor_to_base64(tensor, max_size=1024):
    if tensor.ndim == 4: tensor = tensor[0]
    img_np = (255. * tensor.cpu().numpy()).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(img_np)
    if max(img.size) > max_size:
        scale = max_size / max(img.size)
        img = img.resize((int(img.size[0] * scale), int(img.size[1] * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def base64_to_tensor(b64):
    img_data = base64.decodebytes(b64.encode('utf-8'))
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
        except: return None
    try:
        cap = cv2.VideoCapture(tmp_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
        cap.release()
        if os.path.exists(tmp_path): os.remove(tmp_path)
        return torch.from_numpy(np.array(frames)) if frames else None
    except: return None

# ==========================================
# 💡 核心改动：增强的全局配置存取机制
# ==========================================

_GLOBAL_AI_CONFIG = {}

def set_global_ai_config(key: str, config):
    """存入全局配置并自动清理空格"""
    global _GLOBAL_AI_CONFIG
    if not key: return
    clean_key = key.strip()
    _GLOBAL_AI_CONFIG[clean_key] = config
    print(f"📡 [Universal AI] Config stored under key: {clean_key}")

def get_global_ai_config(key: str):
    """读取全局配置的副本，防止多节点串扰"""
    global _GLOBAL_AI_CONFIG
    config = _GLOBAL_AI_CONFIG.get(key.strip())
    return copy.deepcopy(config) if config else None

def get_all_active_config_keys():
    """💡 供后端 API 路由拉取，返回所有存在的 Key 列表"""
    global _GLOBAL_AI_CONFIG
    return list(_GLOBAL_AI_CONFIG.keys())