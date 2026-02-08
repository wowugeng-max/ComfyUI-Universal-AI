import os
import json
import base64
import requests
import time
import urllib3
from typing import List, Dict, Any, Optional

# ======================
# 全局配置与安全设置
# ======================
# 💡 统一控制：False 表示跳过证书验证，解决代理/VPN 环境下的报错
VERIFY_SSL = False 

if not VERIFY_SSL:
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 💡 统一 Session：提高连接效率并共享 SSL 配置
api_session = requests.Session()
api_session.verify = VERIFY_SSL

# ======================
# 工具函数
# ======================
def parse_extra_params(extra_str):
    try:
        return json.loads(extra_str) if extra_str.strip() else {}
    except:
        return {}

def get_api_key(provided_key):
    if provided_key and provided_key.strip():
        return provided_key.strip()
    env_key = os.getenv("UNIVERSAL_AI_API_KEY")
    if env_key:
        return env_key
    raise ValueError("API key is required but not provided.")

def extract_all_text(parts: List[Dict]) -> str:
    texts = [p["data"] for p in parts if p["type"] == "text"]
    return "\n\n".join(texts)

def extract_all_images(parts: List[Dict]) -> List[str]:
    return [p["data"] for p in parts if p["type"] == "image"]

def safe_process_image(img_data: Any) -> Optional[str]:
    """
    💡 核心修复函数：
    1. 检查类型，防止 Tensor 对象调用 .replace() 报错
    2. 清理 Base64 干扰字符
    3. 补全 Data URI 前缀
    """
    if not isinstance(img_data, str):
        print(f"⚠️ [Universal AI] Warning: Expected Base64 string, but got {type(img_data)}. Please check node connection.")
        return None
    
    # 清理换行符和空格
    clean_data = img_data.replace("\n", "").replace("\r", "").strip()
    
    # 如果已经是完整 Data URI 则直接返回，否则补全
    if clean_data.startswith("data:image"):
        return clean_data
    return f"data:image/jpeg;base64,{clean_data}"

# ======================
# 各平台调用实现
# ======================

def gemini_call(model_name, api_key, system_prompt, parts, temperature, extra_params):
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name,
        system_instruction=system_prompt,
        safety_settings={
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        }
    )

    gemini_parts = []
    for part in parts:
        if part["type"] == "text":
            gemini_parts.append(part["data"])
        elif part["type"] == "image":
            # Gemini SDK 内部处理 bytes，这里逻辑保持
            if isinstance(part["data"], str):
                img_bytes = base64.b64decode(part["data"])
                gemini_parts.append({"mime_type": "image/jpeg", "data": img_bytes})

    response = model.generate_content(
        gemini_parts,
        generation_config=genai.types.GenerationConfig(
            temperature=temperature,
            **extra_params
        )
    )

    if response.candidates and response.candidates[0].content.parts:
        first_part = response.candidates[0].content.parts[0]
        if hasattr(first_part, 'mime_type') and first_part.mime_type and "image" in first_part.mime_type:
            img_b64 = base64.b64encode(first_part.data).decode()
            return {"type": "image", "content": img_b64}
        else:
            text = first_part.text if hasattr(first_part, 'text') else str(first_part)
            return {"type": "text", "content": text}
    return {"type": "text", "content": ""}


def openai_call(model_name, api_key, system_prompt, parts, temperature, extra_params):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    user_content = []
    for part in parts:
        if part["type"] == "text":
            user_content.append({"type": "text", "text": part["data"]})
        elif part["type"] == "image":
            processed_url = safe_process_image(part['data'])
            if processed_url:
                user_content.append({"type": "image_url", "image_url": {"url": processed_url}})

    messages.append({"role": "user", "content": user_content})
    payload = {"model": model_name, "messages": messages, "temperature": temperature, **extra_params}

    resp = api_session.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers, timeout=120)
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    return {"type": "text", "content": content}


def qwen_call(model_name, api_key, system_prompt, parts, temperature, extra_params):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    user_content = []
    for part in parts:
        if part["type"] == "text":
            user_content.append({"type": "text", "text": part["data"]})
        elif part["type"] == "image":
            processed_url = safe_process_image(part['data'])
            if processed_url:
                user_content.append({"type": "image_url", "image_url": {"url": processed_url}})
    
    messages.append({"role": "user", "content": user_content})
    payload = {"model": model_name, "messages": messages, "temperature": temperature, **extra_params}
    
    url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    resp = api_session.post(url, json=payload, headers=headers, timeout=120)
    
    if resp.status_code != 200:
        raise RuntimeError(f"Qwen API Error: {resp.text}")
        
    result = resp.json()
    message = result["choices"][0]["message"]
    
    if "audio" in message:
        return {"type": "audio", "content": message["audio"].get("data")}
    
    return {"type": "text", "content": message.get("content", "")}


def doubao_call(model_name, api_key, system_prompt, parts, temperature, extra_params):
    url = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    user_content = []
    for part in parts:
        if part["type"] == "text":
            user_content.append({"type": "text", "text": part["data"]})
        elif part["type"] == "image":
            processed_url = safe_process_image(part['data'])
            if processed_url:
                user_content.append({"type": "image_url", "image_url": {"url": processed_url}})

    messages.append({"role": "user", "content": user_content})
    payload = {"model": model_name, "messages": messages, "temperature": temperature, **extra_params}

    resp = api_session.post(url, json=payload, headers=headers, timeout=120)
    resp.raise_for_status()
    
    result = resp.json()
    message = result["choices"][0]["message"]
    
    if "audio" in message:
        return {"type": "audio", "content": message["audio"].get("data")}
    return {"type": "text", "content": message.get("content", "")}


def doubao_image_gen_call(model_name, api_key, prompt, extra_params):
    submit_url = "https://ark.cn-beijing.volces.com/api/v3/cv/generation_task"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    
    payload = {"model": model_name, "prompt": prompt, **extra_params}
    resp = api_session.post(submit_url, json=payload, headers=headers, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"Doubao Image Gen Submit Failed: {resp.text}")
    
    task_id = resp.json().get("task_id")
    check_url = f"https://ark.cn-beijing.volces.com/api/v3/cv/get_task_result?task_id={task_id}"
    
    start_time = time.time()
    while time.time() - start_time < 180:
        time.sleep(3)
        try:
            r_resp = api_session.get(check_url, headers=headers, timeout=10)
            if r_resp.status_code != 200: continue
            r = r_resp.json()
            if r.get("status") == "success":
                data_list = r.get("data", [])
                if not data_list: continue
                img_item = data_list[0]
                if "b64_json" in img_item:
                    return {"type": "image", "content": img_item["b64_json"]}
                elif "url" in img_item:
                    img_resp = api_session.get(img_item["url"])
                    return {"type": "image", "content": base64.b64encode(img_resp.content).decode()}
            elif r.get("status") == "failed":
                raise RuntimeError(f"Doubao Gen Failed: {r.get('reason')}")
        except Exception as e:
            if "failed" in str(e): raise e
            continue
    raise RuntimeError("Doubao Image Generation Timeout")


def grok_call(model_name, api_key, system_prompt, parts, temperature, extra_params):
    all_text = extract_all_text(parts)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": all_text})

    payload = {"model": model_name, "messages": messages, "temperature": temperature, **extra_params}
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    resp = api_session.post("https://api.x.ai/v1/chat/completions", json=payload, headers=headers, timeout=120)
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    return {"type": "text", "content": content}


def hailuo_call(model_name, api_key, system_prompt, parts, temperature, extra_params):
    all_text = extract_all_text(parts)
    images = extract_all_images(parts)
    # 💡 同样处理海螺的图片数据
    processed_images = [safe_process_image(img) for img in images if isinstance(img, str)]
    
    payload = {"model": model_name, "prompt": all_text, "temperature": temperature, **extra_params}
    if processed_images:
        payload["image"] = processed_images[0]

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    resp = api_session.post("https://api.hailuoai.com/v1/multimodal", json=payload, headers=headers, timeout=120)
    resp.raise_for_status()
    result = resp.json()
    content = result.get("response", "")
    return {"type": "text", "content": content}


def luma_call(model_name, api_key, system_prompt, parts, temperature, extra_params):
    all_text = extract_all_text(parts)
    return {"type": "text", "content": f"[Luma] Received prompt: {all_text[:100]}..."}


# ======================
# 统一入口
# ======================
def call_universal_api(ai_config: Dict[str, Any], system_prompt: Optional[str], parts: List[Dict[str, Any]], temperature: float = 0.7, seed: int = 0) -> Dict[str, Any]:
    provider = ai_config.get("provider")
    raw_model_name = ai_config.get("model_name", "")
    api_key = ai_config.get("api_key")
    extra_params = parse_extra_params(ai_config.get("extra_params"))

    model_name = raw_model_name.split(" ")[-1] if " " in raw_model_name else raw_model_name
    print(f"🚀 [Universal AI] Routing -> Provider: {provider}, Model: {model_name}")

    try:
        if provider == "Gemini":
            return gemini_call(model_name, api_key, system_prompt, parts, temperature, extra_params)
        elif provider == "OpenAI":
            return openai_call(model_name, api_key, system_prompt, parts, temperature, extra_params)
        elif provider == "Qwen":
            return qwen_call(model_name, api_key, system_prompt, parts, temperature, extra_params)
        elif provider == "Doubao":
            if "[IMAGE]" in raw_model_name:
                prompt = extract_all_text(parts)
                return doubao_image_gen_call(model_name, api_key, prompt, extra_params)
            else:
                return doubao_call(model_name, api_key, system_prompt, parts, temperature, extra_params)
        elif provider == "Grok":
            return grok_call(model_name, api_key, system_prompt, parts, temperature, extra_params)
        elif provider == "Hailuo":
            return hailuo_call(model_name, api_key, system_prompt, parts, temperature, extra_params)
        elif provider == "Luma":
            return luma_call(model_name, api_key, system_prompt, parts, temperature, extra_params)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    except Exception as e:
        raise RuntimeError(f"API call failed for {provider} (Model: {model_name}): {str(e)}")