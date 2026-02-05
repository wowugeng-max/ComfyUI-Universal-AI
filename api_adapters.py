import os
import json
import base64
import requests
import time
from typing import List, Dict, Any, Optional

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

# ======================
# 多模态内容处理工具
# ======================
def extract_all_text(parts: List[Dict]) -> str:
    """提取所有文本部分，用于不支持多文本的平台"""
    texts = [p["data"] for p in parts if p["type"] == "text"]
    return "\n\n".join(texts)

def extract_all_images(parts: List[Dict]) -> List[str]:
    """提取所有图像 base64"""
    return [p["data"] for p in parts if p["type"] == "image"]

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
            img_bytes = base64.b64decode(part["data"])
            gemini_parts.append({
                "mime_type": "image/jpeg",
                "data": img_bytes
            })

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
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{part['data']}"}
            })

    messages.append({"role": "user", "content": user_content})

    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        **extra_params
    }

    resp = requests.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers, timeout=120)
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    return {"type": "text", "content": content}


def qwen_call(model_name, api_key, system_prompt, parts, temperature, extra_params):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # 构建 OpenAI 风格的 messages
    messages = []
    
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    user_content = []
    for part in parts:
        if part["type"] == "text":
            user_content.append({"type": "text", "text": part["data"]})
        elif part["type"] == "image":
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{part['data']}"}
            })
    
    messages.append({"role": "user", "content": user_content})
    
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        **extra_params
    }
    
    # 使用兼容 OpenAI 的 endpoint
    url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    
    resp = requests.post(url, json=payload, headers=headers, timeout=120)
    resp.raise_for_status()
    
    content = resp.json()["choices"][0]["message"]["content"]
    return {"type": "text", "content": content}
def doubao_call(model_config, system_prompt, parts, temperature, extra_params):
    # Doubao 使用字节跳动 API，需从 model_config 提取 app_id 等
    app_id = model_config.get("model_name", "").split("/")[0] if "/" in model_config.get("model_name", "") else "default"
    api_key = model_config["api_key"]
    model_name = model_config["model_name"]

    all_text = extract_all_text(parts)
    images = extract_all_images(parts)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    content = all_text
    # Doubao 当前 API 不支持多图像，只取第一张（或忽略）
    if images:
        # 注：Doubao 多模态 API 尚未公开，此处按文本 fallback
        pass

    messages.append({"role": "user", "content": content})

    payload = {
        "app_id": app_id,
        "messages": messages,
        "temperature": temperature,
        **extra_params
    }

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    resp = requests.post("https://ark.cn-beijing.volces.com/api/v3/chat/completions", json=payload, headers=headers, timeout=120)
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    return {"type": "text", "content": content}


def grok_call(model_name, api_key, system_prompt, parts, temperature, extra_params):
    all_text = extract_all_text(parts)
    # Grok 目前不支持图像输入（截至 2025），仅文本
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": all_text})

    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        **extra_params
    }

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    resp = requests.post("https://api.x.ai/v1/chat/completions", json=payload, headers=headers, timeout=120)
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    return {"type": "text", "content": content}


def hailuo_call(model_name, api_key, system_prompt, parts, temperature, extra_params):
    # Hailuo AI（海螺）多模态 API
    all_text = extract_all_text(parts)
    images = extract_all_images(parts)

    payload = {
        "model": model_name,
        "prompt": all_text,
        "temperature": temperature,
        **extra_params
    }

    # 如果有图像，添加（假设其 API 支持 base64 图像字段）
    if images:
        payload["image"] = images[0]  # 只传第一张

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    resp = requests.post("https://api.hailuoai.com/v1/multimodal", json=payload, headers=headers, timeout=120)
    resp.raise_for_status()
    result = resp.json()
    content = result.get("response", "")
    return {"type": "text", "content": content}


def luma_call(model_name, api_key, system_prompt, parts, temperature, extra_params):
    # Luma 主要用于视频生成，此处作为文本+图像理解 fallback
    all_text = extract_all_text(parts)
    # Luma 的 Dream Machine 不支持文本理解，此处模拟
    # 实际使用中可跳过或报错
    return {"type": "text", "content": f"[Luma] Received prompt: {all_text[:100]}..."}


# ======================
# 统一入口
# ======================
def call_universal_api(
    ai_config: Dict[str, Any],
    system_prompt: Optional[str],
    parts: List[Dict[str, Any]],
    temperature: float = 0.7,
    seed: int = 0
) -> Dict[str, Any]:
    provider = ai_config["provider"]
    model_name = ai_config["model_name"]
    api_key = ai_config["api_key"]
    extra_params = parse_extra_params(ai_config.get("extra_params"))

    try:
        if provider == "Gemini":
            return gemini_call(model_name, api_key, system_prompt, parts, temperature, extra_params)
        elif provider == "OpenAI":
            return openai_call(model_name, api_key, system_prompt, parts, temperature, extra_params)
        elif provider == "Qwen":
            return qwen_call(model_name, api_key, system_prompt, parts, temperature, extra_params)
        elif provider == "Doubao":
            return doubao_call(ai_config, system_prompt, parts, temperature, extra_params)
        elif provider == "Grok":
            return grok_call(model_name, api_key, system_prompt, parts, temperature, extra_params)
        elif provider == "Hailuo":
            return hailuo_call(model_name, api_key, system_prompt, parts, temperature, extra_params)
        elif provider == "Luma":
            return luma_call(model_name, api_key, system_prompt, parts, temperature, extra_params)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    except Exception as e:
        raise RuntimeError(f"API call failed for {provider}: {str(e)}")