import requests
import json

def call_universal_api(config, system_prompt, user_prompt, temperature, base64_images=None, seed=None):
    p = config["provider"]
    key = config["api_key"]
    model = config["model_name"]
    custom_url = config.get("custom_base_url", "").strip()
    api_version = config.get("api_version", "v1beta")
    
    extra = {}
    if config.get("extra_params"):
        try: extra = json.loads(config["extra_params"])
        except: print("⚠️ [Universal AI] JSON 解析失败")

    # 拦截用户可能设置的 stream 模式
    extra.pop("stream", None)

    # 注入种子（若 API 支持）
    if seed is not None: 
        # 修复：Gemini 和许多 API 仅支持 INT32 (最大约 21 亿)
        # ComfyUI 的种子太大会导致 API 报 400 错误
        extra["seed"] = seed % 2147483647

    try:
        # --- Gemini 适配 ---
        if p == "Gemini":
            base = custom_url if custom_url else "https://generativelanguage.googleapis.com"
            
            # 优化：注入安全设置防止拦截
            safety_settings = extra.pop("safetySettings", [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
            ])

            if "imagen" in model.lower():
                url = f"{base}/v1beta/models/{model}:predict?key={key}"
                resp = requests.post(url, json={"instances": [{"prompt": user_prompt}], "parameters": extra}, timeout=60)
            else:
                url = f"{base}/{api_version}/models/{model}:generateContent?key={key}"
                contents = [{"role": "user", "parts": [{"text": f"System Context: {system_prompt}\n\nUser Question: {user_prompt}"}]}]
                if base64_images:
                    for b64 in base64_images:
                        contents[0]["parts"].append({"inline_data": {"mime_type": "image/jpeg", "data": b64}})
                
                payload = {"contents": contents, "generationConfig": {"temperature": temperature, **extra}, "safetySettings": safety_settings}
                resp = requests.post(url, json=payload, timeout=90)

            res = resp.json()
            if resp.status_code != 200:
                raise Exception(f"Gemini API Error {resp.status_code}: {json.dumps(res)}")
            
            if "imagen" in model.lower():
                return {"type": "image", "content": res['predictions'][0]['bytesBase64Encoded']}
            
            # 处理 Gemini 可能因安全原因返回空 candidate 的情况
            if not res.get('candidates'):
                return {"type": "text", "content": f"⚠️ Gemini 拒绝生成内容，原因: {json.dumps(res.get('promptFeedback', {}))}"}
                
            return {"type": "text", "content": res['candidates'][0]['content']['parts'][0]['text']}

        # --- OpenAI 兼容协议 (OpenAI, Grok, Qwen, Doubao) ---
        else:
            eps = {
                "OpenAI": "https://api.openai.com/v1", 
                "Grok": "https://api.x.ai/v1", 
                "Qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "Doubao": "https://ark.cn-beijing.volces.com/api/v3" # 火山方舟适配
            }
            base = custom_url if custom_url else eps.get(p, "")
            full_url = f"{base}/chat/completions"
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [{"type": "text", "text": user_prompt}]}
            ]
            if base64_images:
                for b64 in base64_images: 
                    messages[1]["content"].append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
            
            resp = requests.post(full_url, headers={"Authorization": f"Bearer {key}"}, 
                                 json={"model": model, "messages": messages, "temperature": temperature, **extra}, timeout=90)
            res = resp.json()
            
            if resp.status_code != 200:
                raise Exception(f"{p} API Error {resp.status_code}: {json.dumps(res)}")
                
            return {"type": "text", "content": res['choices'][0]['message']['content']}

    except Exception as e:
        return {"type": "text", "content": f"❌ API Error: {str(e)}"}