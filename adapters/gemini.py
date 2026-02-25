import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import base64
from .base import BaseAdapter
from ..utils import api_session  # 可选，Gemini SDK 不使用 session
from .factory import AdapterFactory

@AdapterFactory.register("Gemini")
class GeminiAdapter(BaseAdapter):
    provider_name = "Gemini"  # 也可用于自动发现

    def call(self, ai_config, system_prompt, parts, temperature, seed):
        api_key = ai_config["api_key"]
        model_name = ai_config["model_name"]

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
                if isinstance(part["data"], str):
                    img_bytes = base64.b64decode(part["data"])
                    gemini_parts.append({"mime_type": "image/jpeg", "data": img_bytes})

        extra_params = ai_config.get("extra_params", {})
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