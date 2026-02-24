from .base import BaseAdapter
from ..utils import extract_all_text

class LumaAdapter(BaseAdapter):
    def call(self, ai_config, system_prompt, parts, temperature, seed):
        # Luma 目前仅作为占位，返回模拟响应
        all_text = extract_all_text(parts)
        return {"type": "text", "content": f"[Luma] Received prompt: {all_text[:100]}..."}