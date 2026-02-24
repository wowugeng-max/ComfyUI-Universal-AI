import importlib
from typing import Type
from .base import BaseAdapter

class AdapterFactory:
    _provider_to_module = {
        "Gemini": ".gemini",
        "OpenAI": ".openai",
        "Qwen": ".qwen",
        "Doubao": ".doubao",
        "Grok": ".grok",
        "Hailuo": ".hailuo",
        "Luma": ".luma",
    }

    _provider_to_classname = {
        "Gemini": "GeminiAdapter",
        "OpenAI": "OpenAIAdapter",
        "Qwen": "QwenAdapter",
        "Doubao": "DoubaoAdapter",
        "Grok": "GrokAdapter",
        "Hailuo": "HailuoAdapter",
        "Luma": "LumaAdapter",
    }

    @classmethod
    def get_adapter(cls, provider: str) -> BaseAdapter:
        module_path = cls._provider_to_module.get(provider)
        class_name = cls._provider_to_classname.get(provider)
        if not module_path or not class_name:
            raise ValueError(f"Unsupported provider: {provider}")

        try:
            # 动态导入模块（相对于当前包）
            module = importlib.import_module(module_path, package=__package__)
        except ImportError as e:
            # 根据错误信息提示缺失的依赖
            missing_pkg = None
            if "google" in str(e):
                missing_pkg = "google-generativeai"
            elif "requests" in str(e):
                missing_pkg = "requests"
            else:
                missing_pkg = "unknown"
            raise ImportError(
                f"Failed to import adapter for {provider}. "
                f"Please install required dependencies: pip install {missing_pkg}"
            ) from e

        adapter_class = getattr(module, class_name)
        return adapter_class()