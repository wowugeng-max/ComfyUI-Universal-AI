# syncers/factory.py
from .geminiSyncer import GeminiSyncer
from .openaiSyncer import OpenAISyncer
from .grokSyncer import GrokSyncer
from .qwenSyncer import QwenSyncer
from .doubaoSyncer import DoubaoSyncer
# 其他提供商导入...

class ModelSyncerFactory:
    _syncers = {
        "Gemini": GeminiSyncer,
        "OpenAI": OpenAISyncer,
        "Grok": GrokSyncer,
        "Qwen": QwenSyncer,
        "Doubao": DoubaoSyncer,
        # 继续添加...
    }

    @classmethod
    def get_syncer(cls, provider: str, api_key: str):
        syncer_class = cls._syncers.get(provider)
        if not syncer_class:
            return None
        return syncer_class(api_key)