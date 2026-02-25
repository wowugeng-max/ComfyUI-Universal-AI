import requests
from .baseSyncer import BaseModelSyncer
from ..utils import get_model_tag
from .modelSyncerFactory import ModelSyncerFactory


@ModelSyncerFactory.register("Qwen")
class QwenSyncer(BaseModelSyncer):
    """阿里云通义千问模型同步器"""
    provider_name = "Qwen"
    def sync(self) -> list[str]:
        url = "https://dashscope.aliyuncs.com/compatible-mode/v1/models"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            resp = requests.get(url, headers=headers, timeout=10, verify=False)
            if resp.status_code == 200:
                data = resp.json().get("data", [])
                models = []
                for item in data:
                    raw_name = item.get("id")
                    if raw_name:
                        models.append(get_model_tag(raw_name, "Qwen"))
                return models
        except Exception:
            # 任何异常均返回空列表，由上层调用处理日志
            pass
        return []