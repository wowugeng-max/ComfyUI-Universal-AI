import requests
from .baseSyncer import BaseModelSyncer
from ..utils import get_model_tag
from .modelSyncerFactory import ModelSyncerFactory

@ModelSyncerFactory.register("Doubao")
class DoubaoSyncer(BaseModelSyncer):
    """字节豆包（火山引擎）模型同步器"""
    provider_name = "Doubao"
    def sync(self) -> list[str]:
        regions = ["cn-beijing", "cn-shanghai"]
        models = []
        headers = {"Authorization": f"Bearer {self.api_key}"}

        for region in regions:
            url = f"https://ark.{region}.volces.com/api/v3/endpoints"
            try:
                resp = requests.get(url, headers=headers, timeout=15, verify=False)
                if resp.status_code == 200:
                    items = resp.json().get("items", [])
                    for ep in items:
                        model_name = str(ep.get("model", {}).get("name", "")).lower()
                        endpoint_id = ep.get("endpoint_id")
                        if model_name and endpoint_id:
                            tag = get_model_tag(model_name, "Doubao")
                            models.append(f"{tag} {endpoint_id}")
            except Exception:
                # 单个区域失败不影响其他区域
                continue
        return models