from .baseSyncer import BaseModelSyncer
from ..utils import get_model_tag
from .modelSyncerFactory import ModelSyncerFactory

@ModelSyncerFactory.register("DeepSeek")
class DeepSeekSyncer(BaseModelSyncer):
    provider_name = "DeepSeek"

    def sync(self) -> list[str]:
        url = "https://api.deepseek.com/v1/models"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            resp = requests.get(url, headers=headers, timeout=10, verify=False)
            if resp.status_code == 200:
                data = resp.json().get("data", [])
                models = []
                for item in data:
                    raw_name = item.get("id")
                    if raw_name:
                        models.append(get_model_tag(raw_name, "DeepSeek"))
                return models
        except Exception:
            pass
        return []