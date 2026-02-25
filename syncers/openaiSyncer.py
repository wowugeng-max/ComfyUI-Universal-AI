# syncers/openai.py
import requests
from .baseSyncer import BaseModelSyncer
from ..utils import get_model_tag
from .modelSyncerFactory import ModelSyncerFactory

@ModelSyncerFactory.register("OpenAI")
class OpenAISyncer(BaseModelSyncer):
    provider_name = "OpenAI"

    def sync(self) -> list[str]:
        url = "https://api.openai.com/v1/models"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            resp = requests.get(url, headers=headers, timeout=10, verify=False)
            if resp.status_code == 200:
                models = []
                for m in resp.json().get("data", []):
                    models.append(get_model_tag(m["id"], "OpenAI"))
                return models
        except Exception:
            pass
        return []