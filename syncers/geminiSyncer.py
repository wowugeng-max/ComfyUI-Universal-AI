# syncers/geminiSyncer.py
import requests
from .baseSyncer import BaseModelSyncer
from ..utils import get_model_tag
from .modelSyncerFactory import ModelSyncerFactory

@ModelSyncerFactory.register("Gemini")
class GeminiSyncer(BaseModelSyncer):
    provider_name = "Gemini"

    def sync(self) -> list[str]:
        url = f"https://generativelanguage.googleapis.com/v1beta/models?key={self.api_key}"
        try:
            resp = requests.get(url, timeout=10, verify=False)
            if resp.status_code == 200:
                models = []
                for m in resp.json().get("models", []):
                    raw_name = m["name"].replace("models/", "")
                    models.append(get_model_tag(raw_name, "Gemini"))
                return models
        except Exception:
            pass
        return []