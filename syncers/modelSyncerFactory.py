import importlib
import pkgutil
import inspect
from .baseSyncer import BaseModelSyncer

class ModelSyncerFactory:
    _syncers = {}

    @classmethod
    def register(cls, provider):
        def wrapper(syncer_class):
            if not issubclass(syncer_class, BaseModelSyncer):
                raise TypeError
            cls._syncers[provider] = syncer_class
            return syncer_class
        return wrapper

    @classmethod
    def get_syncer(cls, provider: str, api_key: str):
        if not cls._syncers:
            cls._discover_syncers()
        syncer_class = cls._syncers.get(provider)
        if not syncer_class:
            return None
        return syncer_class(api_key)

    @classmethod
    def _discover_syncers(cls):
        package = importlib.import_module("..syncers", __package__)
        for _, module_name, _ in pkgutil.iter_modules(package.__path__):
            if module_name.startswith("__"):
                continue
            module = importlib.import_module(f"..syncers.{module_name}", __package__)
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, BaseModelSyncer) and obj is not BaseModelSyncer:
                    if hasattr(obj, "provider_name"):
                        cls._syncers[obj.provider_name] = obj