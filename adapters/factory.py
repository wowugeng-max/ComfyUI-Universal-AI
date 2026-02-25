import importlib
import pkgutil
import inspect
from .base import BaseAdapter

class AdapterFactory:
    _adapters = {}  # provider -> adapter_class

    @classmethod
    def register(cls, provider):
        """装饰器：注册适配器类"""
        def wrapper(adapter_class):
            if not issubclass(adapter_class, BaseAdapter):
                raise TypeError(f"{adapter_class} must inherit from BaseAdapter")
            cls._adapters[provider] = adapter_class
            return adapter_class
        return wrapper

    @classmethod
    def get_adapter(cls, provider: str) -> BaseAdapter:
        if not cls._adapters:
            cls._discover_adapters()
        adapter_class = cls._adapters.get(provider)
        if not adapter_class:
            raise ValueError(f"Unsupported provider: {provider}")
        return adapter_class()

    @classmethod
    def _discover_adapters(cls):
        """自动扫描 adapters 包下的所有模块，收集被 @register 装饰的类"""
        package = importlib.import_module("..adapters", __package__)
        for _, module_name, _ in pkgutil.iter_modules(package.__path__):
            if module_name.startswith("__"):
                continue
            module = importlib.import_module(f"..adapters.{module_name}", __package__)
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, BaseAdapter) and obj is not BaseAdapter:
                    # 如果类已经被 @register 装饰，会自动填入 _adapters
                    # 这里也可以通过约定的属性（如 provider_name）来注册
                    if hasattr(obj, "provider_name"):
                        cls._adapters[obj.provider_name] = obj