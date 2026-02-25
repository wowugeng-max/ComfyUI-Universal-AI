# syncers/baseSyncer.py
from abc import ABC, abstractmethod
from typing import List


class BaseModelSyncer(ABC):
    """模型同步器基类，所有提供商同步器必须实现 sync 方法"""

    def __init__(self, api_key: str):
        self.api_key = api_key

    @abstractmethod
    def sync(self) -> List[str]:
        """
        执行同步，返回带 UI 标签的模型名称列表（例如 '[CHAT] gpt-4o'）。
        若同步失败应返回空列表，异常由调用方处理。
        """
        pass