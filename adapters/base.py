from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

class BaseAdapter(ABC):
    """所有平台适配器的抽象基类"""

    @abstractmethod
    def call(self, ai_config: Dict[str, Any], system_prompt: Optional[str],
             parts: List[Dict[str, Any]], temperature: float, seed: int) -> Dict[str, Any]:
        """
        调用模型 API
        :param ai_config: 包含 provider, api_key, model_name, custom_base_url, extra_params 等
        :param system_prompt: 系统提示词
        :param parts: 多模态输入列表，每个元素为 {"type": "text"/"image", "data": ...}
        :param temperature: 温度
        :param seed: 随机种子（部分平台支持）
        :return: 字典 {"type": "text"/"image"/"audio", "content": ...}
        """
        pass