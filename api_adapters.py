"""
统一 API 入口，使用适配器工厂分发请求
"""
from typing import Dict, Any, List, Optional
from .utils import parse_extra_params
from .adapters import AdapterFactory

def call_universal_api(ai_config: Dict[str, Any], system_prompt: Optional[str],
                       parts: List[Dict[str, Any]], temperature: float = 0.7,
                       seed: int = 0) -> Dict[str, Any]:
    """
    统一 API 调用函数，根据 provider 选择适配器并执行
    """
    provider = ai_config.get("provider")
    if not provider:
        raise ValueError("Missing 'provider' in ai_config")

    # 预处理 extra_params（如果是以字符串形式传入）
    if "extra_params" in ai_config and isinstance(ai_config["extra_params"], str):
        ai_config["extra_params"] = parse_extra_params(ai_config["extra_params"])

    print(f"🚀 [Universal AI] Routing -> Provider: {provider}, Model: {ai_config.get('model_name')}")

    adapter = AdapterFactory.get_adapter(provider)
    try:
        result = adapter.call(ai_config, system_prompt, parts, temperature, seed)
        return result
    except Exception as e:
        raise RuntimeError(f"API call failed for {provider}: {str(e)}")