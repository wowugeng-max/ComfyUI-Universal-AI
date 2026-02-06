from .universal_nodes import UniversalAILoader, UniversalAIRunner, UniversalAISetConfig, UniversalAIGetConfig

NODE_CLASS_MAPPINGS = {
    "UniversalAILoader": UniversalAILoader,
    "UniversalAIRunner": UniversalAIRunner,
    "UniversalAISetConfig": UniversalAISetConfig,
    "UniversalAIGetConfig": UniversalAIGetConfig,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UniversalAILoader": "🌍 AI Model Loader (Ultimate)",
    "UniversalAIRunner": "🌍 AI Task Runner (Ultimate)",
    "UniversalAISetConfig": "💾 Set Global AI Config",
    "UniversalAIGetConfig": "📥 Get Global AI Config",
}