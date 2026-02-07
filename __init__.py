# __init__.py

from .universal_nodes import (
    UniversalAILoader,
    UniversalAIRunner,
    UniversalAISetConfig,
    UniversalAIGetConfig
)
from .file_nodes import (
    UniversalFileWriter,
    UniversalFileReader
)
from .text_nodes import (
    TextInputNode
)

# 💡 关键：告诉 ComfyUI 插件的 JS 存放路径
WEB_DIRECTORY = "./web"

# Class mappings
NODE_CLASS_MAPPINGS = {
    "UniversalAILoader": UniversalAILoader,
    "UniversalAIRunner": UniversalAIRunner,
    "UniversalAISetConfig": UniversalAISetConfig,
    "UniversalAIGetConfig": UniversalAIGetConfig,
    "UniversalFileWriter": UniversalFileWriter,
    "UniversalFileReader": UniversalFileReader,
    "TextInput": TextInputNode
}

# Display names
NODE_DISPLAY_NAME_MAPPINGS = {
    "UniversalAILoader": "AI Load Model • 🧠 wowugeng",
    "UniversalAIRunner": "AI Run Task • 🧠 wowugeng",
    "UniversalAISetConfig": "AI Set Global Config • 🧠 wowugeng",
    "UniversalAIGetConfig": "AI Get Global Config • 🧠 wowugeng",
    "UniversalFileWriter": "File Write TXT CSV • 🧠 wowugeng",
    "UniversalFileReader": "File Read TXT CSV • 🧠 wowugeng",
    "TextInput": "Text Input • 🧠 wowugeng"
}

# 导出所有必要变量
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]