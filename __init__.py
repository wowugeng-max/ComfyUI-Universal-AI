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
# 新增导入
from .text_nodes import (
    TextInputNode
)

# Class mappings
NODE_CLASS_MAPPINGS = {
    "UniversalAILoader": UniversalAILoader,
    "UniversalAIRunner": UniversalAIRunner,
    "UniversalAISetConfig": UniversalAISetConfig,
    "UniversalAIGetConfig": UniversalAIGetConfig,
    "UniversalFileWriter": UniversalFileWriter,
    "UniversalFileReader": UniversalFileReader,
    # 新增节点
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
    # 新增显示名
    "TextInput": "Text Input • 🧠 wowugeng"
}