# text_nodes.py

class TextInputNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"
    CATEGORY = "Universal_AI/Utils"
    OUTPUT_NODE = False

    def execute(self, text):
        return (text,)
