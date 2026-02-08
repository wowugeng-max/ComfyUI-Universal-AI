import os
import csv
import comfy.utils
from datetime import datetime  # 💡 引入时间模块处理时间戳

class UniversalFileWriter:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
                "file_path": ("STRING", {"default": "/path/to/output"}),
                "file_name": ("STRING", {"default": "output"}),
                "file_type": (["txt", "csv"], {"default": "txt"}),
                # 💡 新增：是否覆盖现有文件
                "overwrite": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path_full",)
    FUNCTION = "write_file"
    CATEGORY = "Universal_AI/Utils"
    OUTPUT_NODE = True

    def write_file(self, text, file_path, file_name, file_type, overwrite):
        path_clean = file_path.strip()
        os.makedirs(path_clean, exist_ok=True)
        
        ext = file_type.lower()
        name_clean = file_name.strip()
        
        # 初始目标路径
        full_path = os.path.join(path_clean, f"{name_clean}.{ext}")

        # 💡 核心逻辑：如果不覆盖且文件已存在，生成带时间戳的副本名
        if not overwrite and os.path.exists(full_path):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            full_path = os.path.join(path_clean, f"{name_clean}_{timestamp}.{ext}")

        try:
            if ext == "txt":
                with open(full_path, "w", encoding="utf-8") as f:
                    f.write(text)
            elif ext == "csv":
                lines = [line for line in text.strip().split("\n") if line]
                with open(full_path, "w", encoding="utf-8", newline="") as f:
                    writer = csv.writer(f)
                    for line in lines:
                        writer.writerow(line.split(","))
            else:
                raise ValueError("Unsupported file type")
        except Exception as e:
            raise RuntimeError(f"File write failed: {e}")

        print(f"[UniversalFileWriter] Saved to: {full_path}")
        return (full_path,)


class UniversalFileReader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "file_path": ("STRING", {"default": "/path/to/input"}),
                "file_name": ("STRING", {"default": "input"}),
                "file_type": (["txt", "csv"], {"default": "txt"}),
                "preview_length": ("INT", {"default": 500, "min": 10, "max": 2000}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("content", "preview")
    FUNCTION = "read_file"
    CATEGORY = "Universal_AI/Utils"

    def read_file(self, file_path, file_name, file_type, preview_length):
        ext = file_type.lower()
        full_path = os.path.join(file_path.strip(), f"{file_name.strip()}.{ext}")

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"File not found: {full_path}")

        try:
            if ext == "txt":
                with open(full_path, "r", encoding="utf-8") as f:
                    content = f.read()
            elif ext == "csv":
                rows = []
                with open(full_path, "r", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    for row in reader:
                        rows.append(",".join(row))  # 保持为逗号分隔字符串
                content = "\n".join(rows)
            else:
                raise ValueError("Unsupported file type")
        except Exception as e:
            raise RuntimeError(f"File read failed: {e}")

        preview = content[:preview_length] + ("..." if len(content) > preview_length else "")
        return (content, preview)