自用节点
调试了qwen、Gemini，只需填入API KEY, 刷新模型列表

Doubao 麻烦一点需要填入API KEY,custom_model_name里填入ep_开头的Endpoint，因为走的火山平台

首次使用时，运行一次loader，同步模型名称，出现类似的log

🔄 [Universal AI] Starting sync for Qwen...

💾 [Universal AI] Qwen cache updated with 181 items.

刷新页面或者新建节点就能看到模型列表（universal_model_cache.json已经缓存了一些，如果不要删了重新获取）


🧠 ComfyUI-Universal-AI 一套 UI，通吃多模态大模型 —— 通用多平台 AI 接入框架 for ComfyUI
本插件为 ComfyUI 提供统一的图形化接口，无缝调用 Google Gemini、OpenAI、Qwen（通义千问）、Doubao（豆包）、Grok、Hailuo（海螺）、Luma 等七大主流 AI 平台的文本、图像、视频生成与理解能力。支持图文混合输入、视频帧采样、模型动态刷新、全局配置管理等高级功能，极大简化多模态 AI 工作流构建。 💡 作者提示：目前主要调试了 Qwen 和 Gemini，效果较好。填入 API Key 即可开箱使用！

✅ 一、核心特性 🌍 多平台统一接入 一套节点，支持 7 大 AI 提供商，无需为每个平台单独写逻辑。 🧠 真正的多模态输入 同时传入：系统提示 + 用户指令 + 长文本 + 多张图像 + 视频帧，自动处理格式转换。 🖼️ 智能输出处理 若模型返回 Base64 图像（如 Gemini Imagen），自动转为 ComfyUI IMAGE 张量。 若响应包含 视频链接（如 .mp4），自动下载并解码为 video_frames 张量。 🔄 模型列表自动同步 点击“刷新模型列表”，自动从各平台拉取最新可用模型，并缓存至 universal_model_cache.json，避免重复请求。 ⚙️ 高度可扩展 支持自定义 model_name、base_url、api_version 和 extra_params，轻松接入私有部署或实验性模型。 🔧 工程健壮性 内置错误处理、进度条反馈、长文本截断（>10k 字符）、种子控制等细节优化。

📦 二、安装指南 . 克隆仓库 bash

编辑

cd ComfyUI/custom_nodes/ git clone https://github.com/wowugeng-max/ComfyUI-Universal-AI.git . 安装依赖 bash

编辑

cd ComfyUI-Universal-AI pip install -r requirements.txt 依赖包括：opencv-python, requests, Pillow, google-generativeai, chardet . 重启 ComfyUI 启动后，在节点菜单中搜索 “🧠 wowugeng” 即可找到所有节点。

🔌 三、核心节点说明 . AI Model Loader (Ultimate) • 🧠 wowugeng 配置 AI 提供商与模型。 表格 参数 说明 provider 选择平台：Gemini / OpenAI / Qwen / Doubao / Grok / Hailuo / Luma api_key 填写 API Key（支持多个 key 用逗号分隔，随机选用） model_selection 从缓存列表选择模型（带 [CHAT]/[VISION]/[IMAGE] 标签） refresh_list ✅ 勾选以同步最新模型列表（需有效 Key） custom_model_name （可选）手动输入模型 ID，优先级高于下拉框 ✅ 输出：AI_CONFIG（供 Runner 使用）

. AI Task Runner (Ultimate) • 🧠 wowugeng 执行多模态推理任务。 输入： ai_config：来自 Loader system_prompt / user_prompt：角色设定与主指令 text（可选）：附加长文本（自动截断至 10k 字） images（可选）：IMAGE 张量（多图支持） video（可选）：视频帧张量（自动按 max_video_frames 采样关键帧） 输出： text：AI 返回的文本 image：若模型支持生图（如 Qwen 图像模型），返回 IMAGE video_frames：若文本含 .mp4 链接，自动下载并转为视频帧 ⚙️ 支持调节 temperature、seed、max_image_size（自动压缩 Base64 图像）

. 全局配置工具（Utils） AI Set Global Config：将 AI_CONFIG 存入全局字典（指定 key） AI Get Global Config：通过 key 读取配置 适用于复杂工作流，避免重复连接。

. 辅助工具节点 Text Input：纯文本输入 File Read/Write TXT CSV：读写本地 .txt / .csv 文件，支持路径自定义

🌐 四、模型能力标签说明 插件自动为模型打标签，便于识别用途： 表格 标签 能力 示例 [CHAT] 文本对话 qwen-max, gpt-4o [VISION] 图文理解 qwen-vl-max, gemini-1.5-flash [IMAGE] 文生图 qwen-image-max, [IMAGE] art-model-xxx [AUDIO] 语音合成 qwen-tts-flash [CODE] 代码生成 qwen-coder-plus 📁 模型列表缓存在 universal_model_cache.json，可手动删除以重置。 🔑 五、API Key 与安全建议 推荐方式：通过环境变量设置（节点留空时自动读取）： bash

编辑

export UNIVERSAL_AI_API_KEY="your_key_here" python main.py 多 Key 轮询：在 api_key 字段填写 key1,key2,key3，插件会随机选择。 自定义 URL：通过 custom_base_url 接入私有 API 网关。

🎯 六、典型应用场景 多模态内容理解 → 图文问答、OCR 识别、视频摘要（配合视频帧输入） AI 绘图工作流 → 使用 qwen-image-max 或 Doubao 生图模型，直接在 ComfyUI 中生成图像 跨模型对比实验 → 同一输入，快速切换不同平台模型，对比输出效果 自动化内容生产 → AI 生成文案 → File Writer 保存 → 后续节点调用

❓ 七、常见问题（FAQ） Q：模型下拉框为空？ A：请检查 api_key 是否正确，并勾选 refresh_list。部分平台（如 Doubao）需开通对应 region 权限。 Q：为什么生图模型没返回图像？ A：确认模型名称含 [IMAGE] 标签。目前 Qwen 图像模型 和 Doubao 生图模型 支持最佳。 Q：视频无法加载？ A：确保视频 URL 可公开访问。本地视频需先通过其他节点（如 Load Video）转为帧张量。 Q：出现 SSL 证书错误？ A：插件已禁用 SSL 验证（verify=False），通常不影响使用。若仍失败，请检查网络代理设置。

📜 八、致谢 感谢 ComfyUI 提供强大的可视化工作流框架。 感谢各 AI 平台（Google, OpenAI, 阿里云, 字节跳动等）提供卓越的大模型 API。 本项目为个人开发工具，欢迎 Star、Fork、提交 Issue！ —— by wowugeng