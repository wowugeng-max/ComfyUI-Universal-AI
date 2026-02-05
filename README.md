<br>
这套自定义节点是一个 通用多模态 AI 接入框架（Universal AI Integration Framework），
</br>
专为 ComfyUI 工作流打造。它允许用户在图形化界面中无缝调用来自多个主流大模型平台（如 Google Gemini、OpenAI、Qwen、Doubao、Grok 等）的文本、图像甚至视频生成能力，
<br>
支持图文输入、视频帧采样、模型动态刷新等高级功能。
</br>
<br>
✨ 核心特点
多平台统一接入
支持 Gemini / OpenAI / Qwen / Doubao / Grok / Hailuo / Luma 七大 AI 提供商。
自动同步各平台可用模型列表（通过 sync_all_models），并缓存到本地 universal_model_cache.json，避免重复请求。
</br>
<br>
真正的多模态输入
可同时传入：文本提示（user/system prompt）+ 额外长文本 + 多张图像 + 视频（自动抽帧）。
图像/视频帧自动转为 Base64 并压缩（可配置最大尺寸），适配各 API 要求。
</br>
<br>
灵活的模型选择与扩展
内置默认模型（如 gemini-1.5-flash, gpt-4o, qwen-vl-max）。
支持自定义模型名、Base URL、API 版本和额外参数（extra_params），便于私有部署或实验性模型接入。
</br>
智能输出处理
若模型返回图像（如 Gemini 的 imagen 系列），自动解析为 ComfyUI 的 IMAGE 张量。
若响应中包含视频链接（如 .mp4），自动下载并解码为视频帧张量（video_frames 输出）。
<br>
工程健壮性
包含错误处理、进度条反馈（comfy.utils.ProgressBar）、输入截断（长文本 >10k 字符）、种子控制等细节优化。
模块化设计：utils.py 处理数据转换，api_adapters.py 封装各平台调用逻辑，universal_nodes.py 定义节点接口。
</br>
<br>

🧩 节点说明
</br>
<br>
🌍 AI Model Loader (Ultimate)
配置 AI 提供商、API 密钥、模型名称。
可选“刷新模型列表”以获取最新可用模型。
输出 AI_CONFIG，供 Runner 节点使用。
</br>
<br>
🌍 AI Task Runner (Ultimate)
接收 AI_CONFIG 和多模态输入。
执行推理，输出：文本结果 + 生成图像（如有）+ 视频帧（如有）。
支持温度、种子、最大图像尺寸等参数调节。
</br>
<br>
📌 适用场景
多模态内容理解（图文问答、OCR、视频摘要）
AI 绘图（通过支持图像生成的模型）
视频生成工作流（结合 Luma 或返回视频链接的模型）
快速对比不同大模型在相同输入下的表现
构建自动化 AI 内容生产流水线
</br>
💡 总结
你的节点实现了 “一套 UI，通吃多模态大模型” 的目标，极大提升了 ComfyUI 在 AI 应用开发中的灵活性和生产力。无论是研究、创作还是产品集成，这套工具都能显著降低多平台 AI 调用的复杂度。
