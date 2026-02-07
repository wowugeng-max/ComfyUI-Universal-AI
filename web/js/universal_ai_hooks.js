import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "UniversalAI.ModelFilter",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "UniversalAILoader") {
            
            // 拦截节点创建事件
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                
                // 1. 查找对应的 UI 组件 (Widget)
                const providerWidget = this.widgets.find(w => w.name === "provider");
                const modelWidget = this.widgets.find(w => w.name === "model_selection");

                if (!providerWidget || !modelWidget) return r;

                // 2. 定义更新模型列表的异步函数
                const updateModels = async () => {
                    const provider = providerWidget.value;
                    console.log(`[Universal AI] 🔄 Requesting models for: ${provider}`);
                    
                    try {
                        // 向后端请求筛选后的模型列表
                        const response = await fetch(`/universal_ai/get_models?provider=${provider}`);
                        if (!response.ok) throw new Error("Backend API not responding");
                        
                        const models = await response.json();

                        // 💡 关键改动：去掉 models.length > 0 的判断
                        // 只要后端有返回（哪怕是保底模型），就执行更新
                        if (Array.isArray(models)) {
                            // 更新下拉列表的所有可选项
                            modelWidget.options.values = models;
                            
                            // 检查当前选中的值是否还在新列表中
                            // 如果不在（比如从 Gemini 切换到 Grok），则强制选中新列表的第一个
                            if (!models.includes(modelWidget.value)) {
                                modelWidget.value = models[0] || "";
                            }
                            
                            // 强制 ComfyUI 重新绘制画布，确保 UI 立即显示变化
                            app.canvas.setDirty(true, true);
                        }
                    } catch (e) {
                        console.error("[Universal AI] Filter Error:", e);
                    }
                };

                // 3. 监听 Provider 的变化
                // 使用这种方式可以保留原有的 callback 逻辑，同时注入我们的 updateModels
                const oldCallback = providerWidget.callback;
                providerWidget.callback = function () {
                    const result = oldCallback ? oldCallback.apply(this, arguments) : undefined;
                    updateModels();
                    return result;
                };

                // 4. 节点初次加载/创建时，延迟运行一次以初始化列表
                setTimeout(updateModels, 300);

                return r;
            };
        }
    }
});