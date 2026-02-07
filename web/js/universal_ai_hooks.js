import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "UniversalAI.Framework.PureFrontend",
    
    async getCustomWidgets() {
        return {
            UNIVERSAL_KEY: (node, inputName, inputData) => {
                const w = node.addWidget("combo", inputName, inputData[1].default || "default", (v) => {
                    w.value = v;
                    if (node.properties) node.properties.value = v;
                }, { values: ["(Wait) Set Node"] });
                node.keyWidget = w;
                return w;
            }
        };
    },

    async beforeRegisterNodeDef(nodeType, nodeData) {
        
        // --- 1. Loader：恢复模型过滤 ---
        if (nodeData.name === "UniversalAILoader") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                
                const pWidget = this.widgets.find(w => w.name === "provider");
                const mWidget = this.widgets.find(w => w.name === "model_selection");

                const updateModels = async () => {
                    // 💡 恢复根据 provider 获取模型的逻辑
                    const resp = await fetch(`/universal_ai/get_models?provider=${pWidget.value}`);
                    const models = await resp.json();
                    if (Array.isArray(models) && mWidget) {
                        mWidget.options.values = models;
                        if (!models.includes(mWidget.value)) mWidget.value = models[0] || "";
                    }
                    // 联动：让下游 Set 刷新 Key
                    app.graph._nodes.filter(n => n.type === "UniversalAISetConfig").forEach(s => s.refreshKey?.());
                };

                pWidget.callback = updateModels;
                mWidget.callback = () => {
                    app.graph._nodes.filter(n => n.type === "UniversalAISetConfig").forEach(s => s.refreshKey?.());
                };
                return r;
            };
        }

        // --- 2. Set 节点：实时更新值 ---
        if (nodeData.name === "UniversalAISetConfig") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                
                this.refreshKey = () => {
                    const linkId = this.inputs[0].link;
                    if (!linkId) return;
                    const origin = app.graph.getNodeById(app.graph.links[linkId].origin_id);
                    if (origin && origin.type === "UniversalAILoader") {
                        const prov = origin.widgets.find(w => w.name === "provider").value;
                        const mod = origin.widgets.find(w => w.name === "model_selection").value;
                        const modShort = mod.replace(/\[.*?\]\s*/, "").split("-")[0];
                        const time = new Date().toTimeString().split(' ')[0].replace(/:/g, ''); 
                        const newKey = `${prov}_${modShort}_ID${this.id}_${time}`;
                        
                        if (this.keyWidget) {
                            const oldKey = this.keyWidget.value;
                            this.keyWidget.value = newKey;
                            
                            // 💡 联动：直接找到正在引用我的 Get 节点，暴力覆盖它们的值
                            app.graph._nodes.filter(n => n.type === "UniversalAIGetConfig").forEach(gn => {
                                if (gn.keyWidget && (gn.keyWidget.value === oldKey || gn.keyWidget.value.includes(`_ID${this.id}_`))) {
                                    gn.keyWidget.value = newKey;
                                }
                            });
                        }
                    }
                };
                this.onConnectionsChange = this.refreshKey;
                return r;
            };
        }

        // --- 3. Get 节点：纯前端扫描，不请求后端 ---
        if (nodeData.name === "UniversalAIGetConfig") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                
                this.refreshFromCanvas = () => {
                    // 💡 核心改动：直接从画布上的所有 Set 节点里抓取当前显示的 Widget 值
                    const allSetNodes = app.graph._nodes.filter(n => n.type === "UniversalAISetConfig");
                    const keys = allSetNodes.map(n => n.widgets.find(w => w.name === "key")?.value).filter(v => v);
                    
                    if (this.keyWidget) {
                        const current = this.keyWidget.value;
                        this.keyWidget.options.values = keys.length > 0 ? keys : ["(Wait) No Set Nodes Found"];
                        
                        // 如果当前值不在列表里，且列表有新值，尝试自动切换
                        if (!keys.includes(current) && keys.length > 0) {
                            // 如果是初始状态，强制选第一个
                            if (current.includes("Wait")) this.keyWidget.value = keys[0];
                        }
                    }
                };

                // 只要鼠标一靠近或者点开下拉框，就即时扫描全图
                this.onMouseEnter = this.refreshFromCanvas;
                this.onMouseDown = this.refreshFromCanvas;
                return r;
            };
        }
    }
});