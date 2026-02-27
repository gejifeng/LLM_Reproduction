# 2025-2026 最新开源 SOTA LLM 架构调研与 7B 模型复现计划

## 1. 调研结果：最新开源 SOTA LLM 架构趋势 (2025-2026)

在 7B-8B 参数量级，开源社区的架构已经高度收敛，并向着**极致推理效率**和**复杂推理能力 (Reasoning)** 演进。代表模型包括 **Llama 3.1 8B**, **Qwen 2.5 7B**, 以及最新的 **DeepSeek-V3/R1** 体系。

### 1.1 现代 Dense 模型 (7B-8B) 的“标配”组件
*   **GQA (Grouped-Query Attention)**: 替代 MHA，大幅降低推理时的 KV Cache 显存占用。
*   **RoPE (Rotary Position Embedding)**: 旋转位置编码，配合 YaRN 等动态缩放技术，轻松支持 128k 甚至更长的上下文。
*   **SwiGLU 激活函数**: 替代 ReLU/GELU，提供更好的梯度流动和表达能力。
*   **RMSNorm**: 替代 LayerNorm，去除均值计算，提升训练稳定性和速度。
*   **Tied Embeddings (部分模型如 Qwen)**: 绑定输入和输出的词表权重，节省参数量。

### 1.2 最新 MoE (混合专家) 架构的核心创新 (以 DeepSeek-V3/R1 为代表)
*   **细粒度专家 (Fine-grained Experts)**: 摒弃传统的少量大专家 (如 Mixtral 的 8 个大专家)，采用大量小专家 (如 256 个小专家)，每次激活其中几个，实现更精准的知识路由。
*   **共享专家 (Shared Experts)**: 划定几个专家为“常驻专家”，不参与路由，所有 Token 都会经过它们。这解决了传统 MoE 模型中“通用知识遗忘”的问题。
*   **MLA (Multi-head Latent Attention)**: DeepSeek 提出的革命性注意力机制，通过将 Key 和 Value 投影到低维隐空间，实现极端的 KV Cache 压缩，极大提升了推理时的并发量 (Batch Size)。
*   **无辅助损失负载均衡 (Auxiliary-Loss-Free Load Balancing)**: 传统的 MoE 需要加一个 Loss 来防止所有 Token 涌向同一个专家，但这会损害模型性能。最新的方法通过动态调整路由偏置 (Bias) 来实现负载均衡，无需额外的 Loss。
*   **MTP (Multi-Token Prediction)**: 训练时同时预测未来多个 Token，增强模型的表征学习能力。

---

## 2. 学习计划：从基础 Transformer 到最新 MoE

为了快速掌握现代成熟的 LLM 架构，建议分三个阶段进行：

### 阶段一：基石 (Foundation) - 彻底吃透 Transformer
*   **目标**: 理解 Attention 机制的本质，手写一个极简的 GPT-2。
*   **实践**: 参考 Andrej Karpathy 的 `nanoGPT` 或 `llm.c`，实现标准的 Multi-Head Attention (MHA)、LayerNorm、GELU 和绝对位置编码。
*   **里程碑**: 能够在一个莎士比亚数据集上训练出一个能生成连贯文本的微型模型。

### 阶段二：现代化 (Modernization) - 掌握 Llama/Qwen 架构
*   **目标**: 将 GPT-2 升级为现代 SOTA Dense 模型架构。
*   **实践**: 
    1. 将 LayerNorm 替换为 **RMSNorm**。
    2. 将绝对位置编码替换为 **RoPE**。
    3. 将 FFN 替换为 **SwiGLU**。
    4. 将 MHA 替换为 **GQA**。
*   **里程碑**: 写出一个 `Llama-3-7B` 的模型定义文件，并能成功加载 HuggingFace 上的开源权重进行推理。

### 阶段三：前沿化 (Frontier) - 进阶 MoE 与强化学习 (RL)
*   **目标**: 掌握 DeepSeek/Qwen 风格的最新 MoE 架构和推理模型 (Reasoning Models) 的训练方法。
*   **实践**:
    1. 实现 **DeepSeekMoE** 路由机制 (Shared Experts + Routed Experts)。
    2. 了解并尝试实现 **MLA** (可选，实现难度较高，但对理解显存优化至关重要)。
    3. 学习 **GRPO (Group Relative Policy Optimization)**，这是 DeepSeek-R1 采用的强化学习算法，无需 Critic 模型，极大地降低了 RLHF 的显存门槛。

---

## 3. 7B 模型全流程复现实战计划 (受限于硬件的破局之道)

既然受限于硬件，且目标是“完整走几遍训练流程”，我建议你**不要**直接从头预训练一个 7B 的 Dense 模型 (这需要极大的算力和数据)。

**最佳策略：复现一个“总参数 7B，激活参数 1.5B”的 MoE 模型。**
这样既能学习最新的 MoE 架构，又能大幅降低单次 Forward/Backward 的计算量，让你能在有限的硬件 (如单卡 A100/4090 或双卡) 上快速跑通全流程。

### 步骤 1：数据准备 (Data Preparation)
*   **Pre-training 数据**: 下载 HuggingFace 上的 `HuggingFaceFW/fineweb-edu` (高质量教育数据)，抽取 10B-20B Tokens 的子集。
*   **SFT 数据**: 使用 `HuggingFaceH4/ultrachat_200k` 或 `Open-Orca/OpenOrca`。
*   **RL/Reasoning 数据**: 使用 `Open-R1/OpenR1-Math-220k` (用于训练类似 DeepSeek-R1 的思维链能力)。
*   **Tokenization**: 直接复用 `Qwen2.5` 或 `Llama-3` 的 Tiktoken 词表 (128k 词表大小，多语言支持好)。

### 步骤 2：模型架构搭建 (Model Architecture)
*   在 `src/model/` 下从零编写你的 MoE 模型。
*   **配置建议**: 
    *   Hidden size: 2048
    *   Layers: 24
    *   Attention: GQA (8 KV heads, 32 Q heads)
    *   MoE: 1 个 Shared Expert, 64 个 Routed Experts (每次激活 4 个)。
    *   总参数量约 7B，每次前向传播激活参数约 1.5B。

### 步骤 3：微型预训练 (Mini Pre-training)
*   **目标**: 跑通分布式训练框架。
*   **工具**: 使用 PyTorch FSDP (Fully Sharded Data Parallel) 或 DeepSpeed Zero-2/3。
*   **任务**: 在 10B Tokens 的 FineWeb-Edu 子集上进行 Next-Token Prediction 训练。观察 Loss 下降曲线，掌握 Checkpoint 保存与恢复、Wandb 监控、学习率预热与衰减 (Cosine Decay)。

### 步骤 4：指令微调 (Supervised Fine-Tuning - SFT)
*   **目标**: 让模型学会对话格式 (Chat Template)。
*   **任务**: 将 SFT 数据格式化为 ChatML 格式 (`<|im_start|>user...<|im_end|>\n<|im_start|>assistant...`)。
*   **技术**: 如果显存不够全参数微调，实现 **LoRA** 或 **QLoRA** (使用 `peft` 库) 注入到 Attention 和 MoE 专家层中进行微调。

### 步骤 5：强化学习与推理对齐 (RLHF / GRPO)
*   **目标**: 激发模型的 CoT (Chain-of-Thought) 推理能力，复现 R1 的 "Aha moment"。
*   **任务**: 放弃传统的 PPO (需要 4 个模型，显存爆炸)，采用 **DPO (Direct Preference Optimization)** 或最新的 **GRPO**。
*   **实践**: 使用 `trl` (Transformer Reinforcement Learning) 库，在数学/逻辑数据集上，通过规则奖励 (Rule-based Reward，例如答案是否正确) 来运行 GRPO 算法，观察模型输出长度的增加和 `<think>` 标签的涌现。

---

## 4. 推荐的开源框架与工具链
为了避免重复造轮子，建议在理解原理后，基于以下优秀的开源框架进行二次开发和魔改：
1.  **torchtune (Meta)**: 极度推荐！代码极其干净，没有复杂的抽象，非常适合学习和魔改 Llama 架构。
2.  **Llama-Factory**: 适合快速跑通 SFT 和 DPO/GRPO 流程，支持几乎所有主流模型和微调方法。
3.  **TRL (HuggingFace)**: 用于实现 DPO 和 GRPO 的首选库。
4.  **vLLM**: 训练完成后，用于部署和测试你的模型推理速度。