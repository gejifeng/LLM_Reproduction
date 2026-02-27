# LLM Reproduction Project Plan (7B MoE)

This document outlines the detailed plan and directory structure for reproducing a 7B-scale Mixture of Experts (MoE) Large Language Model, optimized for limited hardware resources.

## 📂 Project Directory Structure

```text
LLM_Reproduction/
├── configs/                 # 配置文件 (YAML/JSON)
│   ├── pretrain_7b_moe.yaml # 预训练配置 (模型超参、训练参数)
│   ├── sft_7b_moe.yaml      # 指令微调配置
│   └── dpo_7b_moe.yaml      # 强化学习/对齐配置
├── data/                    # 数据集存放目录 (通常在 .gitignore 中忽略)
│   ├── raw/                 # 原始下载的数据集 (如 FineWeb-Edu, UltraChat)
│   └── processed/           # 预处理后的 Tokenized 数据 (如 .bin 或 .arrow 文件)
├── docs/                    # 项目文档和学习笔记
│   ├── architecture.md      # 模型架构设计文档 (MoE, GQA, RoPE 等细节)
│   └── training_log.md      # 训练日志和实验记录
├── notebooks/               # Jupyter Notebooks (用于数据探索、模型测试、可视化)
│   └── 01_data_exploration.ipynb
├── scripts/                 # 启动脚本 (Shell 脚本)
│   ├── run_pretrain.sh      # 启动分布式预训练的脚本 (如 torchrun)
│   ├── run_sft.sh           # 启动 SFT 的脚本
│   └── run_eval.sh          # 启动评估的脚本
├── src/                     # 核心源代码
│   ├── data/                # 数据处理模块
│   │   ├── dataset.py       # PyTorch Dataset 定义
│   │   └── tokenizer.py     # Tokenizer 加载与处理逻辑
│   ├── eval/                # 评估模块
│   │   └── evaluator.py     # 评估脚本 (如困惑度计算、生成测试)
│   ├── model/               # 模型架构定义
│   │   ├── modern_llm.py    # 基础组件 (RMSNorm, RoPE, GQA)
│   │   ├── moe_layer.py     # MoE 路由与专家层实现 (DeepSeekMoE 风格)
│   │   └── model.py         # 完整的 LLM 模型类组装
│   ├── train/               # 训练循环与优化器
│   │   ├── trainer.py       # 核心训练循环 (Forward, Backward, Step)
│   │   └── utils.py         # 辅助工具 (学习率调度、Checkpoint 保存/加载)
│   └── generate.py          # 推理生成脚本 (用于测试模型输出)
├── README.md                # 项目主文档 (包含整体计划和快速开始)
└── requirements.txt         # Python 依赖包列表
```

---

## 📅 Detailed Execution Plan (阶段性模块化推进与自检指南)

为了方便阶段性推进和自我归档，本计划将整个复现过程拆解为 5 个核心模块 (Modules)。每个模块都包含明确的**目标**、**任务清单 (Checklist)**、**自检标准 (Self-Check Criteria)** 和**归档记录 (Archiving)**。

---

### Module 0: Foundation - GPT-2 Baseline (基石：彻底吃透 Transformer)
**目标:** 理解 Attention 机制的本质，手写一个极简的 GPT-2，作为后续现代架构的对比基线 (Baseline)。

#### 0.1 极简 GPT-2 架构实现 (`src/model/gpt2/model.py`)
*   [ ] 实现标准的 `Multi-Head Attention (MHA)`。
*   [ ] 实现标准的 `LayerNorm`。
*   [ ] 实现 `GELU` 激活函数。
*   [ ] 实现绝对位置编码 (Absolute Position Embedding)。
*   [ ] 组装完整的 GPT-2 模型。
*   **自检标准:** 编写单元测试，确保输入 dummy data 能输出正确的 logits shape。

#### 0.2 微型数据集训练测试 (`notebooks/00_gpt2_shakespeare.ipynb`)
*   [ ] 下载莎士比亚微型数据集 (tinyshakespeare)。
*   [ ] 编写一个极简的训练循环 (无需分布式，单卡即可)。
*   [ ] 训练模型并观察 Loss 下降。
*   **自检标准:** 模型能够生成连贯的英文单词 (即使语法不完美)。
*   **归档记录:** 在 `docs/archive/module_0_gpt2/README.md` 中记录训练 Loss 和生成的样本文本。

---

### Module 1: Modernization & Architecture (现代化架构搭建)
**目标:** 实现现代 SOTA LLM 的核心组件，并组装成一个 7B 规模的 MoE (Mixture of Experts) 模型架构。

#### 1.1 核心组件实现 (`src/model/modern_llm.py`)
*   [x] 实现 `RMSNorm` (替代 LayerNorm)。
*   [x] 实现 `RoPE` (旋转位置编码，支持长上下文)。
*   [x] 实现 `GQA` (分组查询注意力，优化 KV Cache)。
*   [x] 实现 `SwiGLU` (现代 FFN 标配)。
*   **自检标准:** 编写简单的单元测试，确保输入输出的 Tensor shape 正确，且能正常进行 forward/backward。

#### 1.2 MoE 专家层与路由机制 (`src/model/moe_layer.py`)
*   [ ] 实现基础的 Top-K 路由机制 (如 Top-2)。
*   [ ] 实现 DeepSeek 风格的细粒度专家 (Fine-grained Experts)。
*   [ ] 实现共享专家 (Shared Experts) 机制。
*   [ ] (可选/进阶) 实现无辅助损失的负载均衡 (Auxiliary-Loss-Free Load Balancing)。
*   **自检标准:** 
    *   输入一个 batch 的 token，验证路由分布是否合理 (没有全部坍塌到一个专家)。
    *   验证 Shared Experts 是否对所有 token 都激活。
    *   计算单次 forward 的激活参数量，确保符合预期 (如总参数 7B，激活 1.5B)。

#### 1.3 模型组装与验证 (`src/model/model.py`)
*   [ ] 组装完整的 `TransformerBlock` (Attention + MoE/FFN)。
*   [ ] 定义完整的 `MoELanguageModel` 类 (Embedding -> Blocks -> RMSNorm -> LM Head)。
*   [ ] 编写权重初始化函数 (如正态分布初始化，针对残差连接的缩放)。
*   **自检标准:** 
    *   实例化模型，打印模型结构和总参数量。
    *   输入 dummy data (如 `torch.randint(0, vocab_size, (batch_size, seq_len))`)，确保能输出正确的 logits shape `(batch_size, seq_len, vocab_size)`。
    *   **归档记录:** 在 `docs/architecture.md` 中记录最终确定的模型超参数 (层数、隐藏层维度、专家数量等) 和参数量计算过程。

---

### Module 2: Data Pipeline & Tokenization (数据管道与分词)
**目标:** 准备高质量的预训练数据，并构建高效的数据加载管道。

#### 2.1 Tokenizer 集成 (`src/data/tokenizer.py`)
*   [ ] 选择并加载一个成熟的开源 Tokenizer (推荐 Qwen2.5 或 Llama-3 的 tiktoken)。
*   [ ] 编写包装类，处理特殊 token (如 `<|endoftext|>`, `<|im_start|>`)。
*   **自检标准:** 输入一段包含多语言和代码的文本，验证 encode 和 decode 后的文本是否一致。

#### 2.2 预训练数据准备与打包 (`notebooks/01_data_exploration.ipynb` & `src/data/dataset.py`)
*   [ ] 下载 `HuggingFaceFW/fineweb-edu` 的一个子集 (如 10B tokens)。
*   [ ] 编写脚本将文本数据 tokenize，并拼接成固定长度 (如 `seq_len = 2048`) 的 chunks。
*   [ ] 将处理好的数据保存为高效的二进制格式 (如 `.bin` 或 `.arrow`) 到 `data/processed/`。
*   **自检标准:** 
    *   读取生成的 `.bin` 文件，解码前几个 chunk，人工检查文本是否连贯，是否正确处理了文档边界 (EOD token)。
    *   **归档记录:** 在 `docs/training_log.md` 中记录使用的数据集版本、数据量 (Tokens 数) 和预处理耗时。

#### 2.3 PyTorch Dataset 与 DataLoader (`src/data/dataset.py`)
*   [ ] 实现 `PretrainDataset` 类，支持从二进制文件中高效读取内存映射 (memory-mapped) 数据。
*   [ ] 配置 DataLoader，确保在多进程/多 GPU 下数据加载不会成为瓶颈。
*   **自检标准:** 遍历 DataLoader 几个 epoch，测试数据加载的吞吐量 (Tokens/second)。

---

### Module 3: Mini Pre-training (微型预训练)
**目标:** 跑通分布式训练框架，让模型学会基本的语言建模 (Next-Token Prediction)。

#### 3.1 训练循环与优化器 (`src/train/trainer.py` & `src/train/utils.py`)
*   [ ] 实现核心的训练循环 (Forward, Loss 计算, Backward, Optimizer Step)。
*   [ ] 集成 AdamW 优化器，并实现带有 Warmup 的 Cosine 学习率衰减 (`utils.py`)。
*   [ ] 实现梯度裁剪 (Gradient Clipping)。
*   [ ] 实现 Checkpoint 的定期保存与恢复逻辑。
*   **自检标准:** 编写一个过拟合测试 (Overfit Test)：使用极小的数据集 (如 1 个 batch) 训练，观察 Loss 是否能快速降到接近 0。

#### 3.2 分布式训练集成 (FSDP / DeepSpeed)
*   [ ] (如果有多卡) 集成 PyTorch FSDP 或 DeepSpeed Zero-2/3，以支持 7B 模型的训练。
*   [ ] 配置 Wandb 或 TensorBoard 进行实验监控。
*   **自检标准:** 启动多卡训练，验证显存占用是否符合预期，各个 GPU 的负载是否均衡。

#### 3.3 启动微型预训练 (`scripts/run_pretrain.sh`)
*   [ ] 编写 `configs/pretrain_7b_moe.yaml`，设置合理的超参数 (Batch size, LR, etc.)。
*   [ ] 在准备好的 10B token 数据集上启动训练。
*   **自检标准:** 
    *   监控 Wandb 上的 Loss 曲线，确保其平稳下降。
    *   训练结束后，使用 `src/generate.py` 生成一些文本，检查模型是否学会了基本的语法和词汇。
    *   **归档记录:** 在 `docs/training_log.md` 中记录预训练的超参数、最终 Loss、训练时长和生成的 sample 文本。

---

### Module 4: Supervised Fine-Tuning (指令微调 SFT)
**目标:** 让预训练模型学会遵循指令，适应对话格式。

#### 4.1 SFT 数据格式化
*   [ ] 下载指令数据集 (如 `HuggingFaceH4/ultrachat_200k`)。
*   [ ] 将数据格式化为 ChatML 格式 (`<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n...<|im_end|>`)。
*   [ ] 修改 Dataset 逻辑：在计算 Loss 时，**只对 Assistant 的回复部分计算 Loss** (Mask 掉 User prompt 部分)。
*   **自检标准:** 打印几个格式化后的样本及其对应的 Loss Mask，人工核对 Mask 是否正确覆盖了非 Assistant 部分。

#### 4.2 SFT 训练执行 (`scripts/run_sft.sh`)
*   [ ] (可选) 如果显存不足，集成 `peft` 库实现 LoRA/QLoRA 微调。
*   [ ] 编写 `configs/sft_7b_moe.yaml`，通常 SFT 的学习率比预训练小一个数量级。
*   [ ] 启动 SFT 训练。
*   **自检标准:** 
    *   训练完成后，使用 `src/generate.py` 进行交互式对话测试。
    *   输入 "你好" 或 "写一首诗"，检查模型是否能以 Assistant 的口吻正确回复，且不会无限生成。
    *   **归档记录:** 在 `docs/training_log.md` 中记录 SFT 的数据集、超参数和几个典型的对话测试结果。

---

### Module 5: Alignment & Reasoning (对齐与推理强化 RLHF/GRPO)
**目标:** 激发模型的思维链 (CoT) 能力，复现类似 DeepSeek-R1 的推理特性。

#### 5.1 强化学习环境准备
*   [ ] 准备推理数据集 (如 `Open-R1/OpenR1-Math-220k`)，包含问题和标准答案。
*   [ ] 选择算法：推荐使用 **GRPO** (Group Relative Policy Optimization)，因为它不需要 Critic 模型，极大地节省了显存。
*   [ ] 定义基于规则的奖励函数 (Rule-based Reward)：例如，提取模型输出中 `<answer>...</answer>` 的内容，与标准答案比对，正确给正奖励，格式错误给负奖励。

#### 5.2 RL 训练执行 (`scripts/run_eval.sh` -> `dpo_7b_moe.yaml`)
*   [ ] 使用 `trl` (Transformer Reinforcement Learning) 库集成 GRPO 算法。
*   [ ] 启动训练，监控模型输出长度的变化 (通常推理模型的输出会随着训练变长)。
*   **自检标准:** 
    *   观察训练过程中的生成样本，检查模型是否开始自发地使用 `<think>...</think>` 标签进行思考。
    *   测试几个未见过的数学题，评估准确率是否提升。
    *   **归档记录:** 在 `docs/training_log.md` 中记录 RL 阶段的 Reward 曲线、输出长度变化曲线，以及典型的 "Aha moment" (模型自我纠错) 的生成案例。

---

### 🏁 最终评估与总结 (Final Evaluation)
*   [ ] 使用 `lm-evaluation-harness` 在标准 Benchmark (如 MMLU, GSM8K) 上评估最终模型。
*   [ ] 整理所有文档，完成项目复盘。
