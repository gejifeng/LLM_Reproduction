# LLM Reproduction Project Plan (7B MoE)

> **目标**：从零复现一个 7B 规模的 MoE 大语言模型  
> **对象**：计算数学背景的硕士生（具备扎实的线性代数、概率论、优化理论基础）  
> **特点**：理论结合实践，强调数学原理与代码实现的结合

---

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

## 📚 推荐学习资源汇总

### 必读论文 (按学习顺序排列)

| 序号 | 论文/资源 | 难度 | 建议学习阶段 | 核心收获 |
|------|-----------|------|--------------|----------|
| 1 | "Attention Is All You Need" (Vaswani et al., 2017) | ⭐⭐ | Pre-Module | Transformer 架构奠基之作 |
| 2 | nanoGPT 官方仓库 | ⭐ | Module 0 | 简洁的 GPT 实现参考 |
| 3 | "RoFormer: Enhanced Position Embedding through Rotation" | ⭐⭐⭐ | Module 1 | 旋转位置编码的数学原理 |
| 4 | "LLaMA: Open and Efficient Foundation Language Models" | ⭐⭐⭐ | Module 1 | LLaMA 架构设计 |
| 5 | "Mixtral of Experts" (Mixtral 8x7B) | ⭐⭐⭐ | Module 1 | MoE 实战经典案例 |
| 6 | "DeepSeek-V2: Strong, Efficient, and Cheap Language Models" | ⭐⭐⭐⭐ | Module 1 | 细粒度 MoE 与负载均衡 |
| 7 | "Language Modeling with Gated Linear Networks" | ⭐⭐⭐ | Module 1 | SwiGLU 的数学背景 |
| 8 | "GRPO: Group Relative Policy Optimization" | ⭐⭐⭐ | Module 5 | 强化学习对齐算法 |

### 视频教程

| 资源 | 链接 | 适合阶段 |
|------|------|----------|
| 3Blue1Brown: Neural Networks | https://www.3blue1brown.com/topics/neural-networks | Pre-Module |
| Andrej Karpathy: Let's build GPT | https://www.youtube.com/watch?v=kCc8FmEb1nY | Module 0 |
| Building a MoE from Scratch | https://www.youtube.com/watch?v=... | Module 1 |

### 代码参考

| 资源 | 链接 | 用途 |
|------|------|------|
| nanoGPT | https://github.com/karpathy/nanoGPT | GPT-2 实现参考 |
| TinyStories | https://huggingface.co/datasets/roneneldan/TinyStories | 小型训练数据集 |
| lit-gpt | https://github.com/Lightning-AI/lit-gpt | 完整的 LLM 实现 |

### 可视化理解

| 资源 | 链接 | 用途 |
|------|------|------|
| The Illustrated Transformer | http://jalammar.github.io/illustrated-transformer/ | Transformer 可视化 |
| LLMVisualizer | https://bbycroft.net/llm | LLM 架构可视化 |
| RoPE Visualizer | https://github.com/graykode/rope-visualization | RoPE 原理可视化 |

---

##  Detailed Execution Plan (阶段性模块化推进与自检指南)

为了方便阶段性推进和自我归档，本计划将整个复现过程拆解为 6 个核心模块 (Modules)。每个模块都包含明确的**目标**、**任务清单 (Checklist)**、**自检标准 (Self-Check Criteria)**、**推荐阅读**和**归档记录 (Archiving)**。

---

## Pre-Module: 预备知识模块 (建议学习时间：1-2 周)

> **目标**：建立扎实的理论基础，熟悉开发环境，为后续模块做好准备  
> **对象**：计算数学背景的硕士生 - 您已具备线性代数、概率论、优化理论的基础

### P.1 Transformer 数学基础

#### 核心数学公式

**Self-Attention 的矩阵形式：**
```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```
其中：
- Q ∈ ℝ^(n×d_k): Query 矩阵 (n 为序列长度，d_k 为每个 head 的维度)
- K ∈ ℝ^(n×d_k): Key 矩阵
- V ∈ ℝ^(n×d_v): Value 矩阵
- √d_k: 缩放因子，防止梯度消失

**多头注意力 (Multi-Head Attention)：**
```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

**位置编码的必要性：**
- Attention 机制本身是 permutation-invariant（置换不变）的
- 需要显式注入位置信息：绝对位置编码 vs 相对位置编码

#### 推荐阅读
- 📖 "Attention Is All You Need" 第 3-4 节
- 🌐 The Illustrated Transformer (Jalammar)

#### 自检标准
- [ ] 能手写 Multi-Head Attention 的 PyTorch 代码
- [ ] 理解为什么需要 √d_k 缩放因子（从数学角度解释）
- [ ] 能解释 Attention 矩阵的计算复杂度：O(n²·d)

---

### P.2 神经网络训练基础

#### 核心概念

**损失函数 - Cross-Entropy：**
```
L = -∑_i y_i log(ŷ_i)
```
从信息论角度：最小化交叉熵 = 最小化预测分布与真实分布的 KL 散度

**反向传播与梯度下降：**
- 链式法则在深度网络中的应用
- 随机优化：SGD, Adam, AdamW 的数学原理

**过拟合与正则化：**
- L2 正则化 (Weight Decay)
- Dropout
- 早停法 (Early Stopping)

#### 推荐阅读
- 📖 "Language Modeling with Gated Linear Networks" (了解 SwiGLU 的数学背景)
- 📖 优化器论文：Adam (Kingma & Ba, 2014)

#### 自检标准
- [ ] 能解释 Cross-Entropy 的信息论含义
- [ ] 理解 AdamW 与 Adam 的区别
- [ ] 理解梯度裁剪 (Gradient Clipping) 的数学意义

---

### P.3 开发环境搭建

#### 任务清单
*   [ ] 配置 Python 环境 (推荐 conda)
*   [ ] 安装 PyTorch (确认 CUDA 版本匹配)
*   [ ] 安装 transformers, tiktoken, datasets 等基础库
*   [ ] 配置 GPU 环境验证 (nvidia-smi, torch.cuda.is_available())
*   [ ] 安装 Jupyter Notebook 或配置 VS Code

#### 自检标准
- [ ] 运行 `python -c "import torch; print(torch.cuda.is_available())"` 返回 True
- [ ] 能用 PyTorch 创建简单的 Tensor 并在 GPU 上运算

---

## Module 0: Foundation - GPT-2 Baseline (基石：彻底吃透 Transformer)

**目标:** 理解 Attention 机制的本质，手写一个极简的 GPT-2，作为后续现代架构的对比基线 (Baseline)。

> **数学背景关联**：本模块重点练习矩阵运算、Softmax 的数值稳定性

### 0.1 论文阅读与理论理解
*   [ ] 阅读 "Attention Is All You Need" 全文
*   [ ] 阅读 nanoGPT 代码仓库的 README 和核心代码
*   [ ] 理解 GPT 架构：Embedding → Transformer Blocks → LM Head

#### 推荐阅读
- 📖 "Attention Is All You Need" (Vaswani et al., 2017)
- 💻 https://github.com/karpathy/nanoGPT

---

### 0.2 极简 GPT-2 架构实现 (`src/model/gpt2/model.py`)

#### 任务清单
*   [ ] 实现标准的 `Multi-Head Attention (MHA)`。
    *   数学实现：QK^T 矩阵乘法 → 缩放 → Softmax → 乘 V
    *   思考：如何处理 Mask（因果掩码）？
*   [ ] 实现标准的 `LayerNorm`。
    *   公式：LayerNorm(x) = γ * (x - μ) / σ + β
    *   其中 μ = mean(x), σ = std(x)
*   [ ] 实现 `GELU` 激活函数。
    *   近似公式：GELU(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
*   [ ] 实现绝对位置编码 (Absolute Position Embedding)。
    *   正弦/余弦编码：PE(pos, 2i) = sin(pos/10000^(2i/d_model))
*   [ ] 组装完整的 GPT-2 模型。

#### 自检标准
- [ ] 编写单元测试，确保输入 dummy data 能输出正确的 logits shape
- [ ] 验证因果掩码正确工作（未来位置注意力为 -inf）
- [ ] 验证 Loss 能正常下降

---

### 0.3 微型数据集训练测试 (`notebooks/00_gpt2_shakespeare.ipynb`)

#### 任务清单
*   [ ] 下载莎士比亚微型数据集 (tinyshakespeare)
*   [ ] 编写一个极简的训练循环 (无需分布式，单卡即可)
*   [ ] 训练模型并观察 Loss 下降

#### 自检标准
- [ ] 模型能够生成连贯的英文单词 (即使语法不完美)
- [ ] Loss 从初始值下降到稳定值

#### 归档记录
- [ ] 在 `docs/archive/module_0_gpt2/README.md` 中记录：
    - 训练超参数 (batch_size, learning_rate, max_iters)
    - Loss 曲线图
    - 生成的样本文本 (3-5 个例子)

---

## Module 1: Modernization & Architecture (现代化架构搭建)

**目标:** 实现现代 SOTA LLM 的核心组件，并组装成一个 7B 规模的 MoE (Mixture of Experts) 模型架构。

> **数学背景关联**：本模块重点理解旋转矩阵、群论在 RoPE 中的应用、门控机制的数学原理

### 1.1 核心组件实现 (`src/model/modern_llm.py`)

#### 任务清单

*   [x] 实现 `RMSNorm` (替代 LayerNorm)
    *   公式：RMSNorm(x) = x / RMS(x) * γ, 其中 RMS(x) = √(mean(x²))
    *   优势：比 LayerNorm 更快，效果相当

*   [x] 实现 `RoPE` (旋转位置编码，支持长上下文)
    *   数学原理：对 Query 和 Key 乘以旋转矩阵
    *   公式：RoPE(x_m) = x_m * [cos(mθ), sin(mθ), cos(mθ), sin(mθ), ...]
    *   关键性质：相对位置编码、内积衰减

*   [x] 实现 `GQA` (分组查询注意力，优化 KV Cache)
    *   概念：多个 Query 头共享一个 Key/Value 头
    *   数学：减少 KV 缓存量的同时保持性能

*   [x] 实现 `SwiGLU` (现代 FFN 标配)
    *   公式：SwiGLU(x) = Swish(W_1 x) ⊗ (V x)
    *   其中 Swish(x) = x * sigmoid(x)
    *   相比 ReLU/GELU：更平滑的梯度，更好的表达能力

#### 推荐阅读
- 📖 "RoFormer: Enhanced Position Embedding through Rotation"
- 📖 "Language Modeling with Gated Linear Networks"
- 📖 "LLaMA: Open and Efficient Foundation Language Models"

#### 自检标准
- [ ] 编写简单的单元测试，确保输入输出的 Tensor shape 正确
- [ ] 能正常进行 forward/backward
- [ ] RoPE 的位置编码能正确处理不同位置

---

### 1.2 MoE 专家层与路由机制 (`src/model/moe_layer.py`)

#### 任务清单

*   [ ] 实现基础的 Top-K 路由机制 (如 Top-2)
    *   对每个 token，选择概率最高的 K 个专家
    *   公式：gate(x) = Softmax(W_gate x)，选择 top-k

*   [ ] 实现 DeepSeek 风格的细粒度专家 (Fine-grained Experts)
    *   思想：将大专家拆分为多个小专家，增加灵活性

*   [ ] 实现共享专家 (Shared Experts) 机制
    *   所有 token 都会经过共享专家
    *   作用：捕获通用知识，减少路由不确定性

*   [ ] (可选/进阶) 实现无辅助损失的负载均衡
    *   Auxiliary-Loss-Free Load Balancing (DeepSeek-V2)

#### 数学背景

**路由机制的数学表达：**
```
对于输入 x ∈ ℝ^d：
1. 计算门控分数：g = Softmax(W_gate x) ∈ ℝ^E (E 为专家数)
2. 选择 top-k 专家：indices = topk(g, k)
3. 计算输出：y = Σ_{i∈indices} g_i * Expert_i(x)
```

**负载均衡问题：**
- 理想情况：每个专家处理的 token 数大致相等
- 问题：可能发生路由坍塌 (routing collapse)，即大部分 token 集中到少数专家

#### 推荐阅读
- 📖 "Mixtral of Experts" (Mixtral 8x7B)
- 📖 "DeepSeek-V2: Strong, Efficient, and Cheap Language Models"

#### 自检标准
- [ ] 输入一个 batch 的 token，验证路由分布是否合理 (没有全部坍塌到一个专家)
- [ ] 验证 Shared Experts 是否对所有 token 都激活
- [ ] 计算单次 forward 的激活参数量，确保符合预期 (如总参数 7B，激活 1.5B)

---

### 1.3 模型组装与验证 (`src/model/model.py`)

#### 任务清单

*   [ ] 组装完整的 `TransformerBlock` (Attention + MoE/FFN)
*   [ ] 定义完整的 `MoELanguageModel` 类
    *   Embedding → Blocks → RMSNorm → LM Head
*   [ ] 编写权重初始化函数
    *   正态分布初始化，针对残差连接的缩放

#### 自检标准
- [ ] 实例化模型，打印模型结构和总参数量
- [ ] 输入 dummy data，确保能输出正确的 logits shape
- [ ] **归档记录:** 在 `docs/architecture.md` 中记录最终确定的模型超参数

#### 模型超参数模板

| 参数 | 值 | 说明 |
|------|-----|------|
| vocab_size | | 词表大小 |
| hidden_size | | 隐藏层维度 |
| num_layers | | 层数 |
| num_heads | | 注意力头数 |
| num_kv_heads | | KV 头数 (GQA) |
| num_experts | | 专家总数 |
| num_shared_experts | | 共享专家数 |
| num_activated_experts | | 激活的专家数 (Top-K) |

---

## Module 2: Data Pipeline & Tokenization (数据管道与分词)

**目标:** 准备高质量的预训练数据，并构建高效的数据加载管道。

> **数学背景关联**：本模块理解信息编码、熵的概念在分词中的应用

### 2.0 数据获取策略

> 数据是训练 LLM 的核心资源。本模块介绍三种主要的数据获取策略及其适用场景。

#### 2.0.1 开源数据集 (推荐首选)

**优点**：获取简单、质量有保障、社区验证丰富

**常用预训练数据集：**

| 数据集 | 规模 | 特点 | 链接 |
|--------|------|------|------|
| FineWeb-Edu | 100B+ tokens | 高质量教育数据，去除了低质量内容 | HuggingFace |
| SlimPajama | 627B tokens | 多来源清洗后的数据 | Cerebras |
| Dolma | 3T tokens | OpenAI 发布的开放数据集 | AI2 |
| The Stack | 3TB+ 代码 | 开源许可证的代码数据 | HuggingFace |

**常用 SFT 数据集：**

| 数据集 | 规模 | 特点 | 链接 |
|--------|------|------|------|
| UltraChat | 200k 对话 | 多轮对话，覆盖广泛主题 | HuggingFaceH4 |
| OpenOrca | 1M+ 对话 | GPT-4 合成的 CoT 数据 | OpenOrca |
| Magicoder | 350k 代码 | 代码生成指令数据 | ISCAS |

**常用 RL/推理数据集：**

| 数据集 | 规模 | 特点 | 链接 |
|--------|------|------|------|
| OpenR1-Math-220k | 220k 数学 | 数学推理训练数据 | HuggingFace |
| PRM800K | 800k 数学 | 过程奖励数据 | OpenAI |

#### 2.0.2 LLM 合成数据 (进阶选择)

**优点**：可定制性强、可以生成稀缺领域数据、避免版权问题

**适用场景**：
- 特定领域数据（如医学、法律、金融）
- 稀缺语言数据
- 指令遵循数据
- 代码注释/文档生成

**合成方法**：

1. **种子数据引导生成**
   ```
   提示词模板：
   请生成 10 条关于 {领域} 的问答对，要求：
   1. 问题涵盖 {子主题1}、{子主题2}、{子主题3}
   2. 答案详细、准确、符合事实
   3. 格式为 JSONL
   ```

2. **Self-Instruct 方法**
   - 使用少量人工编写的种子指令
   - 让 LLM 批量生成新指令
   - 通过过滤和筛选保证质量

3. **Evol-Instruct 方法**
   - 迭代式增强指令复杂度
   - 从简单指令逐步进化到复杂指令

**注意事项**：
- ⚠️ 合成数据可能引入模型偏见
- ⚠️ 需要强大的教师模型（如 GPT-4）才能保证质量
- ⚠️ 建议与真实数据混合使用

**工具推荐**：
- `distilabel`: HuggingFace 的数据合成库
- `argilla`: 数据标注和合成平台

#### 2.0.3 爬取与清洗 (可选，了解即可)

**优点**：可以获取最新信息、覆盖特定领域

**缺点**：工作量大、需要处理法律合规问题

**基本流程**：
1. 确定数据源（新闻网站、论坛、文档等）
2. 编写爬虫脚本（Scrapy、BeautifulSoup）
3. 清洗 HTML/JSON，提取纯文本
4. 去重、质量过滤
5. 存储到本地

**注意事项**：
- ⚠️ 遵守 robots.txt 和网站使用条款
- ⚠️ 注意版权问题
- ⚠️ 清洗工作量大，建议优先使用开源数据

**简单提及**：此方案需要较多人工投入，作为了解内容即可，实际项目中推荐使用前两种方案。

#### 数据获取策略对比

| 方案 | 成本 | 质量 | 规模 | 推荐度 |
|------|------|------|------|--------|
| 开源数据集 | 低 | 高 | 大 | ⭐⭐⭐⭐⭐ |
| LLM 合成 | 中 | 中 | 可控 | ⭐⭐⭐⭐ |
| 爬取清洗 | 高 | 低-中 | 可控 | ⭐⭐ |

#### 推荐的数据获取路径

**预训练阶段**：
1. FineWeb-Edu (10B-20B tokens 子集) → 性价比最高
2. SlimPajama → 如果需要更大规模

**SFT 阶段**：
1. UltraChat 200k → 通用对话
2. OpenOrca → 如果需要 CoT 能力

**RL 阶段**：
1. OpenR1-Math-220k → 数学推理

---

### 2.1 Tokenizer 集成 (`src/data/tokenizer.py`)

#### 任务清单

*   [ ] 选择并加载一个成熟的开源 Tokenizer
    *   推荐：Qwen2.5 或 Llama-3 的 tiktoken 实现
*   [ ] 编写包装类，处理特殊 token
    *   `<|endoftext|>`: 文本结束
    *   `<|im_start|>`: 对话开始
    *   `<|im_end|>`: 对话结束

#### 自检标准
- [ ] 输入一段包含多语言和代码的文本，验证 encode 和 decode 后的文本是否一致

#### 推荐阅读
- 📖 tiktoken 官方文档
- 📖 SentencePiece 论文 (用于理解 BPE 原理)

---

### 2.2 预训练数据准备与打包

#### 任务清单

*   [ ] 下载 `HuggingFaceFW/fineweb-edu` 的一个子集 (如 10B tokens)
*   [ ] 编写脚本将文本数据 tokenize
*   [ ] 拼接成固定长度 (如 seq_len = 2048) 的 chunks
*   [ ] 保存为高效的二进制格式 (.bin 或 .arrow)

#### 自检标准
- [ ] 读取生成的 .bin 文件，解码前几个 chunk，人工检查文本是否连贯
- [ ] 正确处理了文档边界 (EOD token)

#### 归档记录
- [ ] 在 `docs/training_log.md` 中记录：
    - 使用的数据集版本
    - 数据量 (Tokens 数)
    - 预处理耗时

---

### 2.3 PyTorch Dataset 与 DataLoader

#### 任务清单

*   [ ] 实现 `PretrainDataset` 类
    *   支持内存映射 (memory-mapped) 数据读取
*   [ ] 配置 DataLoader
    *   num_workers 设置
    *   prefetch_factor 设置

#### 自检标准
- [ ] 遍历 DataLoader 几个 epoch，测试数据加载的吞吐量 (Tokens/second)

---

## Module 3: Mini Pre-training (微型预训练)

**目标:** 跑通分布式训练框架，让模型学会基本的语言建模 (Next-Token Prediction)。

> **数学背景关联**：本模块重点理解随机优化、梯度下降的收敛性、学习率调度

### 3.1 训练循环与优化器

#### 任务清单

*   [ ] 实现核心的训练循环
    *   Forward → Loss 计算 → Backward → Optimizer Step
*   [ ] 集成 AdamW 优化器
*   [ ] 实现带有 Warmup 的 Cosine 学习率衰减
*   [ ] 实现梯度裁剪 (Gradient Clipping)
*   [ ] 实现 Checkpoint 的定期保存与恢复

#### 数学背景

**Cosine Annealing 学习率调度：**
```
lr(t) = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(π * t / T))
```
其中 T 为总步数，t 为当前步数

**梯度裁剪：**
```
if ||g|| > max_norm:
    g = g * max_norm / ||g||
```
防止梯度爆炸

#### 推荐阅读
- 📖 Adam 论文 (Kingma & Ba, 2014)
- 📖 SGDR 论文 (Cosine Annealing)

#### 自检标准
- [ ] 编写过拟合测试：使用极小数据集训练，Loss 应能快速降到接近 0

---

### 3.2 分布式训练集成

#### 任务清单

*   [ ] (如果有多卡) 集成 PyTorch FSDP 或 DeepSpeed Zero-2/3
*   [ ] 配置 Wandb 或 TensorBoard 进行实验监控

#### 自检标准
- [ ] 启动多卡训练，验证显存占用是否符合预期
- [ ] 各个 GPU 的负载是否均衡

---

### 3.3 启动微型预训练

#### 任务清单

*   [ ] 编写 `configs/pretrain_7b_moe.yaml`
*   [ ] 在准备好的 10B token 数据集上启动训练
*   [ ] 监控 Loss 曲线

#### 自检标准
- [ ] Loss 平稳下降
- [ ] 训练结束后生成文本，检查模型是否学会基本语法

#### 归档记录
- [ ] 在 `docs/training_log.md` 中记录：
    - 预训练超参数
    - 最终 Loss
    - 训练时长
    - 生成的 sample 文本

---

## Module 4: Supervised Fine-Tuning (指令微调 SFT)

**目标:** 让预训练模型学会遵循指令，适应对话格式。

### 4.1 SFT 数据格式化

#### 任务清单

*   [ ] 下载指令数据集 (如 `HuggingFaceH4/ultrachat_200k`)
*   [ ] 格式化为 ChatML 格式
    ```
    <|im_start|>user
    你好，请介绍一下自己<|im_end|>
    <|im_start|>assistant
    你好！我是一个AI助手...<|im_end|>
    ```
*   [ ] 实现 Loss Mask：只对 Assistant 回复部分计算 Loss

#### 自检标准
- [ ] 打印格式化后的样本，人工核对 Mask 是否正确

---

### 4.2 SFT 训练执行

#### 任务清单

*   [ ] (可选) 集成 `peft` 库实现 LoRA/QLoRA
*   [ ] 编写 `configs/sft_7b_moe.yaml`
    *   注意：SFT 学习率通常比预训练小一个数量级
*   [ ] 启动 SFT 训练

#### 自检标准
- [ ] 交互式对话测试：输入 "你好"，检查模型是否正确回复
- [ ] 不会无限生成

#### 归档记录
- [ ] 在 `docs/training_log.md` 中记录：
    - SFT 数据集
    - 超参数
    - 典型对话测试结果

---

## Module 5: Alignment & Reasoning (对齐与推理强化 RLHF/GRPO)

**目标:** 激发模型的思维链 (CoT) 能力，复现类似 DeepSeek-R1 的推理特性。

> **数学背景关联**：本模块重点理解策略梯度、KL 散度、奖励建模

### 5.1 强化学习环境准备

#### 任务清单

*   [ ] 准备推理数据集 (如 `Open-R1/OpenR1-Math-220k`)
*   [ ] 选择算法：推荐 **GRPO** (Group Relative Policy Optimization)
*   [ ] 定义基于规则的奖励函数

#### 数学背景

**GRPO 目标函数：**
```
L(θ) = -E[(r - b) * log π_θ(a|o)]
```
其中：
- r: 奖励 (reward)
- b: 基线 (baseline，通常是组内平均)
- π_θ: 策略模型
- o: 提示 (prompt)
- a: 回答 (answer)

**KL 散度约束：**
```
L(θ) = -E[(r - b) * log π_θ(a|o)] - β * KL(π_θ || π_ref)
```
防止策略模型偏离参考模型太远

#### 推荐阅读
- 📖 "GRPO: Group Relative Policy Optimization"
- 📖 PPO 论文 (了解 RLHF 基础)

---

### 5.2 RL 训练执行

#### 任务清单

*   [ ] 使用 `trl` 库集成 GRPO 算法
*   [ ] 启动训练，监控输出长度变化

#### 自检标准
- [ ] 观察生成样本：模型是否开始使用 `<think>...</think>` 标签
- [ ] 测试数学题准确率是否提升

#### 归档记录
- [ ] 在 `docs/training_log.md` 中记录：
    - Reward 曲线
    - 输出长度变化曲线
    - "Aha moment" 生成案例

---

## 🏁 最终评估与总结

### 任务清单

*   [ ] 使用 `lm-evaluation-harness` 在标准 Benchmark 上评估
    *   MMLU: 多任务语言理解
    *   GSM8K: 数学推理
    *   HumanEval: 代码生成
*   [ ] 整理所有文档，完成项目复盘

### 预期成果

完成本计划后，您将：
1. ✅ 深入理解 Transformer 架构的数学原理
2. ✅ 掌握现代 LLM 的核心组件 (RoPE, GQA, SwiGLU, MoE)
3. ✅ 具备从零训练大语言模型的实际经验
4. ✅ 理解 RLHF/GRPO 的原理和实现

---

## 📋 学习时间估算

| 模块 | 建议时间 | 累计时间 |
|------|----------|----------|
| Pre-Module | 1-2 周 | 1-2 周 |
| Module 0 | 1-2 周 | 2-4 周 |
| Module 1 | 2-3 周 | 4-7 周 |
| Module 2 | 1 周 | 8 周 |
| Module 3 | 2-4 周 | 10-14 周 |
| Module 4 | 1-2 周 | 11-16 周 |
| Module 5 | 2-3 周 | 13-19 周 |

> **注意**：以上时间为估算，实际时间取决于您的背景和投入时间。

---

*最后更新：2026-03-04*
*版本：v2.0 (完善版)*