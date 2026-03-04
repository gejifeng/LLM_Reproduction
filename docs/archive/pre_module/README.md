# Pre-Module: 预备知识模块

> **目标**：建立扎实的理论基础，熟悉开发环境，为后续模块做好准备  
> **建议学习时间**：1-2 周  
> **对象**：计算数学背景的硕士生 - 您已具备线性代数、概率论、优化理论的基础

---

## 📚 推荐学习资源

### 必读论文

| 论文 | 链接 | 核心收获 |
|------|------|----------|
| "Attention Is All You Need" | https://arxiv.org/abs/1706.03762 | Transformer 架构奠基之作 |
| "Language Modeling with Gated Linear Networks" | https://arxiv.org/abs/1712.01897 | 理解 SwiGLU 的数学背景 |
| Adam 论文 (Kingma & Ba, 2014) | https://arxiv.org/abs/1412.6980 | Adam 优化器数学原理 |

### 视频教程

| 资源 | 链接 | 适合阶段 |
|------|------|----------|
| 3Blue1Brown: Neural Networks | https://www.3blue1brown.com/topics/neural-networks | 直观理解神经网络 |
| Andrej Karpathy: Let's build GPT | https://www.youtube.com/watch?v=kCc8FmEb1nY | 手把手实现 GPT |

### 可视化理解

| 资源 | 链接 | 用途 |
|------|------|------|
| The Illustrated Transformer | http://jalammar.github.io/illustrated-transformer/ | Transformer 可视化 |
| LLMVisualizer | https://bbycroft.net/llm | LLM 架构可视化 |

---

## P.1 Transformer 数学基础

*   **状态**: [ ] 未开始 / [ ] 进行中 / [ ] 已完成
*   **完成日期**: YYYY-MM-DD

### 核心数学公式

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

### 自检标准

- [ ] 能手写 Multi-Head Attention 的 PyTorch 代码
- [ ] 理解为什么需要 √d_k 缩放因子（从数学角度解释）
- [ ] 能解释 Attention 矩阵的计算复杂度：O(n²·d)

### 关键笔记

* (记录学习过程中的疑问和理解)
* (记录数学推导的要点)

---

## P.2 神经网络训练基础

*   **状态**: [ ] 未开始 / [ ] 进行中 / [ ] 已完成
*   **完成日期**: YYYY-MM-DD

### 核心概念

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

### 自检标准

- [ ] 能解释 Cross-Entropy 的信息论含义
- [ ] 理解 AdamW 与 Adam 的区别
- [ ] 理解梯度裁剪 (Gradient Clipping) 的数学意义

### 关键笔记

* (记录优化器选择的考量)
* (记录正则化策略的理解)

---

## P.3 开发环境搭建

*   **状态**: [ ] 未开始 / [ ] 进行中 / [ ] 已完成
*   **完成日期**: YYYY-MM-DD

### 环境配置清单

| 任务 | 状态 | 备注 |
|------|------|------|
| 配置 Python 环境 (conda) | [ ] | |
| 安装 PyTorch (确认 CUDA 版本) | [ ] | |
| 安装基础库 (transformers, tiktoken, datasets) | [ ] | |
| 配置 GPU 环境验证 | [ ] | |
| 安装 Jupyter Notebook / 配置 VS Code | [ ] | |

### 自检标准

- [ ] 运行 `python -c "import torch; print(torch.cuda.is_available())"` 返回 True
- [ ] 能用 PyTorch 创建简单的 Tensor 并在 GPU 上运算

### 关键笔记

* (记录环境配置过程中的问题)
* (记录 CUDA 版本和 PyTorch 版本)

---

## 📋 预备模块学习检查清单

- [ ] 理解 Self-Attention 的数学原理
- [ ] 能手写 Multi-Head Attention 代码
- [ ] 理解 √d_k 缩放因子的数学意义
- [ ] 理解 Cross-Entropy 的信息论含义
- [ ] 理解 AdamW 优化器的数学原理
- [ ] 能配置开发环境并运行 GPU 代码

---

## 🎯 预备模块完成标准

完成本模块后，您应该：
1. ✅ 深入理解 Transformer 的数学原理
2. ✅ 掌握神经网络训练的基础概念
3. ✅ 配置好开发环境
4. ✅ 准备好进入 Module 0 的学习

---

*归档时间：YYYY-MM-DD*
*版本：v2.0*