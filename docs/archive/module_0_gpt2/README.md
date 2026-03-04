# Module 0: Foundation - GPT-2 Baseline (基石：彻底吃透 Transformer)

> **目标**：理解 Attention 机制的本质，手写一个极简的 GPT-2，作为后续现代架构的对比基线  
> **数学背景关联**：矩阵运算、Softmax 的数值稳定性

---

## 📚 推荐阅读

| 资源 | 链接 | 重点 |
|------|------|------|
| "Attention Is All You Need" | https://arxiv.org/abs/1706.03762 | Transformer 架构奠基之作 |
| nanoGPT 官方仓库 | https://github.com/karpathy/nanoGPT | 简洁的 GPT 实现参考 |
| The Illustrated Transformer | http://jalammar.github.io/illustrated-transformer/ | Transformer 可视化 |
| Andrej Karpathy: Let's build GPT | https://www.youtube.com/watch?v=kCc8FmEb1nY | 视频教程 |

---

## 0.1 极简 GPT-2 架构实现

*   **状态**: [ ] 未开始 / [ ] 进行中 / [ ] 已完成
*   **完成日期**: YYYY-MM-DD

### 核心数学公式

**Self-Attention 矩阵形式：**
```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

**多头注意力：**
```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

**LayerNorm：**
```
LayerNorm(x) = γ * (x - μ) / σ + β
其中 μ = mean(x), σ = std(x)
```

**GELU 激活函数：**
```
GELU(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
```

**绝对位置编码：**
```
PE(pos, 2i) = sin(pos/10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))
```

### 自检结果

| 组件 | 测试结果 | 备注 |
|------|----------|------|
| `Multi-Head Attention (MHA)` | (记录测试结果) | 验证因果掩码正确工作 |
| `LayerNorm` | (记录测试结果) | 验证 shape 正确 |
| `GELU` | (记录测试结果) | 验证数值范围 |
| `Absolute Position Embedding` | (记录测试结果) | 验证不同位置编码不同 |

### 关键决策/笔记

*   (例如：参考了 nanoGPT 的哪些设计？)
*   (为什么选择 GELU 而不是 ReLU？)

---

## 0.2 微型数据集训练测试

*   **状态**: [ ] 未开始 / [ ] 进行中 / [ ] 已完成
*   **完成日期**: YYYY-MM-DD

### 训练超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| batch_size | | |
| learning_rate | | |
| max_iters | | |
| weight_decay | | |
| warmup_steps | | |

### 自检结果

| 测试项 | 结果 | 备注 |
|--------|------|------|
| 莎士比亚数据集训练 Loss | (记录 Loss 曲线) | 是否从初始值下降到稳定值 |
| 生成文本测试 | (记录样例) | 模型是否能生成连贯的英文单词 |

### 生成的样本文本

```
1. (记录生成的文本样例 1)
2. (记录生成的文本样例 2)
3. (记录生成的文本样例 3)
```

### 关键决策/笔记

*   (记录训练过程中的调整)
*   (Loss 不下降时的排查思路)

---

## 📋 本模块学习检查清单

- [ ] 能手写 Multi-Head Attention 的 PyTorch 代码
- [ ] 理解为什么需要 √d_k 缩放因子
- [ ] 能解释 Attention 矩阵的计算复杂度：O(n²·d)
- [ ] 理解因果掩码的数学原理
- [ ] 能在 Shakespeare 数据集上训练并生成文本

---

*归档时间：YYYY-MM-DD*
*版本：v2.0*