# Module 1: Modernization & Architecture (现代化架构搭建)

> **目标**：实现现代 SOTA LLM 的核心组件，并组装成一个 7B 规模的 MoE 模型架构  
> **数学背景关联**：旋转矩阵、群论在 RoPE 中的应用、门控机制的数学原理

---

## 📚 推荐阅读

| 资源 | 链接 | 重点 |
|------|------|------|
| "RoFormer: Enhanced Position Embedding through Rotation" | https://arxiv.org/abs/2104.09864 | 旋转位置编码的数学原理 |
| "LLaMA: Open and Efficient Foundation Language Models" | https://arxiv.org/abs/2302.13971 | LLaMA 架构设计 |
| "Mixtral of Experts" | https://arxiv.org/abs/2401.04088 | MoE 实战经典案例 |
| "DeepSeek-V2" | https://arxiv.org/abs/2405.04434 | 细粒度 MoE 与负载均衡 |
| "Language Modeling with Gated Linear Networks" | https://arxiv.org/abs/1712.01897 | SwiGLU 的数学背景 |
| LLMVisualizer | https://bbycroft.net/llm | LLM 架构可视化 |
| RoPE Visualizer | https://github.com/graykode/rope-visualization | RoPE 原理可视化 |

---

## 1.1 核心组件实现

*   **状态**: [ ] 未开始 / [ ] 进行中 / [x] 已完成
*   **完成日期**: 2026-02-27

### 核心数学公式

**RMSNorm：**
```
RMSNorm(x) = x / RMS(x) * γ
其中 RMS(x) = √(mean(x²))
```

**RoPE (旋转位置编码)：**
```
RoPE(x_m) = x_m * [cos(mθ), sin(mθ), cos(mθ), sin(mθ), ...]
其中 θ_i = base^(-2i/d)
```
关键性质：相对位置编码、内积衰减

**SwiGLU：**
```
SwiGLU(x) = Swish(W_1 x) ⊗ (V x)
其中 Swish(x) = x * sigmoid(x)
```

**GQA (分组查询注意力)：**
- 多个 Query 头共享一个 Key/Value 头
- 减少 KV 缓存量的同时保持性能

### 自检结果

| 组件 | 测试结果 | 备注 |
|------|----------|------|
| `RMSNorm` | (记录测试结果，如 shape 是否正确) | 验证数值稳定性 |
| `RoPE` | (记录测试结果) | 验证相对位置编码 |
| `GQA` | (记录测试结果) | 验证 KV 缓存减少 |
| `SwiGLU` | (记录测试结果) | 验证梯度流动 |

### 关键决策/笔记

*   (例如：为什么选择 GQA 而不是 MHA？)
*   (为什么选择 SwiGLU 而不是 GELU？)

---

## 1.2 MoE 专家层与路由机制

*   **状态**: [ ] 未开始 / [ ] 进行中 / [ ] 已完成
*   **完成日期**: YYYY-MM-DD

### 核心数学公式

**路由机制：**
```
对于输入 x ∈ ℝ^d：
1. 计算门控分数：g = Softmax(W_gate x) ∈ ℝ^E (E 为专家数)
2. 选择 top-k 专家：indices = topk(g, k)
3. 计算输出：y = Σ_{i∈indices} g_i * Expert_i(x)
```

**负载均衡问题：**
- 理想情况：每个专家处理的 token 数大致相等
- 问题：可能发生路由坍塌 (routing collapse)

### 自检结果

| 测试项 | 结果 | 备注 |
|--------|------|------|
| 路由分布测试 | (记录是否发生路由坍塌) | 验证分布是否均匀 |
| 激活参数量计算 | (记录单次 forward 的激活参数量) | 如总参数 7B，激活 1.5B |
| Shared Experts 验证 | (记录是否对所有 token 激活) | 验证共享专家工作正常 |

### 关键决策/笔记

*   (选择了哪种路由策略？)
*   (Shared Experts 的数量是多少？)
*   (是否实现了 Auxiliary-Loss-Free Load Balancing？)

---

## 1.3 模型组装与验证

*   **状态**: [ ] 未开始 / [ ] 进行中 / [ ] 已完成
*   **完成日期**: YYYY-MM-DD

### 最终确定的模型超参数

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

### 自检结果

| 测试项 | 结果 | 备注 |
|--------|------|------|
| 总参数量 | (记录模型的总参数量) | 验证是否符合 7B 目标 |
| Forward Pass 测试 | (记录输入 dummy data 后的输出 shape) | 应为 (batch, seq_len, vocab_size) |
| 参数量计算 | (记录计算过程) | 验证激活参数量 |

### 关键决策/笔记

*   (模型架构选择的原因)
*   (权重初始化策略)

---

## 📋 本模块学习检查清单

- [ ] 理解 RoPE 的旋转矩阵数学原理
- [ ] 能解释 SwiGLU 相比 GELU 的优势
- [ ] 理解 GQA 如何减少 KV 缓存
- [ ] 能实现 Top-K 路由机制
- [ ] 理解负载均衡问题的数学本质
- [ ] 能组装完整的 MoE 模型并验证

---

*归档时间：YYYY-MM-DD*
*版本：v2.0*