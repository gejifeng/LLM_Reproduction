# Module 1: Foundation & Architecture 归档记录

## 1. 核心组件实现 (1.1)
*   **状态**: [ ] 未开始 / [ ] 进行中 / [x] 已完成
*   **完成日期**: 2026-02-27
*   **自检结果**:
    *   `RMSNorm`: (记录测试结果，如 shape 是否正确)
    *   `RoPE`: (记录测试结果)
    *   `GQA`: (记录测试结果)
    *   `SwiGLU`: (记录测试结果)
*   **关键决策/笔记**:
    *   (例如：为什么选择 GQA 而不是 MHA？)

## 2. MoE 专家层与路由机制 (1.2)
*   **状态**: [ ] 未开始 / [ ] 进行中 / [ ] 已完成
*   **完成日期**: YYYY-MM-DD
*   **自检结果**:
    *   路由分布测试: (记录是否发生路由坍塌)
    *   激活参数量计算: (记录单次 forward 的激活参数量)
*   **关键决策/笔记**:
    *   (例如：选择了哪种路由策略？Shared Experts 的数量是多少？)

## 3. 模型组装与验证 (1.3)
*   **状态**: [ ] 未开始 / [ ] 进行中 / [ ] 已完成
*   **完成日期**: YYYY-MM-DD
*   **自检结果**:
    *   总参数量: (记录模型的总参数量)
    *   Forward Pass 测试: (记录输入 dummy data 后的输出 shape 是否正确)
*   **最终确定的模型超参数**:
    *   `vocab_size`: 
    *   `hidden_size`: 
    *   `num_layers`: 
    *   `num_heads`: 
    *   `num_kv_heads`: 
    *   `num_experts`: 
    *   `num_shared_experts`: 
    *   `num_activated_experts`: 
