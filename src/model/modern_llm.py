import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# -----------------------------------------------------------------------------
# 现代 LLM 核心组件 (Llama-3 / Qwen-2.5 风格)
# -----------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    替代了传统的 LayerNorm，去除了均值计算，提升了计算效率。
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    预计算 RoPE (Rotary Position Embedding) 的频率张量。
    现代模型通常使用更大的 theta (如 Llama-3 的 500000.0) 来支持长上下文。
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    """
    将 RoPE 应用于 Query 和 Key 张量。
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

class SwiGLU(nn.Module):
    """
    SwiGLU 激活函数 (Swish-Gated Linear Unit)。
    现代 LLM 的标配 FFN 结构，通常包含三个线性层 (Gate, Up, Down)。
    """
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False) # Gate
        self.w2 = nn.Linear(hidden_dim, dim, bias=False) # Down
        self.w3 = nn.Linear(dim, hidden_dim, bias=False) # Up

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class Attention(nn.Module):
    """
    Grouped-Query Attention (GQA)。
    当 n_kv_heads == n_heads 时，退化为标准的 Multi-Head Attention (MHA)。
    当 n_kv_heads == 1 时，退化为 Multi-Query Attention (MQA)。
    """
    def __init__(self, args):
        super().__init__()
        self.n_kv_heads = args.n_kv_heads
        self.n_heads = args.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: torch.Tensor):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        # 应用 RoPE
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # GQA: 复制 KV heads 以匹配 Q heads 的数量
        xk = torch.repeat_interleave(xk, dim=2, repeats=self.n_rep)
        xv = torch.repeat_interleave(xv, dim=2, repeats=self.n_rep)

        # 转置以进行注意力计算: (bsz, n_heads, seqlen, head_dim)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # Flash Attention (在实际训练中应使用 F.scaled_dot_product_attention 或 flash-attn 库)
        output = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=mask, is_causal=True if mask is None else False)
        
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)

# -----------------------------------------------------------------------------
# 接下来你可以尝试实现 MoE 层 (Mixture of Experts) 来替换 SwiGLU 层！
# -----------------------------------------------------------------------------
