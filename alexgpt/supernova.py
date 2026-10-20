"""
Supernova: the Galaxy transformer, upgraded to the modern (LLaMA-style) pieces.

What changed compared to Galaxy, and why:
  LayerNorm            -> RMSNorm      only rescales the vector to size ~1 (no mean subtraction, no bias): simpler, as good
  position table       -> RoPE         q and k are ROTATED by an angle that depends on the position, so the
                                       attention score depends on the DISTANCE between two tokens
  GELU MLP             -> SwiGLU       two parallel projections; one of them decides how much of the other passes
  loop over heads      -> F.scaled_dot_product_attention   the same math as your loop, as one fast GPU kernel
  (new)                   QK-Norm      RMSNorm on q and k before the dot product: scores can't explode -> stabler training
  biases               -> none         fewer parameters, no loss in quality
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """x / sqrt(mean(x^2)) * weight.  Example: [3, 4] -> mean of squares 12.5 -> sqrt 3.54 -> [0.85, 1.13]."""

    def __init__(self, size, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(size))

    def forward(self, x):
        mean_square = x.float().pow(2).mean(dim=-1, keepdim=True)
        return (x.float() * torch.rsqrt(mean_square + self.eps)).type_as(x) * self.weight


def rope_tables(head_size, max_length, base=10000.0):
    """cos and sin for every (position, pair of numbers). Pair i turns slower as i grows:
    the first pairs measure short distances, the last pairs long ones."""
    speeds = 1.0 / (base ** (torch.arange(0, head_size, 2).float() / head_size))   # head_size/2 speeds
    positions = torch.arange(max_length).float()
    angles = torch.outer(positions, speeds)                                          # (length, head_size/2)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x, cos, sin):
    """Rotate every pair (a, b) of x by its angle: (a, b) -> (a·cos - b·sin, a·sin + b·cos).
    x: (batch, heads, length, head_size)."""
    first, second = x[..., 0::2], x[..., 1::2]
    rotated_first = first * cos - second * sin
    rotated_second = first * sin + second * cos
    return torch.stack((rotated_first, rotated_second), dim=-1).flatten(-2)


class Attention(nn.Module):

    def __init__(self, size, heads):
        super().__init__()
        self.heads = heads
        self.head_size = size // heads
        self.qkv = nn.Linear(size, 3 * size, bias=False)
        self.out = nn.Linear(size, size, bias=False)
        self.q_norm = RMSNorm(self.head_size)
        self.k_norm = RMSNorm(self.head_size)

    def forward(self, x, cos, sin):
        batch, length, size = x.shape
        q, k, v = self.qkv(x).split(size, dim=2)
        # (batch, length, size) -> (batch, heads, length, head_size): every head gets its own slice, like your loop
        q = q.view(batch, length, self.heads, self.head_size).transpose(1, 2)
        k = k.view(batch, length, self.heads, self.head_size).transpose(1, 2)
        v = v.view(batch, length, self.heads, self.head_size).transpose(1, 2)
        q, k = self.q_norm(q), self.k_norm(k)
        q, k = apply_rope(q, cos[:length], sin[:length]), apply_rope(k, cos[:length], sin[:length])
        # softmax(q @ k.T / sqrt(head_size) with the future hidden) @ v, for all heads at once
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(batch, length, size)
        return self.out(out)


class SwiGLU(nn.Module):
    """out = down( silu(gate(x)) * up(x) ).  `gate` decides, number by number, how much of `up` passes."""

    def __init__(self, size):
        super().__init__()
        hidden = int(8 * size / 3)
        hidden = 64 * ((hidden + 63) // 64)            # a multiple of 64: faster on the GPU
        self.gate = nn.Linear(size, hidden, bias=False)
        self.up = nn.Linear(size, hidden, bias=False)
        self.down = nn.Linear(hidden, size, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):

    def __init__(self, size, heads):
        super().__init__()
        self.norm1 = RMSNorm(size)
        self.attention = Attention(size, heads)
        self.norm2 = RMSNorm(size)
        self.mlp = SwiGLU(size)

    def forward(self, x, cos, sin):
        x = x + self.attention(self.norm1(x), cos, sin)     # residual, like Galaxy
        x = x + self.mlp(self.norm2(x))
        return x


class Supernova(nn.Module):

    def __init__(self, vocab_size, context_size=1024, size=768, blocks=12, heads=12):
        super().__init__()
        self.config = {"vocab_size": vocab_size, "context_size": context_size, "size": size,
                       "blocks": blocks, "heads": heads}
        self.context_size = context_size
        self.token_table = nn.Embedding(vocab_size, size)       # no position table: RoPE does positions
        self.blocks = nn.ModuleList([Block(size, heads) for _ in range(blocks)])
        self.norm = RMSNorm(size)
        self.out = nn.Linear(size, vocab_size, bias=False)
        self.out.weight = self.token_table.weight              # weight tying, like Galaxy
        cos, sin = rope_tables(size // heads, context_size)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self.init_weights)
        # the layers that write into the residual stream start smaller, so 12 blocks adding up don't blow x up
        for name, parameter in self.named_parameters():
            if name.endswith("attention.out.weight") or name.endswith("mlp.down.weight"):
                nn.init.normal_(parameter, mean=0.0, std=0.02 / math.sqrt(2 * blocks))

    @staticmethod
    def init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, ids, targets=None):
        x = self.token_table(ids)
        for block in self.blocks:
            x = block(x, self.rope_cos, self.rope_sin)
        logits = self.out(self.norm(x))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), targets.reshape(-1))
        return logits, loss

    def num_params(self):
        total = 0
        for name, parameter in self.named_parameters():
            if name != "out.weight":
                total += parameter.numel()
        return total
