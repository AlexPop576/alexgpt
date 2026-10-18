"""
Galaxy: the same Transformer as Pulsar, written in PyTorch so it runs on the GPU.

Pulsar (your NumPy code)              Galaxy (PyTorch)
------------------------              ---------------
Tensor(np.random.randn(...))          nn.Linear / nn.Embedding (create the weights for us)
matmul(x, W)                          x @ W   (same thing, on the GPU)
causal_softmax(...)                   scores.masked_fill(...) + F.softmax(...)
tanh                                  GELU (a smoother ReLU, what GPT uses)
loss.backward()                       loss.backward()   (PyTorch does what your backward_step did)
p.data -= lr * p.grad                 optimizer.step()  (AdamW: a smarter update)
New here: several attention heads, LayerNorm, dropout.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    """Several attention heads side by side. Each head has its own q, k, v (like your Pulsar head)."""

    def __init__(self, size, heads, dropout):
        super().__init__()
        self.heads = heads
        self.head_size = size // heads
        self.qkv = nn.Linear(size, 3 * size)        # Wq, Wk, Wv for all heads, in one matrix
        self.out = nn.Linear(size, size)            # mixes what the heads found
        self.dropout = nn.Dropout(dropout)
        self.last_weights = None                    # kept for the interface

    def forward(self, x):
        batch, length, size = x.shape                     # e.g. 48, 256, 512
        q, k, v = self.qkv(x).split(size, dim=2)          # each (batch, length, size)
        future = torch.triu(torch.ones(length, length, dtype=torch.bool, device=x.device), diagonal=1)

        results = []
        all_weights = []                                  # for the Visualisation (only when not training)
        for head in range(self.heads):
            start = head * self.head_size                                   # head 0 -> 0, head 1 -> 64, head 2 -> 128 ...
            end = (head+1) * self.head_size
            q_head = q[:, :, start:end]
            k_head = k[:, :, start:end]
            v_head = v[:, :, start:end]

            scores = q_head @ k_head.transpose(-2, -1) / self.head_size ** 0.5                               # q_head @ k_head transposed, divided by sqrt(64)
            scores = scores.masked_fill(future, float("-inf"))   # hide the future
            weights = torch.softmax(scores, dim=-1)                                 # softmax on every row
            if not self.training:
                all_weights.append(weights)
            weights = self.dropout(weights)
            results.append(weights @ v_head)                           # weights times the values

        if not self.training:
            self.last_weights = torch.stack(all_weights, dim=1).detach()   # (batch, heads, length, length)
        glued = torch.cat(results, dim=2)                 # 8 pieces of 64 side by side -> 512
        return self.out(glued)                                        # the layer that mixes the heads


class Block(nn.Module):
    """One Transformer block: LayerNorm -> attention -> add, LayerNorm -> MLP -> add."""

    def __init__(self, size, heads, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(size)
        self.attention = Attention(size, heads, dropout)
        self.norm2 = nn.LayerNorm(size)
        self.mlp = nn.Sequential(
            nn.Linear(size, 4 * size),
            nn.GELU(),
            nn.Linear(4 * size, size),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attention(self.norm1(x))       # residual: add the notes
        x = x + self.mlp(self.norm2(x))             # residual again
        return x


class Galaxy(nn.Module):

    def __init__(self, vocab_size, context_size=256, size=384, blocks=6, heads=6, dropout=0.1):
        super().__init__()
        self.config = {"vocab_size": vocab_size, "context_size": context_size, "size": size,
                       "blocks": blocks, "heads": heads, "dropout": dropout}
        self.context_size = context_size
        self.token_table = nn.Embedding(vocab_size, size)
        self.position_table = nn.Embedding(context_size, size)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList()
        for _ in range(blocks):
            self.blocks.append(Block(size, heads, dropout))
        self.norm = nn.LayerNorm(size)
        self.out = nn.Linear(size, vocab_size, bias=False)
        self.out.weight = self.token_table.weight   # weight tying: W_out = token_table.T (your idea!)
        self.apply(self.init_weights)

    @staticmethod
    def init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, ids, targets=None):
        batch, length = ids.shape
        positions = torch.arange(length, device=ids.device)
        x = self.token_table(ids) + self.position_table(positions)    # what + where
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        logits = self.out(self.norm(x))                                # matching layer
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def num_params(self):
        total = 0
        for name, parameter in self.named_parameters():
            if name != "out.weight":                # shared with token_table, count it once
                total += parameter.numel()
        return total
