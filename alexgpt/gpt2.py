"""
OpenAI's GPT-2 small (124M, 2019), written with our own code, so it can go through OUR benchmarks
(eval_benchmarks.py) with exactly the same scoring as Supernova / Singularity.

Only the weights and the tokenizer files come from Hugging Face (downloaded once, ~550 MB, into the
Hugging Face cache). No `transformers` library: the model is ~60 lines below.

Differences from Supernova, so you can compare:
  GPT-2                              Supernova / Singularity
  learned position table (wpe)       RoPE
  LayerNorm (with bias)              RMSNorm
  GELU MLP, 768 -> 3072 -> 768        SwiGLU, 768 -> 2048 -> 768
  biases everywhere                  no biases
  vocabulary 50,257 (English bytes)  16,384 (English + Romanian)
"""
import json
import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = "openai-community/gpt2"


# ------------------------------------------------------------------ tokenizer (byte-level BPE, like OpenAI's encoder.py)
def bytes_to_unicode():
    """GPT-2 maps every byte 0..255 to a printable character, so a byte string becomes a normal string."""
    printable = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    characters = printable[:]
    extra = 0
    for byte in range(256):
        if byte not in printable:
            printable.append(byte)
            characters.append(256 + extra)
            extra += 1
    return {byte: chr(character) for byte, character in zip(printable, characters)}


# GPT-2 splits text into pieces first: 's 't 're ... / " word" / " 123" / " ?!." / spaces.
# (OpenAI uses \p{L} and \p{N}; [^\W\d_] = letters and \d = digits are the same thing in Python's re.)
SPLIT_PATTERN = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+""")


class GPT2Tokenizer:
    def __init__(self, vocab_path, merges_path):
        with open(vocab_path, encoding="utf-8") as file:
            self.piece_id = json.load(file)                     # "Ġthe" -> 262
        with open(merges_path, encoding="utf-8") as file:
            lines = file.read().split("\n")[1:]                 # first line is a version comment
        self.rank = {tuple(line.split()): index for index, line in enumerate(lines) if line.strip()}
        self.byte_char = bytes_to_unicode()
        self.cache = {}
        self.end_id = self.piece_id["<|endoftext|>"]

    def encode_piece(self, piece):
        if piece in self.cache:
            return self.cache[piece]
        parts = list(piece)
        while len(parts) >= 2:
            pairs = [(self.rank.get((parts[i], parts[i + 1]), math.inf), i) for i in range(len(parts) - 1)]
            best_rank, index = min(pairs)
            if best_rank == math.inf:
                break
            parts[index:index + 2] = [parts[index] + parts[index + 1]]
        ids = [self.piece_id[part] for part in parts]
        self.cache[piece] = ids
        return ids

    def encode(self, text, allow_special=False):
        ids = []
        for piece in SPLIT_PATTERN.findall(text):
            mapped = "".join(self.byte_char[byte] for byte in piece.encode("utf-8"))
            ids.extend(self.encode_piece(mapped))
        return ids


# ------------------------------------------------------------------ model
class GPT2Block(nn.Module):
    def __init__(self, size, heads):
        super().__init__()
        self.heads = heads
        self.ln_1 = nn.LayerNorm(size)
        self.c_attn = nn.Linear(size, 3 * size)
        self.attn_proj = nn.Linear(size, size)
        self.ln_2 = nn.LayerNorm(size)
        self.c_fc = nn.Linear(size, 4 * size)
        self.mlp_proj = nn.Linear(4 * size, size)

    def forward(self, x):
        batch, length, size = x.shape
        q, k, v = self.c_attn(self.ln_1(x)).split(size, dim=2)
        q, k, v = (t.view(batch, length, self.heads, size // self.heads).transpose(1, 2) for t in (q, k, v))
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.attn_proj(attended.transpose(1, 2).reshape(batch, length, size))
        x = x + self.mlp_proj(F.gelu(self.c_fc(self.ln_2(x)), approximate="tanh"))
        return x


class GPT2(nn.Module):
    def __init__(self, vocab_size=50257, context_size=1024, size=768, blocks=12, heads=12):
        super().__init__()
        self.context_size = context_size
        self.wte = nn.Embedding(vocab_size, size)
        self.wpe = nn.Embedding(context_size, size)
        self.h = nn.ModuleList([GPT2Block(size, heads) for _ in range(blocks)])
        self.ln_f = nn.LayerNorm(size)

    def forward(self, ids, targets=None):
        positions = torch.arange(ids.size(1), device=ids.device)
        x = self.wte(ids) + self.wpe(positions)
        for block in self.h:
            x = block(x)
        logits = self.ln_f(x) @ self.wte.weight.T             # output = the token table again (tied)
        return logits, None


def load_gpt2(device="cpu"):
    """-> (model, tokenizer). Downloads the official weights once."""
    from huggingface_hub import hf_hub_download
    weights_path = hf_hub_download(REPO, "pytorch_model.bin")
    tokenizer = GPT2Tokenizer(hf_hub_download(REPO, "vocab.json"), hf_hub_download(REPO, "merges.txt"))
    official = torch.load(weights_path, map_location="cpu", weights_only=True)
    official = {name.removeprefix("transformer."): value for name, value in official.items()}
    model = GPT2()
    ours = {}
    # OpenAI stored the linear layers as "Conv1D": weight [in, out] -> nn.Linear wants [out, in] -> transpose
    renames = {"attn.c_attn": "c_attn", "attn.c_proj": "attn_proj", "mlp.c_fc": "c_fc", "mlp.c_proj": "mlp_proj"}
    for name, value in official.items():
        if name.endswith(".attn.bias") or name.endswith(".attn.masked_bias") or name.startswith("lm_head"):
            continue                                            # old causal-mask buffers, not weights
        new_name = name
        for old, new in renames.items():
            if f".{old}." in name:
                new_name = name.replace(f".{old}.", f".{new}.")
                if name.endswith(".weight"):
                    value = value.T.contiguous()
        ours[new_name] = value
    model.load_state_dict(ours)
    return model.to(device).eval(), tokenizer
