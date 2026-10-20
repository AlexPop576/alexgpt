"""
The puzzle pieces: every step of Supernova / Singularity as a plain function.
No nn.Module, no hidden magic: each function gets tensors, returns tensors.
You connect them in exercises/puzzle_*.py.

Shapes use these names:
    length      how many tokens are in the text right now (e.g. 7)
    size        the vector of one token (768)
    heads       attention heads (12), head_size = size / heads = 64
    hidden      the MLP's hidden layer (2048)
    vocab       how many different tokens exist (16384)

All weights come from the checkpoint as a dictionary, e.g.
    weights["blocks.3.attention.qkv.weight"]   the q, k, v matrices of block 3
Use block_weights(weights, 3) to get only the ones of block 3, with short names.
"""
import math
import torch


# ====================================================================== loading

def load_weights(checkpoint_path):
    """runs/supernova/latest.pt -> (weights dictionary, config dictionary).
    config = {"vocab_size": 16384, "context_size": 1024, "size": 768, "blocks": 12, "heads": 12}"""
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    weights = {}
    for name, tensor in saved["model"].items():
        weights[name] = tensor.float()
    return weights, saved["config"]


def block_weights(weights, index):
    """The weights of one block, with short names:
        norm1, qkv, q_norm, k_norm, attention_out, norm2, gate, up, down"""
    prefix = f"blocks.{index}."
    return {
        "norm1": weights[prefix + "norm1.weight"],                  # (size)
        "qkv": weights[prefix + "attention.qkv.weight"],            # (3*size, size)
        "q_norm": weights[prefix + "attention.q_norm.weight"],      # (head_size)
        "k_norm": weights[prefix + "attention.k_norm.weight"],      # (head_size)
        "attention_out": weights[prefix + "attention.out.weight"],  # (size, size)
        "norm2": weights[prefix + "norm2.weight"],                  # (size)
        "gate": weights[prefix + "mlp.gate.weight"],                # (hidden, size)
        "up": weights[prefix + "mlp.up.weight"],                    # (hidden, size)
        "down": weights[prefix + "mlp.down.weight"],                # (size, hidden)
    }


# ====================================================================== text <-> numbers

def encode(tokenizer, text):
    """"Salut!" -> [3011, 0]  (a list of token ids). Used first, on your message."""
    return tokenizer.encode(text)


def decode(tokenizer, ids):
    """[3011, 0] -> "Salut!". Used last, on the generated ids."""
    return tokenizer.decode(ids)


# ====================================================================== the pieces of the model

def embed(ids, token_table):
    """ids (length) -> x (length, size): row number id of the table, for every token.
    Example: ids [5, 2] -> [token_table[5], token_table[2]].
    Used once, at the start of the forward pass."""
    return token_table[ids]


def linear(x, weight):
    """x (..., in) -> (..., out).  weight has shape (out, in), as PyTorch stores it, so x @ weight.T.
    Example: x [1, 2], weight [[1, 0], [1, 1], [0, 3]] -> [1*1+2*0, 1*1+2*1, 1*0+2*3] = [1, 3, 6].
    Used everywhere a matrix multiplies a vector: qkv, attention_out, gate, up, down, and the logits."""
    return x @ weight.T


def rms_norm(x, weight, eps=1e-6):
    """Every vector (last dimension) divided by sqrt(mean of its squares), then times weight.
    Example: [3, 4] -> squares [9, 16] -> mean 12.5 -> sqrt 3.54 -> [0.85, 1.13] (weight = 1).
    Used: before attention (norm1), before the MLP (norm2), on q and k (q_norm, k_norm), at the end (norm)."""
    mean_square = x.pow(2).mean(dim=-1, keepdim=True)
    return x / torch.sqrt(mean_square + eps) * weight


def split_heads(x, heads):
    """(length, size) -> (heads, length, head_size): every head gets its own slice of 64 numbers.
    Example with size 4, heads 2: token [a, b, c, d] -> head 0 gets [a, b], head 1 gets [c, d].
    Used on q, k and v inside attention."""
    length, size = x.shape
    return x.view(length, heads, size // heads).transpose(0, 1)


def merge_heads(x):
    """(heads, length, head_size) -> (length, size): the heads' results glued back side by side.
    The opposite of split_heads. Used at the end of attention, before attention_out."""
    heads, length, head_size = x.shape
    return x.transpose(0, 1).reshape(length, heads * head_size)


def rope_tables(head_size, length, base=10000.0):
    """cos and sin, (length, head_size/2): the angle for every (position, pair of numbers).
    Pair 0 turns fast (1 radian per position), the last pairs turn very slowly.
    Used once per forward pass, then given to every block."""
    speeds = 1.0 / (base ** (torch.arange(0, head_size, 2).float() / head_size))
    positions = torch.arange(length).float()
    angles = torch.outer(positions, speeds)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x, cos, sin):
    """x (heads, length, head_size): every pair (a, b) turned by its angle:
    (a, b) -> (a*cos - b*sin, a*sin + b*cos).  Example: (1, 0) turned by 90 degrees -> (0, 1).
    Used on q and k (not on v), so q·k depends on how FAR apart two tokens are."""
    first, second = x[..., 0::2], x[..., 1::2]
    turned_first = first * cos - second * sin
    turned_second = first * sin + second * cos
    return torch.stack((turned_first, turned_second), dim=-1).flatten(-2)


def attention_scores(q, k):
    """q, k (heads, length, head_size) -> scores (heads, length, length):
    scores[h, i, j] = how well token i's question matches token j's key = q_i · k_j / sqrt(head_size).
    Dividing by sqrt(64) = 8 keeps the numbers small, so softmax doesn't become all-or-nothing."""
    head_size = q.shape[-1]
    return q @ k.transpose(-2, -1) / math.sqrt(head_size)


def hide_future(scores):
    """Token i may not look at tokens after it: those scores become -infinity (softmax turns them into 0).
    Example for 3 tokens, row by row:  [s, -inf, -inf] / [s, s, -inf] / [s, s, s]."""
    length = scores.shape[-1]
    future = torch.triu(torch.ones(length, length, dtype=torch.bool), diagonal=1)
    return scores.masked_fill(future, float("-inf"))


def softmax(scores):
    """Every row -> chances that add up to 1: exp(score) / sum of exp.
    Example: [2, 1, 0] -> exp [7.39, 2.72, 1] -> / 11.1 -> [0.67, 0.24, 0.09].
    The max is subtracted first (same result, but exp never overflows)."""
    shifted = scores - scores.max(dim=-1, keepdim=True).values
    exps = torch.exp(shifted)
    return exps / exps.sum(dim=-1, keepdim=True)


def weighted_sum(weights, v):
    """weights (heads, length, length) @ v (heads, length, head_size) -> (heads, length, head_size):
    every token gets a mix of the values of the tokens it looks at.
    Example: weights [0.7, 0.3] on values [10, 20] -> 0.7*10 + 0.3*20 = 13."""
    return weights @ v


def silu(z):
    """z * sigmoid(z): ~0 for negative z, ~z for positive z.  silu(3) = 2.86, silu(-3) = -0.14.
    Used inside swiglu, on the gate."""
    return z * torch.sigmoid(z)


def swiglu(x, gate_weight, up_weight, down_weight):
    """The MLP of one block: x (length, size) -> (length, size).
        gate = x @ gate.T, up = x @ up.T          (length, hidden)   two paths
        out  = (silu(gate) * up) @ down.T         (length, size)     the gate decides how much of up passes"""
    gate = linear(x, gate_weight)
    up = linear(x, up_weight)
    return linear(silu(gate) * up, down_weight)


# ====================================================================== after the model

def logits_from(x, token_table):
    """x (length, size) -> scores (length, vocab): x compared with EVERY token's vector (dot product).
    The same table as embed (weight tying). Used once, after the final rms_norm."""
    return linear(x, token_table)


def apply_temperature(logits, temperature):
    """Divide the scores by the temperature. 0.5 -> the gaps double (more sure), 1.5 -> smaller gaps (more random)."""
    return logits / temperature


def pick_token(probabilities, generator=None):
    """Roll the dice: returns one token id, each with its chance. probabilities (vocab) -> int"""
    return int(torch.multinomial(probabilities, 1, generator=generator))
