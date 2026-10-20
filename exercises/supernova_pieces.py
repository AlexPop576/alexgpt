"""
Exercises while Supernova trains:  python exercises/supernova_pieces.py
Write the 3 new pieces of Supernova yourself. The training uses the finished versions in
alexgpt/supernova.py, so nothing breaks while you work here. The tests at the bottom compare your
functions with those, number by number. Do them in order; each one is a few lines.
"""
import sys
import os
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from alexgpt import supernova as real


# ------------------------------------------------------------------ 1. RMSNorm
def rms_norm(x, weight, eps=1e-6):
    """
    x: (..., size). For every vector (the last dimension): divide it by sqrt(mean of its squares + eps),
    then multiply by weight (a vector of `size` numbers).
    Example: x = [3, 4] -> squares [9, 16] -> mean 12.5 -> sqrt 3.536 -> [0.849, 1.131]
    Tools: x.pow(2)   .mean(dim=-1, keepdim=True)   torch.sqrt(...)
    """
    raise NotImplementedError


# ------------------------------------------------------------------ 2. RoPE
def rope(x, cos, sin):
    """
    x: (batch, heads, length, head_size). Take the numbers two by two: (x0, x1), (x2, x3), ...
    and rotate every pair by its angle:   (a, b) -> (a·cos - b·sin,  a·sin + b·cos)
    cos, sin: (length, head_size / 2), one value for every (position, pair) - already computed for you.
    Example: pair (1, 0) rotated by 90° (cos 0, sin 1) -> (0, 1).
    Tools: x[..., 0::2] = the first number of every pair, x[..., 1::2] = the second.
           torch.stack((first, second), dim=-1).flatten(-2) puts them back as pairs.
    """
    raise NotImplementedError


# ------------------------------------------------------------------ 3. SwiGLU
def swiglu(x, gate_weight, up_weight, down_weight):
    """
    x: (..., size). Three matrices (nn.Linear weights, so multiply with their transpose: x @ W.T):
        gate = x @ gate_weight.T          (..., hidden)
        up   = x @ up_weight.T            (..., hidden)
        out  = (silu(gate) * up) @ down_weight.T     (..., size)
    silu(z) = z * sigmoid(z): ~0 for negative z, ~z for positive z. It decides how much of `up` passes.
    Tools: F.silu(z)   (or z * torch.sigmoid(z))
    """
    raise NotImplementedError


# ------------------------------------------------------------------ tests
def check(name, function):
    try:
        ok = function()
    except NotImplementedError:
        print(f"{name:8s} not written yet")
        return False
    print(f"{name:8s} {'OK' if ok else 'WRONG'}")
    return ok


def test_rms_norm():
    small = rms_norm(torch.tensor([[3.0, 4.0]]), torch.ones(2))
    print("         [3, 4] ->", [round(value, 3) for value in small[0].tolist()], "(should be [0.849, 1.131])")
    norm = real.RMSNorm(16)
    norm.weight.data = torch.randn(16)
    x = torch.randn(2, 5, 16) * 30
    return torch.allclose(rms_norm(x, norm.weight), norm(x), atol=1e-4)


def test_rope():
    cos, sin = real.rope_tables(8, 32)
    x = torch.randn(2, 3, 32, 8)
    return torch.allclose(rope(x, cos, sin), real.apply_rope(x, cos, sin), atol=1e-5)


def test_swiglu():
    mlp = real.SwiGLU(32)
    x = torch.randn(2, 5, 32)
    mine = swiglu(x, mlp.gate.weight, mlp.up.weight, mlp.down.weight)
    return torch.allclose(mine, mlp(x), atol=1e-5)


if __name__ == "__main__":
    check("RMSNorm", test_rms_norm)
    check("RoPE", test_rope)
    check("SwiGLU", test_swiglu)
