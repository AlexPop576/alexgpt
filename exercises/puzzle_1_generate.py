"""
PUZZLE 1: from your message to the model's reply, by hand.
    python exercises/puzzle_1_generate.py

All the pieces are ready in alexgpt/pieces.py (open it next to this file: every piece says
what it gets, what it returns and where it is used). You only CONNECT them, in 4 functions,
from the inside out:

    attention()          q, k, v -> scores -> softmax -> mix of values
    transformer_block()  norm -> attention -> add, norm -> MLP -> add
    forward()            ids -> embedding -> 12 blocks -> norm -> logits
    generate()           text -> ids -> forward -> pick a token -> again ... -> text

The tests at the bottom run Supernova's real code on the same weights and compare it with yours,
number by number. Do them in order; the test tells you which function is wrong.
It uses runs/supernova/latest.pt, so it works while training runs (on the CPU, a few seconds).
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from alexgpt.pieces import (load_weights, block_weights, encode, decode, embed, linear, rms_norm,
                            split_heads, merge_heads, rope_tables, apply_rope, attention_scores,
                            hide_future, softmax, weighted_sum, swiglu, logits_from,
                            apply_temperature, pick_token)


# ---------------------------------------------------------------------------- 1. attention
def attention(x, block, heads, cos, sin):
    """
    x: (length, size), already normalized by norm1.  block: block_weights(...) of this block.
    Returns (length, size): what attention adds to every token.

    Steps (one or two lines each):
      1. all_three = linear(x, block["qkv"])              -> (length, 3*size)
         q, k, v = all_three.split(size, dim=1)            -> 3 x (length, size)   (size = x.shape[1])
      2. split q, k, v into heads                          -> (heads, length, head_size)
      3. QK-Norm: rms_norm on q with block["q_norm"], on k with block["k_norm"]
      4. RoPE: apply_rope on q and on k (not on v)
      5. scores, then hide the future, then softmax       -> weights (heads, length, length)
      6. weighted sum of v, merge the heads                -> (length, size)
      7. linear with block["attention_out"]                (mixes what the heads found)
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------- 2. one block
def transformer_block(x, block, heads, cos, sin):
    """
    x: (length, size) -> (length, size).  The residual stream: each part ADDS to x, never replaces it.
      1. x = x + attention( rms_norm(x, block["norm1"]), ... )
      2. x = x + swiglu( rms_norm(x, block["norm2"]), block["gate"], block["up"], block["down"] )
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------- 3. the whole model
def forward(ids, weights, config):
    """
    ids: list of token ids -> logits (length, vocab): a score for every possible next token, at every position.
      1. ids -> tensor:  torch.tensor(ids)
      2. x = embed(...) with weights["token_table.weight"]
      3. cos, sin = rope_tables(head_size, length)     head_size = config["size"] // config["heads"]
      4. for index in range(config["blocks"]): x = transformer_block(x, block_weights(weights, index), ...)
      5. final rms_norm with weights["norm.weight"]
      6. logits_from(...) with weights["token_table.weight"] (the same table as the embedding)
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------- 4. generation
def generate(text, weights, config, tokenizer, new_tokens=30, temperature=0.8, generator=None):
    """
    text -> text + what the model writes after it.  The loop:
      1. ids = encode(...)
      2. repeat new_tokens times:
           logits = forward(ids[-context:], ...)       context = config["context_size"] (1024)
           last = logits[-1]                            only the LAST position predicts the next token
           probabilities = softmax(apply_temperature(last, temperature))
           next_id = pick_token(probabilities, generator)
           stop if next_id == tokenizer.end_id, otherwise ids.append(next_id)
      3. return decode(...)
    """
    raise NotImplementedError


# ============================================================================ tests
def main():
    from alexgpt.bpe_fast import FastBPE
    from alexgpt.supernova import Supernova
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    weights, config = load_weights(os.path.join(root, "runs", "supernova", "latest.pt"))
    tokenizer = FastBPE.load(os.path.join(root, "checkpoints", "supernova_tokenizer.json"))
    real = Supernova(**config)
    real.load_state_dict({name: tensor for name, tensor in weights.items()}, strict=False)
    real.eval()
    heads = config["heads"]
    ids = encode(tokenizer, "Cluj-Napoca is a city in")
    torch.manual_seed(0)
    x = torch.randn(len(ids), config["size"])
    cos, sin = rope_tables(config["size"] // heads, len(ids))

    def check(name, run, expected):
        try:
            with torch.no_grad():
                mine = run()
        except NotImplementedError:
            print(f"  {name:18s} not written yet")
            return False
        except Exception as error:
            print(f"  {name:18s} ERROR  {type(error).__name__}: {error}")
            return False
        with torch.no_grad():
            reference = expected()
        if not isinstance(mine, torch.Tensor) or mine.shape != reference.shape:
            shape = tuple(mine.shape) if isinstance(mine, torch.Tensor) else type(mine).__name__
            print(f"  {name:18s} WRONG SHAPE  yours {shape}, should be {tuple(reference.shape)}")
            return False
        difference = (mine - reference).abs().max().item()
        ok = difference < 1e-3
        print(f"  {name:18s} {'OK' if ok else 'WRONG'}   (biggest difference {difference:.2e})")
        return ok

    block = block_weights(weights, 0)
    print("comparing with Supernova's real code (same weights):")
    good = check("attention", lambda: attention(x, block, heads, cos, sin),
                 lambda: real.blocks[0].attention(x.unsqueeze(0), real.rope_cos, real.rope_sin)[0])
    good = good and check("transformer_block", lambda: transformer_block(x, block, heads, cos, sin),
                          lambda: real.blocks[0](x.unsqueeze(0), real.rope_cos, real.rope_sin)[0])
    good = good and check("forward", lambda: forward(ids, weights, config),
                          lambda: real(torch.tensor([ids]))[0][0])
    if not good:
        return
    print("\nyour generate():")
    generator = torch.Generator().manual_seed(1)
    try:
        with torch.no_grad():
            print("  " + generate("Cluj-Napoca is a city in", weights, config, tokenizer, 30, 0.7, generator))
    except NotImplementedError:
        print("  not written yet")


if __name__ == "__main__":
    main()
