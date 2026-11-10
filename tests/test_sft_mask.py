"""
Checks the SFT mask (alexgpt/supernova.py, segments=...):  python tests/test_sft_mask.py [checkpoint]
Two conversations packed in one row, with the mask, must give EXACTLY the logits each one gives alone.
Without the mask the 2nd one is different (it sees the 1st one): that difference is what the mask removes.
"""
import sys

import torch

sys.path.insert(0, ".")
from alexgpt.bpe_fast import FastBPE            # noqa: E402
from alexgpt.sft_format import build_example     # noqa: E402
from alexgpt.supernova import Supernova          # noqa: E402

checkpoint = sys.argv[1] if len(sys.argv) > 1 else "runs/supernova/final.pt"
saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
model = Supernova(**saved["config"])
model.load_state_dict(saved["weights"] if "weights" in saved else saved["model"])
model.eval()
tokenizer = FastBPE.load("checkpoints/supernova_tokenizer.json")

first, _ = build_example([{"role": "user", "content": "My sister is called Emily. Say hi to her."},
                          {"role": "assistant", "content": "Hi Emily!"}], tokenizer)
second, _ = build_example([{"role": "user", "content": "What is your name?"},
                           {"role": "assistant", "content": "I am Supernova."}], tokenizer)
packed = torch.tensor([first + second])
segments = torch.tensor([[0] * len(first) + [1] * len(second)])
with torch.no_grad():
    alone_first, _ = model(torch.tensor([first]))
    alone_second, _ = model(torch.tensor([second]))
    masked, _ = model(packed, segments=segments)
    unmasked, _ = model(packed)
masked_difference = max((masked[0, :len(first)] - alone_first[0]).abs().max().item(),
                        (masked[0, len(first):] - alone_second[0]).abs().max().item())
unmasked_difference = (unmasked[0, len(first):] - alone_second[0]).abs().max().item()
print(f"max logit difference, packed WITH mask vs alone:    {masked_difference:.6f}   (must be ~0)")
print(f"max logit difference, packed WITHOUT mask vs alone: {unmasked_difference:.6f}   (the leak the mask removes)")
assert masked_difference < 1e-3, "the mask does not isolate the conversations"
print("OK")
