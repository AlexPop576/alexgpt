import numpy as np
from alexgpt.chat import SingularityChat, generate_stream
model = SingularityChat("runs/singularity_sft4/final.pt")
rng = np.random.default_rng(0)
for question in ["cât face 12+30?", "salut!"]:
    keys, tokens, tool = set(), 0, 0
    for step in generate_stream(model, question, 2000, 0.7, rng, [], True):
        if step["type"] == "tool":
            tool += 1
        if step["type"] == "token":
            tokens += 1
            keys |= set(step)
    print(question, "| tools:", tool, "| tokens:", tokens, "| step keys:", sorted(keys))
    inside = step if False else None
