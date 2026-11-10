"""A small mix for a short extra round on top of a chat model: the patch data + a random replay of the big mix
(so the model does not forget the rest).   python build_patch_mix.py  ->  data/sft_mix/singularity_patch.jsonl"""
import json
import random

from alexgpt.sft_format import system_prompt

rng = random.Random(41)
rows = []
for line in open("data/sft/gen_patch.jsonl", encoding="utf-8"):
    messages = json.loads(line)["messages"]
    if rng.random() < 0.3:
        messages = [{"role": "system", "content": system_prompt("Singularity")}] + messages
    rows.append({"source": "gen_patch", "repeat": 2, "messages": messages})
big = [json.loads(line) for line in open("data/sft_mix/singularity.jsonl", encoding="utf-8")]
rows += rng.sample(big, 6000)            # replay: tools, facts, identity, everything, in the usual proportions
rng.shuffle(rows)
with open("data/sft_mix/singularity_patch.jsonl", "w", encoding="utf-8") as file:
    for row in rows:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")
print(len(rows), "conversations -> data/sft_mix/singularity_patch.jsonl")
