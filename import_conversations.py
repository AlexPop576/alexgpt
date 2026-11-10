"""Turns a text file written by another LLM (or by you) into SFT data.
    python import_conversations.py my_file.txt            -> data/sft/gen_ext_my_file.jsonl
Format of the text file (same as sft_data/identity.txt):
    ###
    USER: question                (a message can continue on the next lines, e.g. a code block)
    ASSISTANT: answer
    USER: ...
    ASSISTANT: ...
    ###
Conversations that break the rules (wrong order, empty, special tokens, over 1,024 tokens) are dropped and listed."""
import json
import os
import re
import sys

from alexgpt.bpe_fast import FastBPE
from alexgpt.sft_format import build_example

path = sys.argv[1]
name = os.path.splitext(os.path.basename(path))[0]
tokenizer = FastBPE.load("checkpoints/supernova_tokenizer.json")
conversations, current = [], None
for line in open(path, encoding="utf-8"):
    line = line.rstrip("\n")
    if line.strip() == "###":
        if current:
            conversations.append(current)
        current = []
        continue
    match = re.match(r"^(USER|ASSISTANT):\s?(.*)$", line)
    if current is None:
        continue
    if match:
        current.append({"role": match.group(1).lower(), "content": match.group(2)})
    elif current:
        current[-1]["content"] += "\n" + line
if current:
    conversations.append(current)

kept, dropped = [], 0
for messages in conversations:
    for message in messages:
        message["content"] = message["content"].strip()
    roles = [m["role"] for m in messages]
    good = (len(messages) >= 2 and roles[0] == "user" and roles[-1] == "assistant"
            and all(role == ("user" if i % 2 == 0 else "assistant") for i, role in enumerate(roles))
            and all(m["content"] and "<|" not in m["content"] for m in messages))
    if good and len(build_example(messages, tokenizer)[0]) > 1024:
        good = False
    if good:
        kept.append(messages)
    else:
        dropped += 1
        print("  dropped:", (messages[0]["content"][:70] if messages else "(empty)"))
out = f"data/sft/gen_ext_{name}.jsonl"
with open(out, "w", encoding="utf-8") as file:
    for messages in kept:
        file.write(json.dumps({"messages": messages, "source": "external:" + name}, ensure_ascii=False) + "\n")
print(f"{len(kept)} kept, {dropped} dropped -> {out}")
