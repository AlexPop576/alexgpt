"""Checks every data/sft/gen_*.jsonl: valid roles/order, fits in 1024 tokens, no special-token text, token counts.
    python check_gen_data.py"""
import glob
import json

from alexgpt.bpe_fast import FastBPE
from alexgpt.sft_format import build_example

CONTEXT = 1024
tokenizer = FastBPE.load("checkpoints/supernova_tokenizer.json")
ORDER_OK = {"system": {"user"}, "user": {"assistant", "tool_call"}, "tool_call": {"context"}, "context": {"assistant"}, "assistant": {"user"}}
problems_total = 0
for path in sorted(glob.glob("data/sft/gen_*.jsonl")):
    count = tokens = trained = too_long = bad = 0
    for line in open(path, encoding="utf-8"):
        messages = json.loads(line)["messages"]
        count += 1
        roles = [m["role"] for m in messages]
        order_ok = (roles[0] in ("system", "user") and roles[-1] == "assistant"
                    and all(roles[i + 1] in ORDER_OK[roles[i]] for i in range(len(roles) - 1))
                    and all(m["content"].strip() for m in messages) and not any("<|" in m["content"] for m in messages))
        if not order_ok:
            bad += 1
            print("  BAD ORDER/EMPTY:", roles, messages[-1]["content"][:60])
            continue
        ids, mask = build_example(messages, tokenizer)
        tokens += len(ids)
        trained += sum(mask)
        too_long += len(ids) > CONTEXT
    problems_total += bad + too_long
    print(f"{path:32s} {count:6d} conversations  {tokens:9d} tokens  {100 * trained / max(tokens, 1):4.1f}% trained  {too_long} too long  {bad} bad")
print("ALL OK" if problems_total == 0 else f"{problems_total} PROBLEMS")
