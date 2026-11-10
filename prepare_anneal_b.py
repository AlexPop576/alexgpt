"""
Data for decay B (see train_singularity.py --decay-b):
    python prepare_anneal_b.py chat     conversations written by people -> C:/alexgpt_data/singularity/chat_human_*.bin
    python prepare_anneal_b.py gsm8k    grade-school math, solved step by step -> gsm8k_*.bin (needs internet)

chat: OpenAssistant (English only), and the Romanian NoRobots / Dolly / OpenAssistant (written by people,
translated automatically). Same filters as the SFT mix (no "as an AI language model", no other AI names, no
repeated sentences, answers longer than ~700 tokens cut). Every conversation is in the chat format:
    <|user|>question<|assistant|>answer<|endoftext|>...      (the markers as ids, the text with allow_special=False)
3% of the conversations go to *_val.bin: after the decay we compare A and B on them.
"""
import json
import os
import random
import re
import sys

import numpy as np

from alexgpt.bpe_fast import FastBPE
from alexgpt.sft_format import build_example
ARGUMENTS = sys.argv[1:]
sys.argv = sys.argv[:1]                       # build_sft_mix reads its own command-line options when imported
from build_sft_mix import cut_long_answers, problem   # noqa: E402

DATA_DIR = ARGUMENTS[1] if len(ARGUMENTS) > 1 else "C:/alexgpt_data/singularity"
CHAT_SOURCES = ["en_oasst", "ro_oasst", "ro_norobots", "ro_dolly"]
VAL_SHARE = 0.03


def write(name, conversations_ids):
    random.Random(0).shuffle(conversations_ids)
    val_count = int(len(conversations_ids) * VAL_SHARE)
    for part, items in (("val", conversations_ids[:val_count]), ("train", conversations_ids[val_count:])):
        flat = np.array([token for ids in items for token in ids], dtype=np.uint16)
        flat.tofile(os.path.join(DATA_DIR, f"{name}_{part}.bin"))
        print(f"  {name}_{part}.bin: {len(items):,} conversations, {len(flat) / 1e6:.1f}M tokens")


def chat(tokenizer):
    everything = []
    for source in CHAT_SOURCES:
        kept = thrown = 0
        with open(f"data/sft/{source}.jsonl", encoding="utf-8") as file:
            for line in file:
                messages = json.loads(line)["messages"]
                messages = [m for m in messages if m["content"].strip()]
                if problem(source, messages) or not messages:
                    thrown += 1
                    continue
                messages = cut_long_answers(messages)
                if not messages:
                    thrown += 1
                    continue
                ids, _ = build_example(messages, tokenizer)
                everything.append(ids)                    # build_example already ends every answer with <|endoftext|>
                kept += 1
        print(f"  {source:12s} kept {kept:,}  thrown out {thrown:,}")
    write("chat_human", everything)


def gsm8k(tokenizer):
    from datasets import load_dataset
    rows = load_dataset("openai/gsm8k", "main", split="train")
    everything = []
    for row in rows:
        steps = row["answer"].split("####")
        solution = steps[0].strip()
        solution = re.sub(r"<<[^>]*>>", "", solution)     # GSM8K's calculator notes: <<48/2=24>>
        final = steps[1].strip() if len(steps) > 1 else ""
        messages = [{"role": "user", "content": row["question"]},
                    {"role": "assistant", "content": f"{solution}\nThe answer is {final}."}]
        ids, _ = build_example(messages, tokenizer)
        everything.append(ids)
    write("gsm8k", everything)


if __name__ == "__main__":
    tokenizer = FastBPE.load("checkpoints/supernova_tokenizer.json")
    {"chat": chat, "gsm8k": gsm8k}[ARGUMENTS[0]](tokenizer)
