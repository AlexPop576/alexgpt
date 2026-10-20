"""
Look at the chat data before training on it:
    python show_sft_data.py                 how many conversations each source has + one example of each
    python show_sft_data.py ro_dolly 5      5 random conversations from ro_dolly
"""
import json
import os
import random
import sys

from prepare_sft import SOURCES, OUT_DIR

WIDTH = 100


def load(source):
    path = os.path.join(OUT_DIR, f"{source}.jsonl")
    if not os.path.exists(path):
        return []
    conversations = []
    with open(path, encoding="utf-8") as file:
        for line in file:
            try:
                conversations.append(json.loads(line)["messages"])
            except (json.JSONDecodeError, KeyError):
                pass
    return conversations


def show(messages, limit=700):
    for message in messages:
        label = {"user": "USER     ", "assistant": "ASSISTANT", "system": "SYSTEM   "}[message["role"]]
        text = message["content"]
        if len(text) > limit:
            text = text[:limit] + f" … (+{len(message['content']) - limit} characters)"
        lines = text.split("\n")
        print(f"  {label} {lines[0]}")
        for line in lines[1:]:
            print(f"            {line}")
    print("  " + "-" * WIDTH)


def main():
    if len(sys.argv) > 1:
        source = sys.argv[1]
        count = int(sys.argv[2]) if len(sys.argv) > 2 else 3
        conversations = load(source)
        print(f"\n{source}: {SOURCES[source][4]}   ({len(conversations)} conversations)\n")
        for messages in random.sample(conversations, min(count, len(conversations))):
            show(messages, limit=3000)
        return
    total = 0
    print(f"\n{'source':17s} {'lang':4s} {'conversations':>13s} {'avg turns':>9s} {'avg chars':>9s}   what it teaches")
    examples = {}
    for source, (_, _, language, wanted, teaches) in SOURCES.items():
        conversations = load(source)
        total += len(conversations)
        if conversations:
            turns = sum(len(m) for m in conversations) / len(conversations)
            chars = sum(sum(len(x["content"]) for x in m) for m in conversations) / len(conversations)
            print(f"{source:17s} {language:4s} {len(conversations):13d} {turns:9.1f} {chars:9.0f}   {teaches}")
            examples[source] = random.choice(conversations)
        else:
            print(f"{source:17s} {language:4s} {'(none yet)':>13s}")
    print(f"{'TOTAL':17s}      {total:13d}\n")
    for source, messages in examples.items():
        print(f"==== {source}")
        show(messages)


if __name__ == "__main__":
    main()
