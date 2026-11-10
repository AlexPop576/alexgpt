"""Runs benchmarks/failures300.jsonl through the SAME path as the web app (alexgpt/chat.py generate_stream).
    .venv\\Scripts\\python.exe eval_failures.py --label v4-round1                (all items, ~20 min, live Wikipedia)
    .venv\\Scripts\\python.exe eval_failures.py --label quick --category facts_en --limit 5
Grades every last answer as right / abstain / wrong (see benchmarks/make_failures300.py for the rules) and prints
a table per category, dev and held-out SEPARATELY.  Tune only on dev; held-out shows if the gain is real.
Answers: runs/bench/failures-<label>.jsonl;  one line per run in runs/benchmarks.txt."""
import argparse
import json
import os
import re
import time
import unicodedata

import numpy as np

from alexgpt import tools
from alexgpt.chat import SingularityChat, generate_stream

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default="runs/singularity_sft4/final.pt")
parser.add_argument("--label", default="failures")
parser.add_argument("--category", default=None, help="only this category")
parser.add_argument("--split", choices=["all", "dev", "heldout"], default="all")
parser.add_argument("--limit", type=int, default=0, help="at most this many items (0 = all)")
parser.add_argument("--no-abstain", action="store_true", help="the old behaviour: no 'no source' reply")
args = parser.parse_args()

SAID_NOTHING_FOUND = re.compile(r"couldn't find|could not find|no source|nu am gasit|nu am găsit|don't know|nu stiu|nu știu|"
                                r"never|did not|didn't|nu a |n-a ", re.I)


def fold(text):
    return "".join(c for c in unicodedata.normalize("NFKD", text.lower()) if not unicodedata.combining(c))


def ask(model, turns, number):
    """plays the whole dialogue; -> (last answer, abstained, identity, tool calls of the last turn)"""
    history = []
    for turn in turns:
        rng = np.random.default_rng(number)
        answer, abstained, identity, calls = "", False, False, []
        for step in generate_stream(model, turn, 1500, 0.7, rng, history):
            if step["type"] == "tool":
                calls.append(step["call"] + (" [unused]" if step.get("unused") else ""))
                time.sleep(1)                           # Wikipedia limits fast requests
            elif step["type"] == "token":
                answer += step["char"]
                abstained = abstained or bool(step.get("abstain"))
                identity = identity or bool(step.get("identity"))
        answer = answer.replace("<|endoftext|>", "").strip()
        history += [{"role": "user", "text": turn}, {"role": "assistant", "text": answer}]
    return answer, abstained, identity, calls


def grade(item, answer, abstained):
    """-> 'right', 'abstain' or 'wrong'"""
    plain = fold(answer)
    expect = item["expect"]
    found = re.search(fold(expect), plain) is not None if expect else True
    kind = item["kind"]
    if kind == "fact":
        if abstained:
            return "abstain"
        return "right" if found else "wrong"
    if kind == "false_premise":
        return "right" if abstained or SAID_NOTHING_FOUND.search(answer) else "wrong"
    if kind == "chat":
        if abstained:
            return "wrong"
        return "right" if (tools.is_romanian(answer) == (item["lang"] == "ro")) else "wrong"
    if kind == "style":
        short = len(answer) < 300 and not re.search(r"\n\s*(?:[-•*]|\d+\.)\s", answer)
        return "right" if short and found else "wrong"
    if kind == "list":
        if abstained:
            return "abstain"
        names = {match.group(0) for match in re.finditer(fold(expect), plain)}
        return "right" if len(names) >= item["min_matches"] else "wrong"
    return "right" if found else ("abstain" if abstained else "wrong")      # identity, tool


def report(items):
    categories = list(dict.fromkeys(item["category"] for item in items))
    lines = [f"{'category':20s} {'split':8s} {'n':>4s} {'right':>6s} {'abstain':>8s} {'wrong':>6s}"]
    totals = {}
    for category in categories + ["ALL"]:
        for split in ("dev", "heldout"):
            chosen = [i for i in items if (category == "ALL" or i["category"] == category) and i["split"] == split]
            if not chosen:
                continue
            count = {g: sum(i["grade"] == g for i in chosen) for g in ("right", "abstain", "wrong")}
            n = len(chosen)
            lines.append(f"{category:20s} {split:8s} {n:4d} {100 * count['right'] / n:5.0f}% {100 * count['abstain'] / n:7.0f}% "
                         f"{100 * count['wrong'] / n:5.0f}%")
            if category == "ALL":
                totals[split] = 100 * count["right"] / n
    print("\n".join(lines))
    return totals


def main():
    items = [json.loads(line) for line in open("benchmarks/failures300.jsonl", encoding="utf-8")]
    if args.category:
        items = [i for i in items if i["category"] == args.category]
    if args.split != "all":
        items = [i for i in items if i["split"] == args.split]
    if args.limit:
        items = items[:args.limit]
    model = SingularityChat(args.checkpoint)
    model.abstain_enabled = not args.no_abstain
    print(f"{len(items)} items, abstain {'off' if args.no_abstain else 'on'}")
    start = time.time()
    for number, item in enumerate(items, 1):
        answer, abstained, identity, calls = ask(model, item["turns"], number)
        item.update({"answer": answer, "abstained": abstained, "identity_rag": identity, "tool_calls": calls})
        item["grade"] = grade(item, answer, abstained)
        mark = {"right": "✓", "abstain": "A", "wrong": "✗"}[item["grade"]]
        print(f"  {number:3d} {mark} {item['id']:22s} {answer[:80]!r}", flush=True)
    os.makedirs("runs/bench", exist_ok=True)
    path = f"runs/bench/failures-{args.label}.jsonl"
    with open(path, "w", encoding="utf-8") as file:
        for item in items:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")
    print()
    totals = report(items)
    print(f"\n{time.time() - start:.0f} s.  Answers in {path}")
    with open("runs/benchmarks.txt", "a", encoding="utf-8") as file:
        file.write(time.strftime("%Y-%m-%d %H:%M  ") + f"failures300 {args.label}: right dev {totals.get('dev', 0):.1f}% "
                   f"heldout {totals.get('heldout', 0):.1f}%\n")


if __name__ == "__main__":
    main()
