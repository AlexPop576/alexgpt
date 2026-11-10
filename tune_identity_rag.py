"""Tunes alexgpt/identity_rag.THRESHOLD.
  positives: every Q of identity_qa.txt, matched WITHOUT itself (leave-one-out) -> must find its own group
             + the benchmark's identity questions (never in the file: a real held-out test)
  negatives: facts, chit-chat requests, tools -> must NOT be accepted
    .venv\\Scripts\\python.exe tune_identity_rag.py"""
import json

from alexgpt import identity_rag

NEGATIVES = [
    "who is Alesso?", "who invented the telephone?", "what is the capital of France?", "who is Albert Einstein?",
    "where is Romania?", "cine a scris Luceafarul?", "ce este NATO?", "tell me a joke", "write a poem about the sea",
    "what time is it in Paris?", "what is the weather in Cluj?", "2+2", "what is 15% of 240?", "who was Napoleon?",
    "cine este presedintele Romaniei?", "how do airplanes fly?", "what is love?", "explain photosynthesis",
    "can you write me a python function?", "translate hello to french", "what is the most visited country?",
    "who are the Beatles?", "what are black holes?", "how old is the universe?", "what is your opinion on pizza?",
    "recommend me a movie", "spune-mi o gluma", "ce faci in weekend?", "who made the Eiffel Tower?",
    "what model of car is the best?", "how many parameters does GPT-4 have?", "what is the name of the largest ocean?",
]


def check(threshold):
    identity_rag.THRESHOLD = threshold
    groups = identity_rag.read_groups()
    right, total = 0, 0
    for name, group in groups.items():
        for question in group["questions"]:
            index = identity_rag.IdentityIndex(skip_questions=[question])
            found, score, _ = index.match(question)
            accepted = index.accepts(question, found, score)
            total += 1
            right += accepted and found == name
    full = identity_rag.index()
    false_accepts = [q for q in NEGATIVES if full.accepts(q, *full.match(q)[:2])]
    return right, total, false_accepts


def main():
    benchmark = [json.loads(line) for line in open("benchmarks/alexgpt100.jsonl", encoding="utf-8")]
    benchmark = [item for item in benchmark if item["category"] == "identity"]
    print("threshold  leave-one-out right   false accepts")
    for threshold in (0.25, 0.30, 0.35, 0.40, 0.45, 0.50):
        right, total, false_accepts = check(threshold)
        print(f"   {threshold:.2f}      {right:3d}/{total}            {len(false_accepts)}  {false_accepts[:4]}")
    identity_rag.THRESHOLD = 0.35
    print("\nbenchmark identity questions (held out) at 0.35:")
    full = identity_rag.index()
    for item in benchmark:
        group, score, question = full.match(item["prompt"])
        print(f"   {'YES' if full.accepts(item['prompt'], group, score) else 'no '} {score:.2f} {item['prompt'][:45]:45s} -> {group} ({question})")


main()
