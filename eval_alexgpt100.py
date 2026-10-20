"""
AlexGPT-100: our own benchmark, the same 100 questions for every model.
    python eval_alexgpt100.py                                   Supernova base (runs/supernova/final.pt)
    python eval_alexgpt100.py --checkpoint runs/sft/final.pt --format chat --label supernova-chat
    python eval_alexgpt100.py --grade runs/bench/supernova-base.jsonl     give the manual points (0/1/2)

7 categories (benchmarks/alexgpt100.jsonl, made by benchmarks/make_alexgpt100.py):
    conversation  facts  rag  math  instructions  summary  identity
Every answer gets the automatic checks; conversation and summary also need your points (--grade).
Score of a question = the average of its checks (each 0..1). Score of the model = the average of the
7 categories, x100. The answers and scores are saved in runs/bench/<label>.jsonl.

--format base  the model only continues text, so the question is written as "Question: ... Answer:"
               with two solved examples in front (the usual way to test a base model)
--format chat  after SFT: <|user|> question <|assistant|> answer, like in the chat
"""
import argparse
import json
import os
import re
import time
import unicodedata

import torch

from alexgpt.bpe_fast import FastBPE
from alexgpt.supernova import Supernova

QUESTIONS = "benchmarks/alexgpt100.jsonl"
CATEGORIES = ["conversation", "facts", "rag", "math", "instructions", "summary", "identity"]
MAX_NEW_TOKENS = 150

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default="runs/supernova/final.pt")
parser.add_argument("--format", choices=["base", "chat"], default="base")
parser.add_argument("--label", default=None, help="name of the results file (default: from the checkpoint)")
parser.add_argument("--grade", default=None, help="a results file: give the manual points")
args = parser.parse_args()


# ====================================================================== prompts
BASE_EXAMPLES = ("Question: What is the capital of Spain?\nAnswer: The capital of Spain is Madrid.\n\n"
                 "Întrebare: Câte picioare are un păianjen?\nRăspuns: Un păianjen are opt picioare.\n\n")


def build_prompt(item, text_format):
    """The text the model sees. base: Question/Answer with 2 examples; chat: the special tokens."""
    if text_format == "chat":
        text = ""
        if "context" in item:
            text += "<|context|>" + item["context"]
        return text + "<|user|>" + item["prompt"] + "<|assistant|>"
    question_word, answer_word = ("Question", "Answer") if item["lang"] == "en" else ("Întrebare", "Răspuns")
    text = BASE_EXAMPLES
    if "context" in item:
        text += "Text: " + item["context"] + "\n"
    return text + f"{question_word}: {item['prompt']}\n{answer_word}:"


STOP_TEXTS = {"base": ["\n\n", "\nQuestion:", "\nÎntrebare:"], "chat": ["<|user|>", "<|endoftext|>"]}


# ====================================================================== generation
@torch.no_grad()
def generate(model, tokenizer, prompt, text_format, device):
    """Greedy (always the most likely token) + a small penalty for tokens it already wrote,
    so the answers are the same every run and don't loop."""
    ids = tokenizer.encode(prompt)
    start = len(ids)
    answer = ""
    for _ in range(MAX_NEW_TOKENS):
        context = torch.tensor([ids[-model.context_size:]], device=device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            logits, _ = model(context)
        scores = logits[0, -1].float()
        for token_id in set(ids[start:]):                 # repetition penalty 1.2
            scores[token_id] = scores[token_id] / 1.2 if scores[token_id] > 0 else scores[token_id] * 1.2
        next_id = int(scores.argmax())
        if next_id == tokenizer.end_id:
            break
        ids.append(next_id)
        answer = tokenizer.decode(ids[start:])
        stop_positions = [answer.find(stop) for stop in STOP_TEXTS[text_format] if stop in answer]
        if stop_positions:
            answer = answer[:min(stop_positions)]
            break
    return answer.strip()


# ====================================================================== checks
def plain(text):
    """lowercase, no diacritics: "Dunărea" -> "dunarea" (so the checks don't fail on ă/ș/ț)"""
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(char for char in text if not unicodedata.combining(char))


def contains(answer, values):
    answer = plain(answer)
    for value in values:
        value = plain(value)
        if len(value) <= 3:                               # short words must stand alone ("au" is not in "August")
            if re.search(r"(?<![a-z0-9])" + re.escape(value) + r"(?![a-z0-9])", answer):
                return True
        elif value in answer:
            return True
    return False


def has_number(answer, value):
    for token in re.findall(r"\d[\d.,]*\d|\d", answer):
        try:
            if int(token.replace(".", "").replace(",", "")) == value:
                return True
            if float(token.replace(",", ".")) == value:
                return True
        except ValueError:
            pass
    return False


def list_lines(answer):
    return [line for line in answer.split("\n") if re.match(r"^\s*([-*•]|\d+[.)])\s+\S", line)]


def sentence_count(answer):
    parts = re.split(r"[.!?]+(?:\s|$)", answer.strip())
    return len([part for part in parts if re.search(r"\w", part)])


RO_WORDS = {"si", "este", "de", "la", "pe", "cu", "ca", "nu", "sa", "un", "o", "sunt", "care", "mai", "pentru",
            "din", "ce", "am", "ai", "foarte", "acest", "eu", "tu", "imi", "esti", "fi", "cand", "dar"}
EN_WORDS = {"the", "and", "is", "to", "of", "it", "you", "that", "for", "are", "with", "this", "on", "i", "be",
            "my", "was", "have", "your", "what", "but", "can", "not", "do"}


def language(answer):
    words = re.findall(r"[a-z]+", plain(answer))
    ro = sum(word in RO_WORDS for word in words)
    en = sum(word in EN_WORDS for word in words)
    if ro + en == 0 and len(words) < 3:
        return "?"                                         # no real words: neither language
    if ro == en:
        return "ro" if re.search(r"[ăâîșțşţ]", answer.lower()) else "en"
    return "ro" if ro > en else "en"


UNSURE = ["don't know", "do not know", "not sure", "can't know", "cannot know", "no way to know", "don't have access",
          "do not have access", "not able to", "unable to", "search", "look it up", "i'm not certain",
          "nu stiu", "nu sunt sigur", "nu am de unde", "nu am acces", "nu pot sti", "nu pot sa stiu", "caut", "verific"]


def run_check(check, answer):
    """-> 0 or 1 for automatic checks, None for a manual one that is not graded yet"""
    kind = check["type"]
    if kind == "contains":
        return int(contains(answer, check["values"]))
    if kind == "number":
        return int(has_number(answer, check["value"]))
    if kind == "bullets":
        return int(len(list_lines(answer)) == check["value"])
    if kind == "max_words":
        return int(0 < len(answer.split()) <= check["value"])
    if kind == "max_sentences":
        return int(0 < sentence_count(answer) <= check["value"])
    if kind == "language":
        return int(answer != "" and language(answer) == check["value"])
    if kind == "bold":
        return int(bool(re.search(r"\*\*[^*\n]+\*\*|__[^_\n]+__", answer)))
    if kind == "heading":
        return int(bool(re.search(r"^\s*#{1,6}\s+\S", answer, re.MULTILINE)))
    if kind == "unsure":
        return int(contains(answer, UNSURE))
    if kind == "manual":
        return None if check.get("points") is None else check["points"] / 2
    raise ValueError(kind)


CONTENT_CHECKS = ("contains", "number", "unsure")


def score_item(item):
    """0..1 for one question.
    With a manual check: your points (0/1/2 -> 0, 0.5, 1) times the share of automatic checks passed;
                         not graded yet -> None (left out of the averages until you grade it).
    Without: if the content is wrong (contains / number / unsure fail) -> 0, because a nice list of
             wrong facts is still wrong; otherwise the average of the checks."""
    results = [run_check(check, item["answer"]) for check in item["checks"]]
    item["results"] = results
    automatic = [value for check, value in zip(item["checks"], results) if check["type"] != "manual"]
    manual = [value for check, value in zip(item["checks"], results) if check["type"] == "manual"]
    passed = sum(automatic) / len(automatic) if automatic else 1.0
    if manual:
        item["score"] = None if manual[0] is None else manual[0] * passed
        return item
    content_failed = any(value == 0 for check, value in zip(item["checks"], results) if check["type"] in CONTENT_CHECKS)
    item["score"] = 0.0 if content_failed else passed
    return item


def report(items, label):
    print(f"\nAlexGPT-100   {label}")
    print(f"{'category':14s} {'score':>6s}   EN      RO     not graded")
    category_scores = []
    for category in CATEGORIES:
        rows = [item for item in items if item["category"] == category]
        scores = [item["score"] for item in rows if item["score"] is not None]
        by_lang = {}
        for lang in ("en", "ro"):
            values = [item["score"] for item in rows if item["lang"] == lang and item["score"] is not None]
            by_lang[lang] = f"{100 * sum(values) / len(values):5.0f}" if values else "    -"
        waiting = sum(1 for item in rows for check in item["checks"] if check["type"] == "manual" and check.get("points") is None)
        if scores:
            category_scores.append(sum(scores) / len(scores))
        value = f"{100 * sum(scores) / len(scores):5.0f}" if scores else "    -"
        print(f"{category:14s} {value}   {by_lang['en']}  {by_lang['ro']}     {waiting if waiting else ''}")
    total = 100 * sum(category_scores) / len(category_scores) if category_scores else 0
    print(f"{'TOTAL':14s} {total:5.1f}")
    return total


def save(items, path):
    with open(path, "w", encoding="utf-8") as file:
        for item in items:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")


# ====================================================================== grading by hand
def grade(path):
    items = [json.loads(line) for line in open(path, encoding="utf-8")]
    for item in items:
        for check in item["checks"]:
            if check["type"] != "manual" or check.get("points") is not None:
                continue
            print("\n" + "=" * 70)
            print(f"{item['id']}   ({check['rubric']})")
            if "context" in item:
                print("TEXT:     " + item["context"][:300] + ("…" if len(item["context"]) > 300 else ""))
            print("QUESTION: " + item["prompt"])
            print("ANSWER:   " + (item["answer"] or "(empty)"))
            while True:
                choice = input("points 0 / 1 / 2  (s = skip, q = save and quit): ").strip().lower()
                if choice in ("0", "1", "2", "s", "q"):
                    break
            if choice == "q":
                save([score_item(row) for row in items], path)
                report(items, os.path.basename(path))
                return
            if choice != "s":
                check["points"] = int(choice)
    items = [score_item(item) for item in items]
    save(items, path)
    report(items, os.path.basename(path))


# ====================================================================== main
def main():
    if args.grade:
        grade(args.grade)
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = Supernova(**saved["config"])
    model.load_state_dict(saved["weights"] if "weights" in saved else saved["model"])
    model = model.to(device).eval()
    tokenizer = FastBPE.load("checkpoints/supernova_tokenizer.json")
    label = args.label or (os.path.basename(os.path.dirname(args.checkpoint)) + "-" + args.format)
    items = [json.loads(line) for line in open(QUESTIONS, encoding="utf-8")]
    print(f"{args.checkpoint} (step {saved.get('step')}) on {device}, format {args.format}, {len(items)} questions")
    start = time.time()
    for number, item in enumerate(items, 1):
        item["answer"] = generate(model, tokenizer, build_prompt(item, args.format), args.format, device)
        score_item(item)
        mark = "…" if item["score"] is None else ("✓" if item["score"] == 1 else ("~" if item["score"] > 0 else "✗"))
        print(f"  {number:3d} {mark} {item['id']:16s} {item['answer'][:70]!r}", flush=True)
    os.makedirs("runs/bench", exist_ok=True)
    path = f"runs/bench/{label}.jsonl"
    save(items, path)
    total = report(items, label)
    print(f"\n{time.time() - start:.0f} s.  Answers in {path}.  Manual points: python eval_alexgpt100.py --grade {path}")
    with open("runs/benchmarks.txt", "a", encoding="utf-8") as file:
        file.write(time.strftime("%Y-%m-%d %H:%M  ") + f"AlexGPT-100 {label}: {total:.1f} (before manual grading)\n")


if __name__ == "__main__":
    main()
