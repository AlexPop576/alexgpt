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
parser.add_argument("--name", default="Supernova", help="the model's name, for the system prompt (chat format)")
parser.add_argument("--tools", action="store_true", help="chat format: run the tool calls (calc, wiki) like the app does")
parser.add_argument("--no-system", action="store_true", help="chat format without the system prompt (like the first SFT)")
parser.add_argument("--app", action="store_true",
                    help="Singularity exactly like the web app: tool_router rules + live Wikipedia + generate_stream")
parser.add_argument("--no-abstain", action="store_true", help="with --app: no 'no source' reply (the old behaviour)")
args = parser.parse_args()


# ====================================================================== prompts
BASE_EXAMPLES = ("Question: What is the capital of Spain?\nAnswer: The capital of Spain is Madrid.\n\n"
                 "Întrebare: Câte picioare are un păianjen?\nRăspuns: Un păianjen are opt picioare.\n\n")


def build_prompt(item, text_format, extra=()):
    """The text the model sees. base: Question/Answer with 2 examples; chat: the special tokens."""
    if text_format == "chat":                        # exactly like the chat (alexgpt/sft_format.py)
        from alexgpt.sft_format import chat_prompt
        question = item["prompt"]
        if "context" in item:
            question = item["context"] + "\n\n" + question      # the text first, then the question (as in the SFT data)
        messages = [{"role": "user", "content": question}]
        if not args.no_system:                      # exactly what the chat sends: the system prompt first
            from alexgpt.sft_format import system_prompt
            messages.insert(0, {"role": "system", "content": system_prompt(args.name)})
        return chat_prompt(messages + list(extra))
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


# ====================================================================== tools (only with --tools)
def eval_calc(expression):
    """small safe calculator, only used if alexgpt/tools.py calc is not written yet"""
    import ast
    import math
    import operator
    operators = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
                 ast.Pow: operator.pow, ast.USub: operator.neg, ast.Mod: operator.mod}
    expression = re.sub(r"(\d+(?:\.\d+)?)\s*%\s*of\s*(\d+(?:\.\d+)?)", r"(\1/100)*\2", expression)

    def walk(node):
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in operators:
            return operators[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in operators:
            return operators[type(node.op)](walk(node.operand))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "sqrt" and len(node.args) == 1:
            return math.sqrt(walk(node.args[0]))
        raise ValueError("not allowed")
    value = walk(ast.parse(expression.strip(), mode="eval"))
    return str(int(value)) if float(value).is_integer() else str(round(value, 6))


def run_tool_for_eval(call):
    try:
        from alexgpt.tools import run_tool
        return run_tool(call), "tools.py"
    except NotImplementedError:
        pass
    except Exception as error:
        return f"error: {error}", "tools.py"
    name, _, arguments = call.partition(":")
    arguments = arguments.strip()
    try:
        if name.strip() == "calc":
            return eval_calc(arguments), "eval fallback"
        if name.strip() == "wiki":
            from gen_data_wiki import clean_and_cut, fetch_intro
            found = fetch_intro("ro" if re.search(r"[ăâîșț]", arguments) else "en", arguments)
            if not found:
                return f"error: nothing found for '{arguments}'", "eval fallback"
            return f"{found[0]} — {clean_and_cut(found[1])}", "eval fallback"
    except Exception as error:
        return f"error: {error}", "eval fallback"
    return f"error: tool '{name}' not available in this evaluation", "eval fallback"


def answer_with_tools(model, tokenizer, item, device):
    """generate; if the model writes <|search|>tool: args, run it, add <|context|>result, generate again (max 2 calls)"""
    extra, calls = [], []
    for _ in range(3):
        text = generate(model, tokenizer, build_prompt(item, "chat", extra), "chat", device)
        if text.startswith("<|search|>"):
            call = text[len("<|search|>"):].strip()
            result, who = run_tool_for_eval(call)
            calls.append(f"{call} -> {result[:60]} [{who}]")
            extra += [{"role": "tool_call", "content": call}, {"role": "context", "content": result}]
            continue
        item["tool_calls"] = calls
        return text
    item["tool_calls"] = calls
    return text


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
    """True if `value` is THE RESULT: the last number of the answer, or the number right after "=" / "is" /
    "este" / "egal". "4 + 3 * 6 = 22" -> yes;  "3 * 6 = 18, plus 4 is 25" -> no (22 appears nowhere as a result).
    (Before, any 22 anywhere counted, so random numbers in a wrong answer got the point.)"""
    numbers = re.findall(r"\d[\d.,]*\d|\d", answer)
    after_equals = re.findall(r"(?:=|\bis\b|\beste\b|\begal(?:\s+cu)?\b|\bsunt\b|\bare\b)\s*(?:about|aproximativ)?\s*(\d[\d.,]*\d|\d)", answer)
    candidates = after_equals + numbers[-1:]
    for token in candidates:
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
def run_app(label):
    """--app: every question goes through the same path as the web page (alexgpt/chat.py generate_stream),
    so the router rules and the 'no source' reply are measured too.  Sampling like the app (temperature 0.7),
    with a fixed seed per question so two runs can be compared."""
    import numpy as np
    from alexgpt.chat import SingularityChat, generate_stream
    model = SingularityChat(args.checkpoint)
    model.abstain_enabled = not args.no_abstain
    items = [json.loads(line) for line in open(QUESTIONS, encoding="utf-8")]
    print(f"{args.checkpoint} like the app, abstain={'off' if args.no_abstain else 'on'}, {len(items)} questions")
    start = time.time()
    for number, item in enumerate(items, 1):
        question = item["prompt"]
        if "context" in item:
            question = item["context"] + "\n\n" + question
        rng = np.random.default_rng(number)
        answer, calls, abstained = "", [], False
        for step in generate_stream(model, question, 600, 0.7, rng, []):
            if step["type"] == "tool":
                calls.append(step["call"] + (" [unused]" if step.get("unused") else "") + " -> " + step["result"][:60])
            elif step["type"] == "token":
                answer += step["char"]
                abstained = abstained or bool(step.get("abstain"))
        item["answer"] = answer.replace("<|endoftext|>", "").strip()
        item["tool_calls"] = calls
        item["abstained"] = abstained
        score_item(item)
        mark = "A" if abstained else ("…" if item["score"] is None else ("✓" if item["score"] == 1 else ("~" if item["score"] > 0 else "✗")))
        print(f"  {number:3d} {mark} {item['id']:16s} {item['answer'][:70]!r}", flush=True)
        time.sleep(1)                                   # Wikipedia answers 429 to fast requests
    os.makedirs("runs/bench", exist_ok=True)
    path = f"runs/bench/{label}.jsonl"
    save(items, path)
    total = report(items, label)
    for category in CATEGORIES:
        count = sum(1 for item in items if item["category"] == category and item["abstained"])
        if count:
            print(f"  abstained in {category}: {count}")
    print(f"\n{time.time() - start:.0f} s.  Answers in {path}.")
    with open("runs/benchmarks.txt", "a", encoding="utf-8") as file:
        file.write(time.strftime("%Y-%m-%d %H:%M  ") + f"AlexGPT-100 {label}: {total:.1f} (before manual grading)\n")


def main():
    if args.grade:
        grade(args.grade)
        return
    if args.app:
        run_app(args.label or ("app-no-abstain" if args.no_abstain else "app-abstain"))
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
        if args.tools and args.format == "chat":
            item["answer"] = answer_with_tools(model, tokenizer, item, device)
        else:
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
