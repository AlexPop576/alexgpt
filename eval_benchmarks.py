"""
One evaluator for every model (ours and OpenAI's GPT-2), so the numbers can be compared fairly.
    python eval_benchmarks.py --model gpt2
    python eval_benchmarks.py --model runs/supernova/final.pt
    python eval_benchmarks.py --model runs/singularity/final.pt --tasks hellaswag,arc_easy,piqa
    python eval_benchmarks.py --model runs/supernova/final.pt --limit 200        (quick check)

Every question = a start + 2..5 possible endings, one is right. The model writes nothing: for every ending we
add up the log-probabilities of its tokens (given the start). The same rules as EleutherAI's lm-evaluation-harness
(what papers and the Open LLM Leaderboard use), so GPT-2's numbers here can be checked against the published ones:
  acc        = the ending with the highest total log-probability
  acc_norm   = the same, divided by the ending's length in BYTES (tokenizer-independent: fair between our
               16K vocabulary and GPT-2's 50K one). This is the main number for HellaSwag.
  95% CI     = bootstrap: draw the questions again with replacement 1000 times, take the 2.5% and 97.5% points.
Tasks (all multiple choice, English):
  hellaswag  10,042 questions, 4 endings, everyday situations          random = 25%
  arc_easy    2,376 questions, 3-5 answers, school science             random ~ 25%
  piqa        1,838 questions, 2 answers, physical common sense         random = 50%
Results: printed, appended to runs/benchmarks.txt and saved to runs/bench/<model>_<task>.json.
"""
import argparse
import json
import os
import re
import time

import numpy as np
import torch
import torch.nn.functional as F

BENCH_DIR = "data/bench"

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, help='"gpt2" or the path of one of our checkpoints')
parser.add_argument("--tasks", default="hellaswag,arc_easy,piqa")
parser.add_argument("--limit", type=int, default=0, help="only the first N questions of every task")
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------ the questions (lm-evaluation-harness format)
def hellaswag_text(text):
    text = text.strip().replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def download(task):
    """-> list of {"context", "choices", "label"}; saved once in data/bench/<task>.jsonl"""
    from datasets import load_dataset
    questions = []
    if task == "hellaswag":
        for row in load_dataset("Rowan/hellaswag", split="validation"):
            context = row["ctx_a"] + " " + row["ctx_b"].capitalize()
            questions.append({"context": hellaswag_text(row["activity_label"] + ": " + context),
                              "choices": [" " + hellaswag_text(ending) for ending in row["endings"]],
                              "label": int(row["label"])})
    elif task == "arc_easy":
        for row in load_dataset("allenai/ai2_arc", "ARC-Easy", split="test"):
            questions.append({"context": "Question: " + row["question"] + "\nAnswer:",
                              "choices": [" " + text for text in row["choices"]["text"]],
                              "label": row["choices"]["label"].index(row["answerKey"])})
    elif task == "piqa":
        rows = None
        for name in ("ybisk/piqa", "baber/piqa"):              # the original needs an old loading script
            try:
                rows = load_dataset(name, split="validation")
                break
            except Exception as error:
                print(f"  {name}: {type(error).__name__}: {str(error)[:150]}")
        if rows is None:
            raise RuntimeError("PIQA could not be downloaded")
        for row in rows:
            questions.append({"context": "Question: " + row["goal"] + "\nAnswer:",
                              "choices": [" " + row["sol1"], " " + row["sol2"]], "label": int(row["label"])})
    else:
        raise ValueError(task)
    return questions


def load_task(task):
    path = os.path.join(BENCH_DIR, task + ".jsonl")
    if not os.path.exists(path):
        os.makedirs(BENCH_DIR, exist_ok=True)
        print(f"downloading {task} ...")
        questions = download(task)
        with open(path, "w", encoding="utf-8") as file:
            for question in questions:
                file.write(json.dumps(question, ensure_ascii=False) + "\n")
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file]


# ------------------------------------------------------------------ models
def load_model(name, device):
    """-> (model, tokenizer, short name). Every model: model(ids) -> (logits, _), tokenizer.encode(text)."""
    if name == "gpt2":
        from alexgpt.gpt2 import load_gpt2
        model, tokenizer = load_gpt2(device)
        return model, tokenizer, "gpt2-small"
    from alexgpt.bpe_fast import FastBPE
    from alexgpt.supernova import Supernova
    saved = torch.load(name, map_location="cpu", weights_only=False)
    model = Supernova(**saved["config"])
    model.load_state_dict(saved["weights"] if "weights" in saved else saved["model"])
    tokenizer = FastBPE.load("checkpoints/supernova_tokenizer.json")
    short = name.replace("runs/", "").replace(".pt", "").replace("/", "-").replace("\\", "-")
    return model.to(device).eval(), tokenizer, short


# ------------------------------------------------------------------ scoring
@torch.no_grad()
def score_question(model, tokenizer, question, device):
    """-> list of (total log-probability of the ending, its length in bytes, its number of tokens), one per choice"""
    context_ids = tokenizer.encode(question["context"])
    rows, ending_lengths = [], []
    for choice in question["choices"]:
        ending_ids = tokenizer.encode(choice)
        keep = model.context_size - len(ending_ids)
        rows.append(context_ids[-keep:] + ending_ids)          # very long starts: keep their end
        ending_lengths.append(len(ending_ids))
    longest = max(len(row) for row in rows)
    ids = torch.zeros(len(rows), longest, dtype=torch.long)
    for index, row in enumerate(rows):
        ids[index, :len(row)] = torch.tensor(row)             # padding at the END: never seen by real tokens
    ids = ids.to(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        logits, _ = model(ids)
    log_probs = F.log_softmax(logits.float(), dim=-1)
    results = []
    for index, row in enumerate(rows):
        ending_start = len(row) - ending_lengths[index]       # first token of the ending
        positions = torch.arange(ending_start - 1, len(row) - 1, device=device)   # the positions that predict it
        targets = ids[index, ending_start:len(row)]
        total = log_probs[index, positions, targets].sum().item()
        results.append((total, len(question["choices"][index].encode("utf-8")), ending_lengths[index]))
    return results


def bootstrap_interval(correct, rounds=1000, seed=0):
    rng = np.random.default_rng(seed)
    correct = np.asarray(correct, dtype=np.float64)
    samples = rng.integers(0, len(correct), size=(rounds, len(correct)))
    means = correct[samples].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def run_task(model, tokenizer, short_name, task, limit, device):
    questions = load_task(task)
    if limit:
        questions = questions[:limit]
    right = {"acc": [], "acc_norm": [], "acc_per_token": []}
    start = time.time()
    for number, question in enumerate(questions, 1):
        scores = score_question(model, tokenizer, question, device)
        picks = {"acc": max(range(len(scores)), key=lambda i: scores[i][0]),
                 "acc_norm": max(range(len(scores)), key=lambda i: scores[i][0] / scores[i][1]),
                 "acc_per_token": max(range(len(scores)), key=lambda i: scores[i][0] / scores[i][2])}
        for metric, pick in picks.items():
            right[metric].append(int(pick == question["label"]))
        if number % 1000 == 0:
            print(f"   {task} {number}/{len(questions)}  acc_norm {100 * np.mean(right['acc_norm']):.1f}%  "
                  f"{time.time() - start:.0f} s", flush=True)
    summary = {"model": short_name, "task": task, "questions": len(questions)}
    for metric, values in right.items():
        low, high = bootstrap_interval(values)
        summary[metric] = round(100 * float(np.mean(values)), 2)
        summary[metric + "_ci95"] = [round(100 * low, 2), round(100 * high, 2)]
    os.makedirs("runs/bench", exist_ok=True)
    with open(f"runs/bench/{short_name}_{task}{'_limit' if limit else ''}.json", "w", encoding="utf-8") as file:
        json.dump({**summary, "per_question_acc_norm": right["acc_norm"]}, file)
    line = (f"{short_name:28s} {task:10s} acc {summary['acc']:5.1f}%  acc_norm {summary['acc_norm']:5.1f}% "
            f"[{summary['acc_norm_ci95'][0]:.1f}-{summary['acc_norm_ci95'][1]:.1f}]  "
            f"per-token {summary['acc_per_token']:5.1f}%  ({len(questions)} q{', LIMITED' if limit else ''})")
    print(line, flush=True)
    with open("runs/benchmarks.txt", "a", encoding="utf-8") as file:
        file.write(time.strftime("%Y-%m-%d %H:%M  ") + line + "\n")
    return summary


def main():
    args = parser.parse_args()
    model, tokenizer, short_name = load_model(args.model, args.device)
    print(f"{short_name} on {args.device}")
    for task in args.tasks.split(","):
        run_task(model, tokenizer, short_name, task.strip(), args.limit, args.device)


if __name__ == "__main__":
    main()
