"""
HellaSwag: the standard test for small base models.  python eval_hellaswag.py
                                                      python eval_hellaswag.py --limit 500   (quick, ~1 minute)
Every question = the start of a situation + 4 possible endings, only 1 makes sense:
    "A man is sitting on a roof. He ..."
        a) "is using wrap to wrap a pair of skis."
        b) "is ripping level tiles off."              <- the right one
        c) "is holding a rubik's cube."
        d) "starts pulling up roofing on a roof."
The model does not WRITE anything. For every ending we measure how surprised it is by it
(the average loss of the ending's tokens, given the start). The least surprising ending = its answer.
Random guessing = 25%.  GPT-2 (OpenAI, 2019): 124M ~29-31%, 350M ~35-37% (exact numbers depend on the scoring).
The 10,042 validation questions are downloaded once into data/hellaswag_val.jsonl.
"""
import argparse
import json
import os
import time
import urllib.request

import torch
import torch.nn.functional as F

from alexgpt.bpe_fast import FastBPE
from alexgpt.supernova import Supernova

URL = "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl"
DATA_FILE = "data/hellaswag_val.jsonl"

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default="runs/supernova/final.pt")
parser.add_argument("--limit", type=int, default=0, help="only the first N questions (0 = all 10,042)")
args = parser.parse_args()


def load_questions():
    if not os.path.exists(DATA_FILE):
        os.makedirs("data", exist_ok=True)
        print("downloading HellaSwag (validation, ~11 MB) ...")
        try:
            urllib.request.urlretrieve(URL, DATA_FILE)
        except Exception as error:                       # GitHub unreachable: the same data from Hugging Face
            print(f"  GitHub failed ({error}), trying Hugging Face ...")
            from datasets import load_dataset
            with open(DATA_FILE, "w", encoding="utf-8") as file:
                for row in load_dataset("Rowan/hellaswag", split="validation"):
                    file.write(json.dumps({"ctx": row["ctx"], "endings": row["endings"], "label": int(row["label"])}) + "\n")
    questions = []
    with open(DATA_FILE, encoding="utf-8") as file:
        for line in file:
            questions.append(json.loads(line))
    return questions


def load_model(path, device):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    model = Supernova(**saved["config"])
    model.load_state_dict(saved["weights"] if "weights" in saved else saved["model"])
    return model.to(device).eval(), saved.get("step")


@torch.no_grad()
def answer(model, tokenizer, question, device):
    """Returns (pick by total loss, pick by average loss per token) for one question."""
    start_ids = tokenizer.encode(question["ctx"])
    rows, ending_lengths = [], []
    for ending in question["endings"]:
        ending_ids = tokenizer.encode(" " + ending)
        rows.append(start_ids + ending_ids)
        ending_lengths.append(len(ending_ids))
    longest = max(len(row) for row in rows)
    ids = torch.zeros(4, longest, dtype=torch.long)
    for index, row in enumerate(rows):
        ids[index, :len(row)] = torch.tensor(row)
    ids = ids.to(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        logits, _ = model(ids)
    # loss of every token = -log(chance the model gave to the token that really came next)
    losses = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.size(-1)), ids[:, 1:].reshape(-1),
                             reduction="none").view(4, -1)
    totals, averages = [], []
    for index in range(4):
        first = len(start_ids) - 1                         # the position that predicts the ending's first token
        ending_losses = losses[index, first:first + ending_lengths[index]]
        totals.append(ending_losses.sum().item())
        averages.append(ending_losses.mean().item())
    return totals.index(min(totals)), averages.index(min(averages))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = FastBPE.load("checkpoints/supernova_tokenizer.json")
    model, step = load_model(args.checkpoint, device)
    questions = load_questions()
    if args.limit:
        questions = questions[:args.limit]
    print(f"{args.checkpoint} (step {step}) on {device}, {len(questions)} questions")
    right_total, right_average = 0, 0
    start = time.time()
    for number, question in enumerate(questions, 1):
        by_total, by_average = answer(model, tokenizer, question, device)
        right_total += by_total == question["label"]
        right_average += by_average == question["label"]
        if number % 500 == 0 or number == len(questions):
            print(f"  {number:6d}   accuracy {100 * right_average / number:5.1f}%  (normalized)   "
                  f"{100 * right_total / number:5.1f}%  (total)   {time.time() - start:4.0f} s", flush=True)
    result = (f"HellaSwag {args.checkpoint} step {step}: {100 * right_average / len(questions):.1f}% normalized, "
              f"{100 * right_total / len(questions):.1f}% total, {len(questions)} questions")
    print("\n" + result)
    os.makedirs("runs", exist_ok=True)
    with open("runs/benchmarks.txt", "a", encoding="utf-8") as file:
        file.write(time.strftime("%Y-%m-%d %H:%M  ") + result + "\n")


if __name__ == "__main__":
    main()
