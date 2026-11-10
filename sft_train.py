"""
SFT: turns a base model (it continues text) into a chat model (it answers).
    python build_sft_mix.py                 first: the cleaned data -> data/sft_mix/supernova.jsonl
    python sft_train.py --pilot             15 minutes, a sneak peek
    python sft_train.py                     the real one, 2 passes over the data (~1.5-2 h): runs/sft_v2/final.pt
    later, for Singularity:
    python sft_train.py --base runs/singularity/final.pt --name Singularity --mix data/sft_mix/singularity.jsonl --out runs/singularity_sft

Before it starts it tests YOUR build_example() (alexgpt/sft_format.py): the training uses it.
Ctrl+C once: saves and exits. Run the same command again: continues from there.

What happens:
  1. reads the mix made by build_sft_mix.py (every source cleaned; identity, banter and repeats already in it)
  2. every conversation -> ids + mask with build_example  (saved next to the mix as *_tokens.npz, done once)
     a conversation longer than 1024 tokens is CUT after the last answer that fits (v1 threw it away whole)
  3. starts from the base model; the special tokens (<|user|>, <|assistant|>, ...) never appeared
     in pre-training, so their vectors start as the average of all the others
  4. packs whole conversations into rows of 1024 tokens; loss = cross-entropy only where mask = 1
  5. learning rate 1e-4 (8x lower than pre-training: we adjust it, we don't rebuild it), warm-up, then down
  6. every 200 steps: validation loss + the answers to a few fixed questions (with the system prompt,
     like the chat), in <out>/log.txt
"""
import argparse
import glob
import json
import math
import os
import random
import signal
import sys
import time
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn.functional as F

from alexgpt.bpe_fast import FastBPE
from alexgpt.sft_format import build_example, chat_prompt, system_prompt
from alexgpt.supernova import Supernova

TOKENIZER = "checkpoints/supernova_tokenizer.json"
NOT_FOR_VALIDATION = ("identity_qa", "identity", "banter", "persona", "mine")      # ours, small: all of it goes to training
CONTEXT = 1024
MICRO_BATCH = 2
ACCUMULATION = 16                      # 32 rows x 1024 = ~32K tokens per step
WARMUP_STEPS = 50
EPOCHS = 2
EVAL_EVERY = 200
SAVE_EVERY_MINUTES = 10
SAMPLE_QUESTIONS = ["What is your name?", "Hi! How are you?", "Why is the sky blue?", "Give me three tips for learning faster.",
                    "Salut! Ce mai faci?", "Care este capitala Franței?", "Alex e prost.",
                    "Vreau să îmi spui câteva orașe din România."]

parser = argparse.ArgumentParser()
parser.add_argument("--pilot", action="store_true", help="15 minutes only, for a quick look")
parser.add_argument("--base", default="runs/supernova/final.pt", help="the pre-trained model to start from")
parser.add_argument("--name", default="Supernova", help="the model's name (system prompt)")
parser.add_argument("--mix", default="data/sft_mix/supernova.jsonl", help="made by build_sft_mix.py")
parser.add_argument("--lr", type=float, default=1e-4, help="peak learning rate (lower, e.g. 3e-5, to continue from a chat model)")
parser.add_argument("--out", default="runs/sft_v2", help="where the checkpoints and the log go")
args = parser.parse_args()
LEARNING_RATE = args.lr
BASE = args.base
NAME = args.name
RUN_DIR = args.out + "_pilot" if args.pilot else args.out
PILOT_SECONDS = 15 * 60
os.makedirs(RUN_DIR, exist_ok=True)
LOG = os.path.join(RUN_DIR, "log.txt")
CHECKPOINT = os.path.join(RUN_DIR, "latest.pt")


def log(text):
    print(text, flush=True)
    with open(LOG, "a", encoding="utf-8") as file:
        file.write(text + "\n")


# ====================================================================== data
worker_tokenizer = None


def start_worker():
    global worker_tokenizer
    worker_tokenizer = FastBPE.load(TOKENIZER)


def encode_batch(conversations):
    """-> for every conversation (ids, mask), or None if not even its first answer fits in 1024 tokens.
    Too long: drop the last user+answer pair and try again (a 6-turn chat becomes a 4-turn one)."""
    result = []
    for messages in conversations:
        item = None
        while messages:
            ids, mask = build_example(messages, worker_tokenizer)
            if len(ids) <= CONTEXT + 1 and sum(mask) > 0:
                item = (np.array(ids, dtype=np.uint16), np.array(mask, dtype=np.uint8))
                break
            messages = messages[:-1]                                 # drop the last answer ...
            while messages and messages[-1]["role"] != "assistant":  # ... and the question before it
                messages = messages[:-1]
        result.append(item)
    return result


def load_examples():
    """-> list of (source, ids, mask, repeat). Tokenized once, then read from <mix>_tokens.npz."""
    rows = [json.loads(line) for line in open(args.mix, encoding="utf-8") if line.strip()]
    cache = args.mix[:-6] + "_tokens.npz"
    stamp = f"{os.path.getmtime(args.mix)} {os.path.getsize(args.mix)} {len(rows)}"
    if os.path.exists(cache):
        saved = np.load(cache, allow_pickle=True)
        if str(saved["stamp"]) == stamp:
            examples = list(zip(saved["sources"], saved["ids"], saved["masks"], saved["repeats"]))
            log(f"{len(examples)} conversations (tokenized before, from {cache})")
            return examples
    log(f"tokenizing {len(rows)} conversations from {args.mix} with your build_example() (once) ...")
    examples, counts = [], {}
    with Pool(max(2, min(8, os.cpu_count() - 2)), initializer=start_worker) as pool:
        batches = [rows[i:i + 256] for i in range(0, len(rows), 256)]
        for batch, batch_result in zip(batches, pool.imap(encode_batch, [[row["messages"] for row in batch] for batch in batches])):
            for row, item in zip(batch, batch_result):
                kept, total = counts.get(row["source"], (0, 0))
                counts[row["source"]] = (kept + (item is not None), total + 1)
                if item is not None:
                    examples.append((row["source"], item[0], item[1], row.get("repeat", 1)))
    for source, (kept, total) in sorted(counts.items()):
        log(f"  {source:17s} {kept:6d} of {total:6d} kept")
    np.savez(cache, stamp=stamp, sources=np.array([e[0] for e in examples]),
             ids=np.array([e[1] for e in examples], dtype=object), masks=np.array([e[2] for e in examples], dtype=object),
             repeats=np.array([e[3] for e in examples]))
    return examples


def pack_rows(examples, seed):
    """Whole conversations, shuffled, put one after another into rows of CONTEXT+1 tokens.
    A conversation never gets cut between two rows; the empty end of a row is padding with mask 0."""
    order = list(range(len(examples)))
    random.Random(seed).shuffle(order)
    rows, row_ids, row_mask, row_segments = [], [], [], []
    for index in order:
        ids, mask = examples[index][1], examples[index][2]
        if len(row_ids) + len(ids) > CONTEXT + 1:
            rows.append((row_ids, row_mask, row_segments))
            row_ids, row_mask, row_segments = [], [], []
        conversation_number = row_segments[-1] + 1 if row_segments else 0
        row_ids = row_ids + ids.tolist()
        row_mask = row_mask + mask.tolist()
        row_segments = row_segments + [conversation_number] * len(ids)
    if row_ids:
        rows.append((row_ids, row_mask, row_segments))
    return rows


def to_batch(rows, end_id, device):
    ids = torch.full((len(rows), CONTEXT + 1), end_id, dtype=torch.long)
    mask = torch.zeros((len(rows), CONTEXT + 1), dtype=torch.float32)
    segments = torch.full((len(rows), CONTEXT + 1), 10_000, dtype=torch.long)   # padding: a "conversation" of its own
    for index, (row_ids, row_mask, row_segments) in enumerate(rows):
        ids[index, :len(row_ids)] = torch.tensor(row_ids)
        mask[index, :len(row_mask)] = torch.tensor(row_mask, dtype=torch.float32)
        segments[index, :len(row_segments)] = torch.tensor(row_segments)
    ids, mask, segments = ids.to(device), mask.to(device), segments.to(device)
    # input, target, "does this target count", which conversation every input token belongs to
    return ids[:, :-1], ids[:, 1:], mask[:, 1:], segments[:, :-1]


def masked_loss(model, inputs, targets, mask, segments=None):
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=inputs.is_cuda):
        logits, _ = model(inputs, segments=segments)
    losses = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="none")
    mask = mask.reshape(-1)
    return (losses * mask).sum() / mask.sum().clamp(min=1)


# ====================================================================== main
def main():
    from exercises.test_sft_format import run_tests
    tokenizer = FastBPE.load(TOKENIZER)
    if not run_tests(tokenizer, verbose=False):
        print("build_example() in alexgpt/sft_format.py is not written or not right yet.")
        print("Check it with:  python exercises/test_sft_format.py")
        sys.exit(1)
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"DEVICE: {device}" + (f" ({torch.cuda.get_device_name(0)}, torch {torch.__version__})" if device == "cuda" else f" (torch {torch.__version__}, NO GPU!)"))
    if device == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.92)      # never borrow normal RAM (that froze the PC)
    torch.backends.cuda.matmul.allow_tf32 = True

    examples = load_examples()
    mine = [example for example in examples if example[0] in NOT_FOR_VALIDATION]      # identity, banter
    others = [example for example in examples if example[0] not in NOT_FOR_VALIDATION]
    random.Random(0).shuffle(others)
    val_count = max(200, len(others) // 100)                  # 1% never trained on: the validation loss
    val_examples, train_examples = others[:val_count], []
    for example in others[val_count:] + mine:
        train_examples.extend([example] * int(example[3]))    # "repeat" from build_sft_mix.py (identity 3-6x, ...)
    by_source = {}
    for example in train_examples:
        by_source[example[0]] = by_source.get(example[0], 0) + len(example[1])
    log("training tokens per pass: " + ", ".join(f"{name} {count / 1e6:.1f}M" for name, count in sorted(by_source.items())))

    rows_per_epoch = len(pack_rows(train_examples, 0))
    steps_per_epoch = rows_per_epoch // (MICRO_BATCH * ACCUMULATION) - 1   # -1: other shuffles pack a bit differently
    total_steps = steps_per_epoch * EPOCHS
    val_rows = pack_rows(val_examples, 0)[:64]

    model = Supernova(**torch.load(BASE, map_location="cpu", weights_only=False)["config"]).to(device)
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
                                  lr=LEARNING_RATE, betas=(0.9, 0.95), fused=device == "cuda")
    step = 0
    if os.path.exists(CHECKPOINT):
        state = torch.load(CHECKPOINT, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        step = state["step"]
        log(f"\n=== resumed at step {step}/{total_steps} ===")
    else:
        saved = torch.load(BASE, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["weights"])
        if not saved.get("chat"):                     # a chat model (from a previous SFT) already knows its special tokens
            with torch.no_grad():                         # the special tokens start as an "average token"
                table = model.token_table.weight
                normal = table[:tokenizer.end_id + 1].mean(dim=0)
                for token, token_id in tokenizer.special_ids.items():
                    if token != "<|endoftext|>":
                        table[token_id] = normal + 0.01 * torch.randn_like(normal)
        log(f"=== SFT from {BASE}: {len(train_examples)} conversations, {rows_per_epoch} rows per pass, "
            f"{total_steps} steps ({EPOCHS} passes){'  PILOT: 15 minutes' if args.pilot else ''} ===")

    def learning_rate_at(current):
        if current < WARMUP_STEPS:
            return LEARNING_RATE * (current + 1) / WARMUP_STEPS
        progress = (current - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
        return LEARNING_RATE * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress))))

    def save(path):
        temporary = path + ".tmp"
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
                    "config": model.config}, temporary)
        os.replace(temporary, path)

    def save_final():
        torch.save({"config": model.config, "weights": model.state_dict(), "step": step, "chat": True},
                   os.path.join(RUN_DIR, "final.pt"))
        log(f"=== saved {RUN_DIR}/final.pt (step {step}) ===")

    @torch.no_grad()
    def evaluate():
        model.eval()
        total = 0.0
        for start in range(0, len(val_rows), MICRO_BATCH):
            inputs, targets, mask, segments = to_batch(val_rows[start:start + MICRO_BATCH], tokenizer.end_id, device)
            total += masked_loss(model, inputs, targets, mask, segments).item()
        model.train()
        return total / math.ceil(len(val_rows) / MICRO_BATCH)

    @torch.no_grad()
    def answer(question, max_tokens=80):
        model.eval()
        ids = tokenizer.encode(chat_prompt([{"role": "system", "content": system_prompt(NAME)},
                                            {"role": "user", "content": question}]))
        start = len(ids)
        for _ in range(max_tokens):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                logits, _ = model(torch.tensor([ids], device=device))
            next_id = int(logits[0, -1].float().argmax())
            if next_id == tokenizer.end_id or next_id == tokenizer.special_ids["<|user|>"]:
                break
            ids.append(next_id)
        model.train()
        return tokenizer.decode(ids[start:]).replace("\n", " / ")

    def report():
        log(f"   val loss (answers only) {evaluate():.3f}")
        for question in SAMPLE_QUESTIONS:
            log(f"   Q: {question}\n   A: {answer(question)}")

    stop = {"requested": False}

    def on_ctrl_c(signal_number, frame):
        if stop["requested"]:
            sys.exit(1)
        stop["requested"] = True
        print("\nCtrl+C: finishing this step, then saving ...", flush=True)

    signal.signal(signal.SIGINT, on_ctrl_c)

    if step == 0:
        report()                                               # how the base model answers, before SFT
    model.train()
    start_time = time.time()
    last_save = time.time()
    epoch_rows, epoch_seen = None, -1
    while step < total_steps:
        epoch = step // steps_per_epoch
        if epoch != epoch_seen:
            epoch_rows = pack_rows(train_examples, seed=1000 + epoch)
            epoch_seen = epoch
        rate = learning_rate_at(step)
        for group in optimizer.param_groups:
            group["lr"] = rate
        first_row = (step % steps_per_epoch) * MICRO_BATCH * ACCUMULATION
        total_loss = 0.0
        for micro in range(ACCUMULATION):
            rows = epoch_rows[first_row + micro * MICRO_BATCH: first_row + (micro + 1) * MICRO_BATCH]
            if not rows:
                continue
            inputs, targets, mask, segments = to_batch(rows, tokenizer.end_id, device)
            loss = masked_loss(model, inputs, targets, mask, segments)
            (loss / ACCUMULATION).backward()
            total_loss += loss.item() / ACCUMULATION
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        if step % 10 == 0:
            minutes = (time.time() - start_time) / 60
            log(f"step {step:5d}/{total_steps}  pass {epoch + 1}  loss {total_loss:.3f}  lr {rate:.1e}  "
                f"grad {norm:.2f}  {minutes:5.1f} min")
        if step % EVAL_EVERY == 0:
            report()
        pilot_over = args.pilot and time.time() - start_time > PILOT_SECONDS
        if stop["requested"] or pilot_over or time.time() - last_save > SAVE_EVERY_MINUTES * 60:
            save(CHECKPOINT)
            last_save = time.time()
            if pilot_over:
                report()
                save_final()
                return
            if stop["requested"]:
                log("stopped. Run the same command again to continue.")
                return
    report()
    save(CHECKPOINT)
    save_final()


if __name__ == "__main__":
    main()
