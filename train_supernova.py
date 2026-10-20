"""
Step 3 for Supernova:  python train_supernova.py            (the real training, ~26 h in several sessions)
                       python train_supernova.py --pilot    (100M tokens, ~1.5 h: checks that everything works)
                       python train_supernova.py --until 08:00   (stops by itself at 08:00)

STOP ANY TIME: press Ctrl+C once. It finishes the current step, saves everything and exits.
               (Ctrl+C twice = exit right away, you lose at most the last 30 minutes.)
CONTINUE:      run the same command again. It finds the checkpoint and goes on from the same step.

Learning rate = WSD: warm up -> stay constant (most of the time) -> go down to 0 at the end (last 15%).
Right before going down, a copy is saved once as runs/supernova/stable.pt: Singularity starts from it.
"""
import argparse
import datetime
import math
import os
import random
import signal
import sys
import time

import numpy as np
import torch

from alexgpt.bpe_fast import FastBPE
from alexgpt.data_sources import SOURCES, TRAIN_MIX, ANNEAL_MIX
from alexgpt.supernova import Supernova

# ---------------------------------------------------------------- settings
DATA_DIR = "C:/alexgpt_data/supernova"
TOKENIZER = "checkpoints/supernova_tokenizer.json"
SIZE, BLOCKS, HEADS, CONTEXT = 768, 12, 12, 1024
MICRO_BATCH = 4                    # windows of 1024 tokens that fit in the GPU at once (from the benchmark)
ACCUMULATION = 32                  # micro-batches added together before one update -> 131k tokens per step
LEARNING_RATE = 8e-4
WARMUP_STEPS = 300
DECAY_FRACTION = 0.15              # the last 15% of the steps: learning rate goes down to 0
TOTAL_TOKENS = 2_000_000_000
SAVE_EVERY_MINUTES = 30
EVAL_EVERY = 250                   # steps
PROMPTS = ["The main reason the sky looks blue is", "Capitala României este", "Once upon a time, a young engineer"]

parser = argparse.ArgumentParser()
parser.add_argument("--pilot", action="store_true", help="short run on 100M tokens")
parser.add_argument("--until", help="stop and save at this time, e.g. 08:00")
args = parser.parse_args()

RUN_DIR = "runs/supernova_pilot" if args.pilot else "runs/supernova"
if args.pilot:
    TOTAL_TOKENS, WARMUP_STEPS, EVAL_EVERY = 100_000_000, 50, 100
TOKENS_PER_STEP = MICRO_BATCH * ACCUMULATION * CONTEXT
TOTAL_STEPS = TOTAL_TOKENS // TOKENS_PER_STEP
DECAY_START = int(TOTAL_STEPS * (1 - DECAY_FRACTION))
os.makedirs(RUN_DIR, exist_ok=True)
CHECKPOINT = os.path.join(RUN_DIR, "latest.pt")
STABLE = os.path.join(RUN_DIR, "stable.pt")
LOG = os.path.join(RUN_DIR, "log.txt")

if sys.platform == "win32":                            # don't let Windows go to sleep while training
    import ctypes
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)

device = "cuda"
torch.cuda.set_per_process_memory_fraction(0.92)     # never borrow normal RAM (that froze the PC)
torch.backends.cuda.matmul.allow_tf32 = True


def log(text):
    print(text, flush=True)
    with open(LOG, "a", encoding="utf-8") as file:
        file.write(text + "\n")


# ---------------------------------------------------------------- data
tokenizer = FastBPE.load(TOKENIZER)
train_data, val_data = {}, {}
for source in SOURCES:
    for split, store in (("train", train_data), ("val", val_data)):
        path = os.path.join(DATA_DIR, f"{source}_{split}.bin")
        if os.path.exists(path) and os.path.getsize(path) > 2 * (CONTEXT + 1) * 2:
            store[source] = np.memmap(path, dtype=np.uint16, mode="r")   # read from disk only what is needed


def random_batch(rng, mix, batch_size, data):
    # every window: pick a source (by the mix), then a random place in it
    names = [name for name in mix if name in data]
    weights = np.array([mix[name] for name in names])
    weights = weights / weights.sum()
    rows = []
    for _ in range(batch_size):
        source = names[rng.choice(len(names), p=weights)]
        tokens = data[source]
        start = int(rng.integers(0, len(tokens) - CONTEXT - 1))
        rows.append(np.array(tokens[start:start + CONTEXT + 1], dtype=np.int64))
    batch = torch.from_numpy(np.stack(rows)).to(device, non_blocking=True)
    return batch[:, :-1], batch[:, 1:]


def learning_rate_at(step):
    if step < WARMUP_STEPS:
        return LEARNING_RATE * (step + 1) / WARMUP_STEPS
    if step < DECAY_START:
        return LEARNING_RATE
    return LEARNING_RATE * (1 - (step - DECAY_START) / max(1, TOTAL_STEPS - DECAY_START))


# ---------------------------------------------------------------- model + optimizer
model = Supernova(tokenizer.vocab_size, CONTEXT, SIZE, BLOCKS, HEADS).to(device)
decay, no_decay = [], []
for name, parameter in model.named_parameters():
    (decay if parameter.dim() >= 2 else no_decay).append(parameter)    # no weight decay on norms
optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
                              lr=LEARNING_RATE, betas=(0.9, 0.95), fused=True)
rng = np.random.default_rng(1234)
step = 0


def save(path):
    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
             "config": model.config, "data_rng": rng.bit_generator.state,
             "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(),
             "python_rng": random.getstate()}
    temporary = path + ".tmp"
    torch.save(state, temporary)
    os.replace(temporary, path)            # the old file is replaced only when the new one is complete


if os.path.exists(CHECKPOINT):
    state = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    step = state["step"]
    rng.bit_generator.state = state["data_rng"]
    torch.set_rng_state(state["torch_rng"])
    torch.cuda.set_rng_state(state["cuda_rng"])
    random.setstate(state["python_rng"])
    log(f"\n=== resumed from step {step} ({datetime.datetime.now():%Y-%m-%d %H:%M}) ===")
else:
    log(f"=== new run: {model.num_params() / 1e6:.1f}M parameters, {TOTAL_STEPS} steps x "
        f"{TOKENS_PER_STEP:,} tokens = {TOTAL_TOKENS / 1e9:.2f}B tokens ===")
    log(f"sources: " + ", ".join(f"{name} {len(tokens) / 1e6:.0f}M" for name, tokens in train_data.items()))


# ---------------------------------------------------------------- stopping
stop_requested = False


def on_ctrl_c(signal_number, frame):
    global stop_requested
    if stop_requested:
        print("\nsecond Ctrl+C: exiting without saving")
        sys.exit(1)
    stop_requested = True
    print("\nCtrl+C: finishing this step, then saving and exiting ...", flush=True)


signal.signal(signal.SIGINT, on_ctrl_c)
stop_at = None
if args.until:
    hour, minute = map(int, args.until.split(":"))
    stop_at = datetime.datetime.now().replace(hour=hour, minute=minute, second=0)
    if stop_at < datetime.datetime.now():
        stop_at += datetime.timedelta(days=1)
    log(f"will stop at {stop_at:%Y-%m-%d %H:%M}")


# ---------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate():
    model.eval()
    eval_rng = np.random.default_rng(0)                  # the same windows every time: comparable numbers
    results = {}
    for language in ("en", "ro"):
        sources = {name: 1.0 for name in val_data if SOURCES[name][2] == language}
        if not sources:
            continue
        total = 0.0
        for _ in range(20):
            inputs, targets = random_batch(eval_rng, sources, MICRO_BATCH, val_data)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = model(inputs, targets)
            total += loss.item()
        results[language] = total / 20
    model.train()
    return results


@torch.no_grad()
def sample(prompt, tokens=40, temperature=0.8):
    model.eval()
    ids = torch.tensor([tokenizer.encode(prompt)], device=device)
    generator = torch.Generator(device=device).manual_seed(7)
    for _ in range(tokens):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, _ = model(ids[:, -CONTEXT:])
        probs = torch.softmax(logits[0, -1].float() / temperature, dim=-1)
        next_id = torch.multinomial(probs, 1, generator=generator)
        if next_id.item() == tokenizer.end_id:
            break
        ids = torch.cat([ids, next_id.view(1, 1)], dim=1)
    model.train()
    return tokenizer.decode(ids[0].tolist()).replace("\n", " ")


# ---------------------------------------------------------------- training
model.train()
last_save = time.time()
last_report = time.time()
tokens_since_report = 0
while step < TOTAL_STEPS:
    if step == DECAY_START and not os.path.exists(STABLE):
        save(STABLE)                                     # Singularity starts from here
        log(f"saved {STABLE} (before the learning rate goes down)")
    mix = ANNEAL_MIX if step >= DECAY_START else TRAIN_MIX
    rate = learning_rate_at(step)
    for group in optimizer.param_groups:
        group["lr"] = rate

    total_loss = 0.0
    for _ in range(ACCUMULATION):
        inputs, targets = random_batch(rng, mix, MICRO_BATCH, train_data)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = model(inputs, targets)
        (loss / ACCUMULATION).backward()                 # gradients add up over the 32 micro-batches
        total_loss += loss.item() / ACCUMULATION
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    step += 1
    tokens_since_report += TOKENS_PER_STEP

    if step % 10 == 0:
        seconds = time.time() - last_report
        speed = tokens_since_report / seconds
        left_hours = (TOTAL_STEPS - step) * TOKENS_PER_STEP / speed / 3600
        log(f"step {step:6d}/{TOTAL_STEPS}  loss {total_loss:.3f}  lr {rate:.2e}  grad {norm:.2f}  "
            f"{speed / 1000:5.1f}K tok/s  ~{left_hours:5.1f} h left")
        last_report, tokens_since_report = time.time(), 0

    if step % EVAL_EVERY == 0:
        losses = evaluate()
        log("   val loss  " + "   ".join(f"{language}: {value:.3f}" for language, value in losses.items()))
        for prompt in PROMPTS:
            log("   > " + sample(prompt))

    out_of_time = stop_at is not None and datetime.datetime.now() >= stop_at
    if stop_requested or out_of_time or time.time() - last_save > SAVE_EVERY_MINUTES * 60:
        save(CHECKPOINT)
        last_save = time.time()
        log(f"   checkpoint saved at step {step}")
        if stop_requested or out_of_time:
            log("stopped. Run the same command again to continue.")
            sys.exit(0)

save(CHECKPOINT)
torch.save({"config": model.config, "weights": model.state_dict(), "step": step},
           os.path.join(RUN_DIR, "final.pt"))
log(f"=== finished: {RUN_DIR}/final.pt ===")
