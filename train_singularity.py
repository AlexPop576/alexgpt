"""
Step 2 for Singularity:  python train_singularity.py --pilot     (100M tokens, ~2 h: checks everything, memory, speed)
                         python train_singularity.py             (the real one: 3.6B tokens, ~60 h in several sessions)
                         python train_singularity.py --until 08:00

Singularity = Supernova made deeper: 12 -> 16 blocks (97.5M -> ~126M parameters, the size of GPT-2 small).
It does NOT start from zero: it starts from runs/supernova/stable.pt (Supernova before its learning rate went
down) and keeps everything Supernova learned:
  - blocks 0..3 and 8..11 are copied as they are
  - blocks 4, 5, 6, 7 are copied TWICE (one after the other): 4, 4', 5, 5', 6, 6', 7, 7'
  - in every copy (4', 5', ...) the last layer of the attention (out) and of the MLP (down) is set to 0
    -> the copy adds 0 to x:  x = x + 0  -> at step 0 Singularity answers EXACTLY like Supernova
    -> training then teaches the copies to add something useful
Data: ~2.7B new tokens (prepare_singularity.py) + Supernova's 2B again, less often (25%).

STOP ANY TIME: Ctrl+C once, or double-click stop_training.bat (creates runs/singularity/STOP): it saves and exits.
CONTINUE: the same command again.

The last 15% (learning rate going down to 0) is run twice, from the same checkpoint
(runs/singularity/singularity_pre_decay_step<N>.pt):
  python train_singularity.py              decay A: the clean text mix (ANNEAL_MIX)
  python train_singularity.py --decay-b    decay B: the same + 3.5% conversations written by people + 0.5% math
Then the better one (validation + benchmarks + chat after SFT) becomes Singularity.
Validation uses Supernova's validation files, so the numbers compare directly with Supernova
(Supernova at the end: EN 2.236, RO 2.674).
"""
import argparse
import hashlib
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
from alexgpt.data_sources import SOURCES as SUPERNOVA_SOURCES
from alexgpt.singularity_data import ANNEAL_MIX, ANNEAL_MIX_B, LANGUAGE, SINGULARITY_DIR, SUPERNOVA_DIR, TRAIN_MIX
from alexgpt.supernova import Supernova

# ---------------------------------------------------------------- settings
TOKENIZER = "checkpoints/supernova_tokenizer.json"
START_FROM = "runs/supernova/stable.pt"
SIZE, BLOCKS, HEADS, CONTEXT = 768, 16, 12, 1024
DUPLICATED_BLOCKS = [4, 5, 6, 7]   # Supernova's blocks that get a (zeroed) copy right after them
ROWS_PER_STEP = 128                # 128 windows x 1024 tokens = 131k tokens per step, like Supernova
LEARNING_RATE = 6e-4               # a bit lower than Supernova's 8e-4: deeper model, continued training
WARMUP_STEPS = 1000                # new optimizer (its memory starts empty): warm up slowly
DECAY_FRACTION = 0.15              # the last 15% of the steps: learning rate goes down to 0, cleanest data
TOTAL_TOKENS = 3_600_000_000
SAVE_EVERY_MINUTES = 30
EVAL_EVERY = 250                   # steps
BENCH_EVERY = 2000                 # HellaSwag (10,042) + PIQA (1,838), ~1.5 min: the curve towards GPT-2 (31.4% / 63.2%)
BENCH_LIMIT = 0                    # 0 = all questions
PROMPTS = ["The main reason the sky looks blue is", "Capitala României este", "Once upon a time, a young engineer"]

parser = argparse.ArgumentParser()
parser.add_argument("--pilot", action="store_true", help="short run on 100M tokens")
parser.add_argument("--until", help="stop and save at this time, e.g. 08:00")
parser.add_argument("--decay-b", action="store_true",
                    help="run the last 15% again from the pre-decay checkpoint, with ANNEAL_MIX_B")
parser.add_argument("--micro-batch", type=int, default=2,
                    help="windows on the GPU at once. 4 uses 7.1 GB: Windows starts moving memory to RAM, 12x slower")
args = parser.parse_args()

MICRO_BATCH = args.micro_batch
ACCUMULATION = ROWS_PER_STEP // MICRO_BATCH      # 4 -> 32 micro-batches per step; 2 -> 64 (same step, slower)
RUN_DIR = "runs/singularity_pilot" if args.pilot else "runs/singularity"
MAIN_RUN_DIR = RUN_DIR
if args.decay_b:
    RUN_DIR = "runs/singularity_decay_b"
if args.pilot:
    TOTAL_TOKENS, WARMUP_STEPS, EVAL_EVERY = 100_000_000, 100, 100
TOKENS_PER_STEP = MICRO_BATCH * ACCUMULATION * CONTEXT
TOTAL_STEPS = TOTAL_TOKENS // TOKENS_PER_STEP
DECAY_START = int(TOTAL_STEPS * (1 - DECAY_FRACTION))
os.makedirs(RUN_DIR, exist_ok=True)
CHECKPOINT = os.path.join(RUN_DIR, "latest.pt")
PRE_DECAY = os.path.join(MAIN_RUN_DIR, f"singularity_pre_decay_step{DECAY_START}.pt")   # fixed name, never overwritten
STOP_FILE = os.path.join(RUN_DIR, "STOP")
ANNEAL = ANNEAL_MIX_B if args.decay_b else ANNEAL_MIX
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
train_data, val_data, new_val_data = {}, {}, {}


def open_tokens(path):
    if os.path.exists(path) and os.path.getsize(path) > 2 * (CONTEXT + 1) * 2:
        return np.memmap(path, dtype=np.uint16, mode="r")             # read from disk only what is needed
    return None


for source in SUPERNOVA_SOURCES:                                       # Supernova's data: "old_<name>"
    tokens = open_tokens(os.path.join(SUPERNOVA_DIR, f"{source}_train.bin"))
    if tokens is not None:
        train_data["old_" + source] = tokens
    tokens = open_tokens(os.path.join(SUPERNOVA_DIR, f"{source}_val.bin"))
    if tokens is not None:
        val_data[source] = tokens                                      # the same validation as Supernova
for source in LANGUAGE:                                                # the new data
    tokens = open_tokens(os.path.join(SINGULARITY_DIR, f"{source}_train.bin"))
    if tokens is not None:
        train_data[source] = tokens
for source in LANGUAGE:                                                # held-out NEW documents (never trained on)
    tokens = open_tokens(os.path.join(SINGULARITY_DIR, f"{source}_val.bin"))
    if tokens is not None and source != "gsm8k":                      # chat_human_val: decay A vs B on chat
        new_val_data[source] = tokens
missing = [name for name in (ANNEAL if args.decay_b else TRAIN_MIX) if name not in train_data]
if missing:
    print(f"missing data: {missing}. Run python prepare_singularity.py first (and let it finish).")
    sys.exit(1)


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
def grown_from_supernova():
    """Supernova (12 blocks) -> 16 blocks that compute exactly the same thing (see the top of the file)."""
    saved = torch.load(START_FROM, map_location="cpu", weights_only=False)
    old_weights = saved["model"] if "model" in saved else saved["weights"]
    new_model = Supernova(tokenizer.vocab_size, CONTEXT, SIZE, BLOCKS, HEADS)
    new_weights = new_model.state_dict()
    plan = []                                      # for every new block: (old block it copies, zeroed?)
    for old_index in range(saved["config"]["blocks"]):
        plan.append((old_index, False))
        if old_index in DUPLICATED_BLOCKS:
            plan.append((old_index, True))
    assert len(plan) == BLOCKS, plan
    for name, value in old_weights.items():
        if not name.startswith("blocks."):
            new_weights[name] = value.clone()      # token table, final norm, (tied) output
    for new_index, (old_index, zeroed) in enumerate(plan):
        prefix_old, prefix_new = f"blocks.{old_index}.", f"blocks.{new_index}."
        for name, value in old_weights.items():
            if name.startswith(prefix_old):
                part = name[len(prefix_old):]
                copy = value.clone()
                if zeroed and part in ("attention.out.weight", "mlp.down.weight"):
                    copy.zero_()                   # this copy adds 0 to x, for now
                new_weights[prefix_new + part] = copy
    new_model.load_state_dict(new_weights)
    log(f"grown from {START_FROM} (step {saved['step']}): blocks " +
        " ".join(f"{old}'" if zeroed else str(old) for old, zeroed in plan))
    return new_model


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


def md5_of(path):
    digest = hashlib.md5()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


RESUME_FROM = CHECKPOINT
if args.decay_b and not os.path.exists(CHECKPOINT):
    if not os.path.exists(PRE_DECAY):
        print(f"decay B needs {PRE_DECAY}: let the main run reach step {DECAY_START} first.")
        sys.exit(1)
    RESUME_FROM = PRE_DECAY
    log(f"=== decay B: starts from {PRE_DECAY} (md5 {md5_of(PRE_DECAY)}) ===")
if os.path.exists(RESUME_FROM):
    state = torch.load(RESUME_FROM, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    step = state["step"]
    rng.bit_generator.state = state["data_rng"]
    torch.set_rng_state(state["torch_rng"].cpu())     # map_location moved it to the GPU: must be on the CPU
    torch.cuda.set_rng_state(state["cuda_rng"].cpu())
    random.setstate(state["python_rng"])
    log(f"\n=== resumed from step {step} ({datetime.datetime.now():%Y-%m-%d %H:%M}) ===")
else:
    model.load_state_dict(grown_from_supernova().state_dict())
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
EVAL_BATCHES_OLD = 20      # the first evaluation (kept: comparable with Supernova's log and with steps 0-1500)
EVAL_WINDOWS = 96          # the bigger one, per source: 96 x 1024 tokens (40 windows were too noisy: +-0.02)


@torch.no_grad()
def evaluate():
    model.eval()
    eval_rng = np.random.default_rng(0)                  # the same windows every time: comparable numbers
    results = {}
    for language in ("en", "ro"):
        sources = {name: 1.0 for name in val_data if SUPERNOVA_SOURCES[name][2] == language}
        if not sources:
            continue
        total = 0.0
        for _ in range(EVAL_BATCHES_OLD):
            inputs, targets = random_batch(eval_rng, sources, MICRO_BATCH, val_data)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = model(inputs, targets)
            total += loss.item()
        results[language] = total / EVAL_BATCHES_OLD
    model.train()
    return results


@torch.no_grad()
def evaluate_per_source():
    """Loss on every validation file separately (Supernova's + the held-out new documents), same windows each time."""
    model.eval()
    results = {}
    for prefix, data in (("", val_data), ("new_", new_val_data)):
        for name in sorted(data):
            eval_rng = np.random.default_rng(1)
            total, batches = 0.0, EVAL_WINDOWS // 4
            for _ in range(batches):
                inputs, targets = random_batch(eval_rng, {name: 1.0}, 4, data)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    _, loss = model(inputs, targets)
                total += loss.item()
            results[prefix + name] = total / batches
    model.train()
    return results


def run_benchmarks():
    """The same scoring as eval_benchmarks.py (lm-evaluation-harness rules), on the model being trained."""
    from eval_benchmarks import load_task, score_question
    model.eval()
    results = {}
    for task, metric in (("hellaswag", "acc_norm"), ("piqa", "acc")):
        questions = load_task(task)
        if BENCH_LIMIT:
            questions = questions[:BENCH_LIMIT]
        right = 0
        for question in questions:
            scores = score_question(model, tokenizer, question, device)
            if metric == "acc_norm":
                pick = max(range(len(scores)), key=lambda i: scores[i][0] / scores[i][1])
            else:
                pick = max(range(len(scores)), key=lambda i: scores[i][0])
            right += pick == question["label"]
        results[f"{task} {metric}"] = 100 * right / len(questions)
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
if step == 0:                                            # must be Supernova's numbers: nothing lost by growing
    losses = evaluate()
    log("   val loss at step 0 (= Supernova stable.pt)  " + "   ".join(f"{k}: {v:.3f}" for k, v in losses.items()))
model.train()
last_save = time.time()
last_report = time.time()
tokens_since_report = 0
while step < TOTAL_STEPS:
    if step == DECAY_START and not args.decay_b and not os.path.exists(PRE_DECAY):
        save(PRE_DECAY)                                  # decay A and decay B both start from here
        log(f"saved {PRE_DECAY} (md5 {md5_of(PRE_DECAY)}), before the learning rate goes down")
    mix = ANNEAL if step >= DECAY_START else TRAIN_MIX
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
        if step == 10:
            log(f"   GPU memory used at most: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB "
                f"(above ~6.5 GB Windows slows everything down)")
        last_report, tokens_since_report = time.time(), 0

    if step % EVAL_EVERY == 0:
        losses = evaluate()
        log("   val loss  " + "   ".join(f"{language}: {value:.3f}" for language, value in losses.items()))
        if step % (EVAL_EVERY * 4) == 0 or step == TOTAL_STEPS or step == DECAY_START:
            per_source = evaluate_per_source()
            log("   val per source  " + "  ".join(f"{name} {value:.3f}" for name, value in per_source.items()))
    if step % BENCH_EVERY == 0 or step == DECAY_START or step == TOTAL_STEPS:
        try:
            log("   benchmarks  " + "   ".join(f"{name} {value:.1f}%" for name, value in run_benchmarks().items())
                + "   (GPT-2 small: hellaswag 31.4%, piqa 63.2%)")
        except Exception as error:                        # a benchmark problem must never stop the training
            log(f"   benchmarks failed: {type(error).__name__}: {error}")
        for prompt in PROMPTS:
            log("   > " + sample(prompt))

    out_of_time = stop_at is not None and datetime.datetime.now() >= stop_at
    if os.path.exists(STOP_FILE):                        # stop_training.bat
        os.remove(STOP_FILE)
        stop_requested = True
    if stop_requested or out_of_time or time.time() - last_save > SAVE_EVERY_MINUTES * 60:
        save(CHECKPOINT)
        last_save = time.time()
        log(f"   checkpoint saved at step {step}")
        if stop_requested or out_of_time:
            log("stopped. Run the same command again to continue.")
            sys.exit(0)

save(CHECKPOINT)
final_path = os.path.join(RUN_DIR, "final.pt")
torch.save({"config": model.config, "weights": model.state_dict(), "step": step}, final_path)
per_source = evaluate_per_source()
log("   final val per source  " + "  ".join(f"{name} {value:.3f}" for name, value in per_source.items()))
log(f"=== finished: {final_path} (md5 {md5_of(final_path)}) ===")
