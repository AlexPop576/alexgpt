"""
Trains Galaxy on the GPU:  python train_galaxy.py
Saves the best model (lowest val loss) to checkpoints/galaxy.pt, so you can stop it any time (Ctrl+C).
"""
import math
import time
import numpy as np
import torch
from alexgpt.galaxy import Galaxy
from alexgpt.bpe import BPETokenizer

# ---------------- settings
CONTEXT = 256          # tokens the model sees (about 850 characters)
SIZE = 512             # numbers per token vector
BLOCKS = 8
HEADS = 8
DROPOUT = 0.1
BATCH = 48             # windows per step
MAX_STEPS = 30000
LEARNING_RATE = 6e-4
WARMUP = 500           # steps with a growing learning rate at the start
EVAL_EVERY = 500
SAMPLE_EVERY = 25      # show a short reply every 25 steps, so you can watch it learn
SAMPLE_TOKENS = 30
PROMPT = "**Child**: Mom, why is the sky blue?\n**Mom**:"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)
torch.manual_seed(0)

tokenizer = BPETokenizer.load("checkpoints/galaxy_tokenizer.json")
train_ids = torch.from_numpy(np.load("data/galaxy_train.npy").astype(np.int64))
val_ids = torch.from_numpy(np.load("data/galaxy_val.npy").astype(np.int64))
print(f"tokens: {len(train_ids):,} train, {len(val_ids):,} val")


def random_batch(source):
    # BATCH random windows; targets = the same windows moved one token to the right
    starts = torch.randint(0, len(source) - CONTEXT - 1, (BATCH,))
    inputs = torch.stack([source[start : start + CONTEXT] for start in starts])
    targets = torch.stack([source[start + 1 : start + CONTEXT + 1] for start in starts])
    return inputs.to(device), targets.to(device)


model = Galaxy(tokenizer.vocab_size, CONTEXT, SIZE, BLOCKS, HEADS, DROPOUT).to(device)
print(f"parameters: {model.num_params() / 1e6:.1f} M")
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.1, betas=(0.9, 0.95))
use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()


def learning_rate_at(step):
    # warm up, then slowly go down (cosine) to 10% of the start
    if step < WARMUP:
        return LEARNING_RATE * (step + 1) / WARMUP
    progress = (step - WARMUP) / (MAX_STEPS - WARMUP)
    return LEARNING_RATE * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


@torch.no_grad()
def average_loss(source, batches=40):
    model.eval()
    total = 0.0
    for _ in range(batches):
        inputs, targets = random_batch(source)
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_bf16):
            _, loss = model(inputs, targets)
        total += loss.item()
    model.train()
    return total / batches


@torch.no_grad()
def sample(prompt, tokens=80, temperature=0.8):
    model.eval()
    # the same "dice" every time: if the reply changes, it is because the model learned, not luck
    dice = torch.Generator(device=device).manual_seed(123)
    ids = torch.tensor([tokenizer.encode(prompt)], device=device)
    for _ in range(tokens):
        logits, _ = model(ids[:, -CONTEXT:])
        probs = torch.softmax(logits[0, -1] / temperature, dim=-1)
        next_id = torch.multinomial(probs, 1, generator=dice)
        ids = torch.cat([ids, next_id.view(1, 1)], dim=1)
        if next_id.item() == tokenizer.end_id:
            break
    model.train()
    return tokenizer.decode(ids[0].tolist())


best_val = float("inf")
start_time = time.time()
for step in range(MAX_STEPS):
    for group in optimizer.param_groups:
        group["lr"] = learning_rate_at(step)
    inputs, targets = random_batch(train_ids)
    with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_bf16):
        _, loss = model(inputs, targets)
    optimizer.zero_grad(set_to_none=True)        # forget the blame from the last step
    loss.backward()                              # the same backprop you wrote, done by PyTorch
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()                             # update every weight

    if step % SAMPLE_EVERY == 0:
        reply = sample(PROMPT, SAMPLE_TOKENS)[len(PROMPT):].split("\n**")[0].replace("\n", " ")
        line = f"step {step:6d}   loss {loss.item():.3f}   {time.time() - start_time:5.0f}s   Mom:{reply}"
        print(line, flush=True)
        with open("checkpoints/galaxy_samples.txt", "a", encoding="utf-8") as log:
            log.write(line + "\n")
    if step % EVAL_EVERY == 0 and step > 0:
        val_loss = average_loss(val_ids)
        print(f"   val loss {val_loss:.3f}", end="")
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"config": model.config, "weights": model.state_dict(), "val_loss": val_loss, "step": step},
                       "checkpoints/galaxy.pt")
            print("   saved (best so far)", end="")
        print()
        print("   " + sample(PROMPT).replace("\n", "\n   "))

print(f"done. best val loss {best_val:.3f}")
