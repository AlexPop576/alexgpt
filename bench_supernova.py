"""
Speed test for Supernova on your GPU:  python bench_supernova.py
For every model size: finds the biggest batch that fits in memory, runs a few real training steps
(forward + backward + AdamW, bf16) on random tokens, and prints tokens per second and how many
hours 1 and 2 billion tokens would take. Takes about 5-10 minutes. Nothing is saved.
"""
import time
import torch
from alexgpt.supernova import Supernova

VOCAB = 16384
CONTEXT = 1024
SIZES = [            # name, size, blocks, heads
    ("small  ", 512, 10, 8),
    ("medium ", 640, 12, 10),
    ("large  ", 768, 12, 12),
]
STEPS = 12

device = "cuda"
# On Windows, when the GPU memory is full the driver silently borrows normal RAM ("shared GPU memory"):
# no error, but everything becomes ~50x slower and the whole PC freezes. This cap makes PyTorch stop
# at 90% of the real GPU memory and raise an out-of-memory error instead, which we catch below.
torch.cuda.set_per_process_memory_fraction(0.90)
LIMIT_GB = 0.90 * torch.cuda.get_device_properties(0).total_memory / 1e9
print(torch.cuda.get_device_name(0), f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"vocabulary {VOCAB}, context {CONTEXT}\n")
print("model     params   batch   tokens/s   memory    1B tokens   2B tokens")


def try_run(size, blocks, heads, batch):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = Supernova(VOCAB, CONTEXT, size, blocks, heads).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=True)
    ids = torch.randint(0, VOCAB, (batch, CONTEXT + 1), device=device)
    inputs, targets = ids[:, :-1], ids[:, 1:]
    for step in range(STEPS + 2):
        if step == 2:                                   # the first 2 steps warm the GPU up, not measured
            torch.cuda.synchronize()
            start = time.time()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = model(inputs, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    seconds = time.time() - start
    tokens_per_second = STEPS * batch * CONTEXT / seconds
    memory = torch.cuda.max_memory_allocated() / 1e9
    params = model.num_params()
    del model, optimizer
    return params, tokens_per_second, memory


for name, size, blocks, heads in SIZES:
    best = None
    for batch in (2, 4, 8, 12, 16):                     # small first: grow until the memory is nearly full
        try:
            params, speed, memory = try_run(size, blocks, heads, batch)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            break
        best = (batch, params, speed, memory)
        if memory > 0.8 * LIMIT_GB:                     # the next batch would not fit
            break
    if best is None:
        print(f"{name} does not fit even with batch 2")
        continue
    batch, params, speed, memory = best
    hours_1b = 1e9 / speed / 3600
    print(f"{name} {params / 1e6:6.1f}M   {batch:4d}   {speed:8,.0f}   {memory:5.1f} GB   "
          f"{hours_1b:7.1f} h   {2 * hours_1b:7.1f} h", flush=True)

print("\n(a bigger effective batch comes from gradient accumulation: several of these batches per update)")
