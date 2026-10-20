"""
Step 2 for Supernova:  python prepare_supernova.py
Streams every source, turns each document into token ids and appends them to a file on the SSD:
    C:/alexgpt_data/supernova/<source>_train.bin   (and _val.bin: every 100th document, ~2M tokens)
2 bytes per token (uint16), documents separated by <|endoftext|>. About 2 billion tokens = ~4 GB.

You can stop it any time (Ctrl+C or closing the window). Run it again and it continues where it was
(progress.json remembers how many documents of each source are done).
Takes a few hours: most of it is the download, the rest is the tokenizer.
"""
import json
import os
import sys
import time
from multiprocessing import Pool, cpu_count

import numpy as np

DATA_DIR = "C:/alexgpt_data/supernova"
TOKENIZER = "checkpoints/supernova_tokenizer.json"
BATCH_DOCS = 512                 # documents sent to the worker processes at once
VAL_EVERY = 100                  # every 100th document goes to validation ...
VAL_TOKENS = 2_000_000           # ... until a source has this many validation tokens

tokenizer = None


def keep_pc_awake():
    """Windows: don't go to sleep while this script runs (the screen may still turn off)."""
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)   # CONTINUOUS | SYSTEM_REQUIRED


def start_worker():
    global tokenizer
    from alexgpt.bpe_fast import FastBPE
    tokenizer = FastBPE.load(TOKENIZER)


def encode_batch(texts):
    # each document -> its ids + <|endoftext|>, as one uint16 array per document
    result = []
    for text in texts:
        ids = tokenizer.encode(text)
        ids.append(tokenizer.end_id)
        result.append(np.array(ids, dtype=np.uint16))
    return result


def load_progress(path):
    if os.path.exists(path):
        with open(path) as file:
            return json.load(file)
    return {}


def save_progress(path, progress):
    temporary = path + ".tmp"
    with open(temporary, "w") as file:
        json.dump(progress, file, indent=1)
    os.replace(temporary, path)                   # all or nothing, even if the PC turns off right now


def main():
    from alexgpt.data_sources import SOURCES, stream_documents
    keep_pc_awake()
    os.makedirs(DATA_DIR, exist_ok=True)
    progress_path = os.path.join(DATA_DIR, "progress.json")
    progress = load_progress(progress_path)
    workers = max(2, min(8, cpu_count() - 2))
    print(f"data folder: {DATA_DIR}   worker processes: {workers}\n")

    with Pool(workers, initializer=start_worker) as pool:
        for source, (_, _, language, wanted) in SOURCES.items():
            state = progress.setdefault(source, {"docs": 0, "train_tokens": 0, "val_tokens": 0})
            if state["train_tokens"] >= wanted:
                print(f"{source:12s} done ({state['train_tokens'] / 1e6:,.0f} M tokens)")
                continue
            print(f"{source:12s} {language}  wanted {wanted / 1e6:,.0f} M tokens, "
                  f"have {state['train_tokens'] / 1e6:,.0f} M", flush=True)
            train_file = open(os.path.join(DATA_DIR, f"{source}_train.bin"), "ab")
            val_file = open(os.path.join(DATA_DIR, f"{source}_val.bin"), "ab")
            start = time.time()
            start_tokens = state["train_tokens"]

            def batches():
                batch, numbers = [], []
                for number, text in stream_documents(source, skip=state["docs"]):
                    batch.append(text)
                    numbers.append(number)
                    if len(batch) == BATCH_DOCS:
                        yield numbers, batch
                        batch, numbers = [], []
                if batch:
                    yield numbers, batch

            pending = []
            for numbers, batch in batches():
                pending.append((numbers, pool.apply_async(encode_batch, (batch,))))
                if len(pending) < workers * 2:            # keep all workers busy
                    continue
                numbers, job = pending.pop(0)
                write_batch(numbers, job.get(), state, train_file, val_file)
                save_progress(progress_path, progress)
                report(source, state, wanted, start, start_tokens)
                if state["train_tokens"] >= wanted:
                    break
            for numbers, job in pending:                  # what is still in flight
                write_batch(numbers, job.get(), state, train_file, val_file)
            save_progress(progress_path, progress)
            train_file.close()
            val_file.close()
            print(f"\n{source:12s} finished: {state['train_tokens'] / 1e6:,.0f} M train, "
                  f"{state['val_tokens'] / 1e6:.1f} M val tokens\n", flush=True)

    total = sum(state["train_tokens"] for state in progress.values())
    print(f"all done: {total / 1e9:.2f} B training tokens in {DATA_DIR}")


def write_batch(numbers, arrays, state, train_file, val_file):
    for number, ids in zip(numbers, arrays):
        if number % VAL_EVERY == 0 and state["val_tokens"] < VAL_TOKENS:
            val_file.write(ids.tobytes())
            state["val_tokens"] += len(ids)
        else:
            train_file.write(ids.tobytes())
            state["train_tokens"] += len(ids)
        state["docs"] = max(state["docs"], number)
    train_file.flush()
    val_file.flush()


def report(source, state, wanted, start, start_tokens):
    seconds = time.time() - start
    speed = (state["train_tokens"] - start_tokens) / max(seconds, 1)
    left = (wanted - state["train_tokens"]) / max(speed, 1)
    print(f"\r  {source:12s} {state['train_tokens'] / 1e6:8,.0f} / {wanted / 1e6:,.0f} M tokens   "
          f"{speed / 1e3:6,.0f} K tokens/s   ~{left / 60:5.0f} min left for this source   ",
          end="", flush=True)


if __name__ == "__main__":          # needed on Windows: the worker processes import this file again
    try:
        main()
    except KeyboardInterrupt:
        print("\nstopped. Run it again to continue from here.")
        sys.exit(0)
