"""
Step 1 for Singularity:  python prepare_singularity.py
Reads ~2.7B NEW tokens into C:/alexgpt_data/singularity/ (2 bytes per token: ~5.5 GB on the SSD).

Every source continues exactly where Supernova stopped (it reads the document counts from
C:/alexgpt_data/supernova/progress.json), so Singularity sees new pages, not the same ones again.
fineweb_edu is split by quality: score 4-5 -> fineweb_edu_hq.bin, score 3 -> fineweb_edu.bin.

Stop any time (Ctrl+C / close the window / PC off); run it again and it continues.
Takes a few hours (download + tokenizer). It doesn't need the GPU: you can chat or train meanwhile.
"""
import json
import os
import shutil
import sys
import time
from multiprocessing import Pool, cpu_count

import numpy as np

from alexgpt.singularity_data import HQ_MIN_SCORE, HQ_SOURCE, NEW_SOURCES, SINGULARITY_DIR, SUPERNOVA_DIR
from prepare_supernova import encode_batch, keep_pc_awake, load_progress, save_progress, start_worker

BATCH_DOCS = 512
VAL_EVERY = 100
VAL_TOKENS = 2_000_000
MIN_FREE_GB = 7                  # the new data needs ~5.5 GB


def document_text(source, row):
    """One row -> (text, the file it goes to) or None to skip it."""
    text = row.get("text") or ""
    if len(text) < 200:
        return None
    if source == "fineweb_edu":
        score = row.get("int_score")
        if score is None:
            score = round(row.get("score") or 0)
        return text, (HQ_SOURCE if score >= HQ_MIN_SCORE else "fineweb_edu")
    return text, source


def stream_rows(source, skip):
    """Yields (row number, text, target file), starting after `skip` rows. Reconnects by itself."""
    from datasets import load_dataset
    path, config, _, _ = NEW_SOURCES[source]
    done = skip
    failures = 0
    while True:
        try:
            dataset = load_dataset(path, config, split="train", streaming=True)
            if done:
                print(f"  [{source}] skipping the {done:,} rows Supernova already read (takes a few minutes) ...",
                      flush=True)
                dataset = dataset.skip(done)
            for row in dataset:
                done += 1
                failures = 0
                result = document_text(source, row)
                if result is not None:
                    yield done, result[0], result[1]
            return
        except Exception as error:
            failures += 1
            if failures > 20:
                raise
            wait = min(300, 15 * failures)
            print(f"\n  [{source}] connection problem ({type(error).__name__}: {str(error)[:120]}), "
                  f"retrying in {wait} s ...", flush=True)
            time.sleep(wait)


class Output:
    """The two files (train / val) of one target + how many tokens are in them."""

    def __init__(self, name, state):
        self.state = state
        self.train = open(os.path.join(SINGULARITY_DIR, f"{name}_train.bin"), "ab")
        self.val = open(os.path.join(SINGULARITY_DIR, f"{name}_val.bin"), "ab")

    def write(self, number, ids):
        if number % VAL_EVERY == 0 and self.state["val_tokens"] < VAL_TOKENS:
            self.val.write(ids.tobytes())
            self.state["val_tokens"] += len(ids)
        else:
            self.train.write(ids.tobytes())
            self.state["train_tokens"] += len(ids)

    def close(self):
        self.train.close()
        self.val.close()


def main():
    keep_pc_awake()
    os.makedirs(SINGULARITY_DIR, exist_ok=True)
    free_gb = shutil.disk_usage(SINGULARITY_DIR).free / 1e9
    print(f"free space on the SSD: {free_gb:.1f} GB")
    if free_gb < MIN_FREE_GB:
        print(f"not enough: the new data needs ~{MIN_FREE_GB} GB. Free some space and run it again.")
        sys.exit(1)
    progress_path = os.path.join(SINGULARITY_DIR, "progress.json")
    progress = load_progress(progress_path)
    supernova_progress = load_progress(os.path.join(SUPERNOVA_DIR, "progress.json"))
    workers = max(2, min(8, cpu_count() - 2))

    with Pool(workers, initializer=start_worker) as pool:
        for source, (_, _, language, wanted) in NEW_SOURCES.items():
            if source not in progress:              # first time: start after the rows Supernova read
                progress[source] = {"docs": supernova_progress.get(source, {}).get("docs", 0),
                                    "train_tokens": 0, "val_tokens": 0}
            state = progress[source]
            targets = [source] + ([HQ_SOURCE] if source == "fineweb_edu" else [])
            for target in targets[1:]:
                progress.setdefault(target, {"docs": 0, "train_tokens": 0, "val_tokens": 0})
            if state.get("finished") or state["train_tokens"] >= wanted:
                print(f"{source:14s} done ({state['train_tokens'] / 1e6:,.0f} M tokens)")
                continue
            print(f"{source:14s} {language}  wanted {wanted / 1e6:,.0f} M new tokens, have "
                  f"{state['train_tokens'] / 1e6:,.0f} M, starting after row {state['docs']:,}", flush=True)
            outputs = {target: Output(target, progress[target]) for target in targets}
            start, start_tokens = time.time(), state["train_tokens"]

            def batches():
                numbers, texts, names = [], [], []
                for number, text, name in stream_rows(source, state["docs"]):
                    numbers.append(number)
                    texts.append(text)
                    names.append(name)
                    if len(texts) == BATCH_DOCS:
                        yield numbers, texts, names
                        numbers, texts, names = [], [], []
                if texts:
                    yield numbers, texts, names

            def write(numbers, names, arrays):
                for number, name, ids in zip(numbers, names, arrays):
                    outputs[name].write(number, ids)
                    state["docs"] = max(state["docs"], number)
                for output in outputs.values():
                    output.train.flush()
                    output.val.flush()

            pending = []
            reached_end = True
            for numbers, texts, names in batches():
                pending.append((numbers, names, pool.apply_async(encode_batch, (texts,))))
                if len(pending) < workers * 2:
                    continue
                numbers, names, job = pending.pop(0)
                write(numbers, names, job.get())
                save_progress(progress_path, progress)
                seconds = time.time() - start
                speed = (state["train_tokens"] - start_tokens) / max(seconds, 1)
                extra = ""
                if source == "fineweb_edu":
                    extra = f"  (+ {progress[HQ_SOURCE]['train_tokens'] / 1e6:,.0f} M in fineweb_edu_hq)"
                print(f"\r  {source:14s} {state['train_tokens'] / 1e6:8,.0f} / {wanted / 1e6:,.0f} M  "
                      f"{speed / 1e3:6,.0f} K tokens/s{extra}   ", end="", flush=True)
                if state["train_tokens"] >= wanted:
                    reached_end = False
                    break
            for numbers, names, job in pending:
                write(numbers, names, job.get())
            if reached_end:
                state["finished"] = True             # the dataset ran out (e.g. Romanian Wikipedia): that's all of it
            save_progress(progress_path, progress)
            for output in outputs.values():
                output.close()
            print(f"\n{source:14s} finished: {state['train_tokens'] / 1e6:,.0f} M new tokens\n", flush=True)

    print("\nall done:")
    for name, state in progress.items():
        print(f"   {name:14s} {state['train_tokens'] / 1e6:8,.0f} M train   {state['val_tokens'] / 1e6:4.1f} M val")
    total = sum(state["train_tokens"] for state in progress.values())
    print(f"   total {total / 1e9:.2f} B new tokens in {SINGULARITY_DIR}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nstopped. Run it again to continue from here.")
        sys.exit(0)
