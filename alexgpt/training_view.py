"""
For the "Live training" page: reads what train_supernova.py writes, while it trains.
    read_log()    runs/supernova/log.txt -> loss curve, validation losses, the sample texts at every check
    peek()        a random document from the training data (C:/alexgpt_data/supernova), as the model reads it
Nothing here touches the training itself: it only reads files.
"""
import os
import re
import time

import numpy as np

STEP_LINE = re.compile(r"step\s+(\d+)/(\d+)\s+loss ([\d.]+)\s+lr ([\d.e+-]+)\s+grad ([\d.]+)\s+([\d.]+)K tok/s\s+~\s*([\d.]+) h left")
VAL_LINE = re.compile(r"val loss\s+en: ([\d.]+)(?:\s+ro: ([\d.]+))?")
PROMPTS = ["The main reason the sky looks blue is", "Capitala României este", "Once upon a time, a young engineer"]
DATA_DIR = os.environ.get("ALEXGPT_DATA", "C:/alexgpt_data/supernova")


def read_log(project_root, run="supernova"):
    path = os.path.join(project_root, "runs", run, "log.txt")
    if not os.path.exists(path):
        return {"found": False}
    steps, val, samples, events = [], [], [], []
    total_steps, last_step, hours_left, speed = 0, 0, None, None
    with open(path, encoding="utf-8", errors="replace") as file:
        for line in file:
            line = line.rstrip("\n")
            match = STEP_LINE.search(line)
            if match:
                last_step, total_steps = int(match[1]), int(match[2])
                steps.append([last_step, float(match[3]), float(match[4])])
                speed, hours_left = float(match[6]), float(match[7])
                continue
            match = VAL_LINE.search(line)
            if match:
                val.append([last_step, float(match[1]), float(match[2]) if match[2] else None])
                samples.append({"step": last_step, "texts": []})
                continue
            if line.startswith("   > ") and samples:
                samples[-1]["texts"].append(line[5:])
                continue
            if line.startswith("===") or "stopped" in line or "saved" in line and "checkpoint" not in line:
                events.append([last_step, line.strip("= ")])
    modified = os.path.getmtime(path)
    return {
        "found": True, "run": run, "total_steps": total_steps, "step": last_step,
        "hours_left": hours_left, "speed": speed, "steps": steps, "val": val,
        "samples": samples, "prompts": PROMPTS, "events": events,
        "seconds_since_update": round(time.time() - modified),
        "running": time.time() - modified < 180,     # the log gets a line every ~1 minute while training
    }


class DataPeek:
    """Shows a random piece of the training data: pick a source (like the training does), a random place,
    then the whole document around it (documents are separated by <|endoftext|>)."""

    def __init__(self, project_root):
        self.project_root = project_root
        self.tokenizer = None

    def peek(self, source=None):
        from alexgpt.data_sources import SOURCES, TRAIN_MIX
        if self.tokenizer is None:
            from alexgpt.bpe_fast import FastBPE
            self.tokenizer = FastBPE.load(os.path.join(self.project_root, "checkpoints", "supernova_tokenizer.json"))
        rng = np.random.default_rng()
        available = [name for name in TRAIN_MIX if os.path.exists(os.path.join(DATA_DIR, f"{name}_train.bin"))]
        if not available:
            return {"error": f"no training data in {DATA_DIR}"}
        if source not in available:
            weights = np.array([TRAIN_MIX[name] for name in available])
            source = available[rng.choice(len(available), p=weights / weights.sum())]
        tokens = np.memmap(os.path.join(DATA_DIR, f"{source}_train.bin"), dtype=np.uint16, mode="r")
        end_id = self.tokenizer.end_id
        place = int(rng.integers(0, len(tokens) - 1))
        # walk back to the start of this document (max 4000 tokens), then forward to its end
        window = np.array(tokens[max(0, place - 4000):place])
        ends = np.nonzero(window == end_id)[0]
        start = max(0, place - 4000) + (int(ends[-1]) + 1 if len(ends) else 0)
        piece = np.array(tokens[start:start + 700])
        ends = np.nonzero(piece == end_id)[0]
        whole = len(ends) > 0
        if whole:
            piece = piece[:int(ends[0])]
        return {"source": source, "language": SOURCES[source][2], "dataset": SOURCES[source][0],
                "text": self.tokenizer.decode(piece.tolist()), "tokens": int(len(piece)),
                "complete": whole, "source_tokens": int(len(tokens)), "share": TRAIN_MIX[source]}
