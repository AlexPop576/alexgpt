"""
Live progress of the Singularity training:  double-click show_progress.bat
Reads runs/singularity/log.txt every 5 seconds. Closing this window does NOT stop the training.
"""
import os
import re
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = [os.path.join(HERE, "runs", "singularity", "log.txt"), os.path.join(HERE, "runs", "singularity_decay_b", "log.txt")]
LOG = LOGS[0]
STEP = re.compile(r"step\s+(\d+)/(\d+)\s+loss ([\d.]+).*?([\d.]+)K tok/s\s+~\s*([\d.]+) h left")
VAL = re.compile(r"val loss at step (\d+).*?en: ([\d.]+)\s+ro: ([\d.]+)")


def read_log():
    with open(LOG, encoding="utf-8", errors="replace") as file:
        return file.read().splitlines()


def show():
    global LOG
    LOG = max((path for path in LOGS if os.path.exists(path)), key=os.path.getmtime)   # the run that is active
    lines = read_log()
    last = None
    vals = []
    for line in lines:
        match = STEP.search(line)
        if match:
            last = match
        match = VAL.search(line)
        if match:
            vals.append(match.groups())
    os.system("cls" if os.name == "nt" else "clear")
    print(f"=== Singularity training: {os.path.basename(os.path.dirname(LOG))} (closing this window does not stop it) ===\n")
    if last is None:
        print("no step yet")
        return
    step, total, loss, speed, hours = int(last[1]), int(last[2]), last[3], last[4], float(last[5])
    percent = 100 * step / total
    bar = "#" * int(percent / 2) + "-" * (50 - int(percent / 2))
    print(f"[{bar}] {percent:5.1f} %")
    print(f"step {step:,} / {total:,}   loss {loss}   {speed}K tokens/s")
    print(f"time left: ~{hours:.1f} h  (~{hours / 24:.1f} days)")
    print(f"log updated {int(time.time() - os.path.getmtime(LOG))} s ago"
          + ("   <-- if this is > 300 the training is NOT running" if time.time() - os.path.getmtime(LOG) > 300 else ""))
    if vals:
        print("\nvalidation loss (lower = better; Supernova was EN 2.236 / RO 2.674):")
        for step_number, en, ro in vals[-5:]:
            print(f"   step {step_number:>6}   en {en}   ro {ro}")
    print("\n(refreshes every 5 s; Ctrl+C closes only this viewer)")


if __name__ == "__main__":
    try:
        while True:
            show()
            time.sleep(5)
    except KeyboardInterrupt:
        pass
