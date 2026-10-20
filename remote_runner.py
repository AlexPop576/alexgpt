"""
Lets Claude start Python scripts on this PC while you are away.
    python remote_runner.py        (leave the window open; Win+L is fine; Ctrl+C stops it)

How it works:
  - while it runs, Windows does not go to sleep (the screen may still turn off)
  - every 5 seconds it looks in the folder jobs/ for a new file  <name>.cmd  with ONE line, e.g.
        python prepare_sft.py
  - it runs that line with this same Python (your .venv), in this folder, one job at a time
  - everything the script prints goes to  jobs/<name>.log ; at the end  jobs/<name>.done  gets the exit code
  - a file jobs/<name>.stop stops that job
Safety: it ONLY runs "python <a .py file inside this project folder> <arguments>". Nothing else
(no pip, no del, no paths outside the project). Only you and Claude can write into jobs/.
"""
import os
import subprocess
import sys
import time

PROJECT = os.path.dirname(os.path.abspath(__file__))
JOBS = os.path.join(PROJECT, "jobs")


def keep_pc_awake():
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)   # CONTINUOUS | SYSTEM_REQUIRED


def parse(line):
    """"python prepare_sft.py ro_dolly" -> [this python, -u, prepare_sft.py, ro_dolly], or None if not allowed"""
    words = line.strip().split()
    if len(words) < 2 or words[0] != "python" or not words[1].endswith(".py"):
        return None
    script = os.path.abspath(os.path.join(PROJECT, words[1]))
    if not script.startswith(PROJECT + os.sep) or not os.path.exists(script):
        return None
    for word in words[2:]:
        if any(char in word for char in "&|<>;`$"):
            return None
    return [sys.executable, "-u", script] + words[2:]


def log(text):
    print(time.strftime("%H:%M:%S  ") + text, flush=True)


def run(name, command):
    log_path = os.path.join(JOBS, name + ".log")
    stop_path = os.path.join(JOBS, name + ".stop")
    log("start  " + " ".join(command[2:]))
    with open(log_path, "w", encoding="utf-8") as output:
        process = subprocess.Popen(command, cwd=PROJECT, stdout=output, stderr=subprocess.STDOUT,
                                   env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        while process.poll() is None:
            if os.path.exists(stop_path):
                process.terminate()
                log("stopped by a .stop file")
                break
            time.sleep(2)
        code = process.wait()
    with open(os.path.join(JOBS, name + ".done"), "w") as done:
        done.write(str(code))
    log(f"end    {name}  exit code {code}")


def main():
    keep_pc_awake()
    os.makedirs(JOBS, exist_ok=True)
    log(f"waiting for jobs in {JOBS}  (the PC will not sleep while this runs)")
    while True:
        for file_name in sorted(os.listdir(JOBS)):
            if not file_name.endswith(".cmd"):
                continue
            name = file_name[:-4]
            if os.path.exists(os.path.join(JOBS, name + ".done")):
                continue
            with open(os.path.join(JOBS, file_name), encoding="utf-8") as file:
                line = file.readline()
            command = parse(line)
            if command is None:
                log(f"refused {file_name}: {line.strip()!r} (only 'python <script in this project>')")
                with open(os.path.join(JOBS, name + ".done"), "w") as done:
                    done.write("refused")
                continue
            run(name, command)
        time.sleep(5)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("runner stopped")
