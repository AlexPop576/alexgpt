"""
SFT step 1:  python prepare_sft.py
Downloads the conversations for the chat training and saves them in ONE simple format:
    data/sft/<source>.jsonl     one conversation per line:
    {"messages": [{"role": "user", "content": "Salut!"}, {"role": "assistant", "content": "Bună! ..."}]}

You can close the window or turn off the PC any time: run it again and it continues where it was
(every conversation is written to the file the moment it is ready).
Then look at what it downloaded:  python show_sft_data.py
"""
import json
import os
import sys
import time

OUT_DIR = "data/sft"

# name: (dataset on Hugging Face, config, language, how many conversations we keep, what it teaches)
SOURCES = {
    "smol_everyday":   ("HuggingFaceTB/smoltalk", "everyday-conversations", "en", 3000,  "small talk: hi, how are you, follow-up questions"),
    "smol_general":    ("HuggingFaceTB/smoltalk", "smol-magpie-ultra",      "en", 40000, "questions and answers on every topic, with markdown"),
    "smol_summarize":  ("HuggingFaceTB/smoltalk", "smol-summarize",         "en", 10000, "summaries of a given text"),
    "smol_rewrite":    ("HuggingFaceTB/smoltalk", "smol-rewrite",           "en", 6000,  "rewrite / fix a given text"),
    "smol_constraints": ("HuggingFaceTB/smoltalk", "smol-constraints",      "en", 6000,  "follow instructions exactly (3 bullets, one word, ...)"),
    "ro_ultrachat":    ("OpenLLM-Ro/ro_sft_ultrachat", None, "ro", 70000, "Romanian conversations (translated); 2/3 say \"Ca model de limbaj AI\", build_sft_mix.py throws those out"),
    "ro_alpaca":       ("OpenLLM-Ro/ro_sft_alpaca",    None, "ro", 10000, "Romanian instructions (translated)"),
    "ro_dolly":        ("OpenLLM-Ro/ro_sft_dolly",     None, "ro", 8000,  "Romanian questions, some with a text (translated)"),
    "ro_norobots":     ("OpenLLM-Ro/ro_sft_norobots",  None, "ro", 9500,  "Romanian, written by people (translated)"),
    # added for SFT v2 (natural conversations written by people, not by a model)
    "en_oasst":        ("OpenAssistant/oasst_top1_2023-08-25", None, "mixed", 13000, "OpenAssistant: real people + the best-rated answers (all languages; build_sft_mix keeps EN)"),
    "ro_oasst":        ("OpenLLM-Ro/ro_sft_oasst",     None, "ro", 10000, "OpenAssistant translated to Romanian"),
}


def keep_pc_awake():
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)


def to_messages(row):
    """Every dataset stores a conversation a bit differently. -> [{"role", "content"}, ...] or None."""
    for key in ("messages", "conversation", "conversations"):
        turns = row.get(key)
        if isinstance(turns, list) and turns and isinstance(turns[0], dict):
            messages = []
            for turn in turns:
                role = turn.get("role") or turn.get("from")
                content = turn.get("content") if "content" in turn else turn.get("value")
                role = {"human": "user", "gpt": "assistant", "model": "assistant", "bot": "assistant",
                        "prompter": "user"}.get(role, role)
                if role in ("system", "user", "assistant") and isinstance(content, str):
                    messages.append({"role": role, "content": content.strip()})
            return messages
    # ChatML text (OpenAssistant top1): "<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n...<|im_end|>"
    text = row.get("text")
    if isinstance(text, str) and "<|im_start|>" in text:
        messages = []
        for part in text.split("<|im_start|>")[1:]:
            role, _, content = part.partition("\n")
            content = content.replace("<|im_end|>", "").strip()
            if role.strip() in ("system", "user", "assistant"):
                messages.append({"role": role.strip(), "content": content})
        return messages
    # UltraChat style: "data" = [user, assistant, user, assistant, ...] as plain strings
    turns = row.get("data")
    if isinstance(turns, list) and len(turns) >= 2 and all(isinstance(turn, str) for turn in turns):
        messages = []
        for index, text in enumerate(turns):
            messages.append({"role": "user" if index % 2 == 0 else "assistant", "content": text.strip()})
        return messages
    # instruction style: instruction (+ input / context) -> output / response
    instruction = row.get("instruction") or row.get("prompt") or row.get("question")
    answer = row.get("output") or row.get("response") or row.get("answer")
    if isinstance(instruction, str) and isinstance(answer, str):
        extra = row.get("input") or row.get("context") or ""
        user = instruction.strip() + ("\n\n" + extra.strip() if isinstance(extra, str) and extra.strip() else "")
        return [{"role": "user", "content": user}, {"role": "assistant", "content": answer.strip()}]
    return None


def good(messages):
    """at least one user -> assistant pair, nothing empty, not absurdly long (our context is 1024 tokens)"""
    if not messages or not any(m["role"] == "assistant" for m in messages):
        return False
    if any(not m["content"] for m in messages):
        return False
    return sum(len(m["content"]) for m in messages) < 6000          # ~1500 tokens at most


def count_good_lines(path):
    """How many conversations are already saved. A line cut in half (PC turned off) is removed."""
    if not os.path.exists(path):
        return 0
    lines = []
    with open(path, encoding="utf-8") as file:
        for line in file:
            try:
                json.loads(line)
                lines.append(line if line.endswith("\n") else line + "\n")
            except json.JSONDecodeError:
                break
    with open(path, "w", encoding="utf-8") as file:
        file.writelines(lines)
    return len(lines)


def download(source):
    from datasets import load_dataset
    path_name, config, language, wanted, _ = SOURCES[source]
    out_path = os.path.join(OUT_DIR, f"{source}.jsonl")
    progress_path = os.path.join(OUT_DIR, f"{source}.progress")
    have = count_good_lines(out_path)
    if have >= wanted:
        print(f"{source:17s} done ({have} conversations)")
        return
    rows_read = int(open(progress_path).read()) if os.path.exists(progress_path) and have else 0
    print(f"{source:17s} {language}  have {have}/{wanted}, reading {path_name} {config or ''} ...", flush=True)
    failures = 0
    while have < wanted:
        try:
            dataset = load_dataset(path_name, config, split="train", streaming=True)
            if rows_read:
                dataset = dataset.skip(rows_read)
            with open(out_path, "a", encoding="utf-8") as file:
                for row in dataset:
                    rows_read += 1
                    if rows_read == 1:                               # keep one raw row, to see the format
                        with open(os.path.join(OUT_DIR, f"{source}.sample.json"), "w", encoding="utf-8") as sample:
                            json.dump(row, sample, ensure_ascii=False, indent=1, default=str)
                    messages = to_messages(row)
                    if messages and good(messages):
                        file.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
                        file.flush()
                        have += 1
                        with open(progress_path, "w") as progress:     # so a restart never saves it twice
                            progress.write(str(rows_read))
                    if rows_read % 200 == 0:
                        with open(progress_path, "w") as progress:
                            progress.write(str(rows_read))
                        print(f"\r  {have:6d}/{wanted}", end="", flush=True)
                    if have >= wanted:
                        break
                    if have == 0 and rows_read >= 500:               # a format I don't understand: stop wasting time
                        print(f"\n  {source}: 500 rows, none understood. Columns: {list(row.keys())}. "
                              f"Skipping it; the first row is in {OUT_DIR}/{source}.sample.json")
                        with open(progress_path, "w") as progress:
                            progress.write("0")
                        return
            with open(progress_path, "w") as progress:
                progress.write(str(rows_read))
            break                                           # read to the end (or got enough)
        except Exception as error:
            failures += 1
            name = type(error).__name__
            if "NotFound" in name or name in ("ValueError", "KeyError", "FileNotFoundError"):
                print(f"\n  {source}: can't open it ({name}: {str(error)[:200]}). Skipping, tell Claude.")
                return
            if failures > 8:
                print(f"\n  {source}: giving up for now ({type(error).__name__}: {str(error)[:200]})")
                return
            wait = 15 * failures
            print(f"\n  {source}: problem ({type(error).__name__}: {str(error)[:150]}), retrying in {wait} s", flush=True)
            time.sleep(wait)
    if have == 0:
        print(f"\n  {source}: 0 conversations: its format is not one I know. Tell Claude what show_sft_data says.")
    print(f"\r  {source}: {have} conversations saved in {out_path}            ")


def main():
    keep_pc_awake()
    os.makedirs(OUT_DIR, exist_ok=True)
    only = sys.argv[1:]                                     # python prepare_sft.py ro_dolly  -> only that one
    for source in SOURCES:
        if not only or source in only:
            download(source)
    print("\nall done. Look at them:  python show_sft_data.py")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nstopped. Run it again to continue.")
