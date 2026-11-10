"""Translates English conversations into Romanian with a LOCAL model (EuroLLM through Ollama).
    python translate_local.py sft_data/ext_en_C.txt            -> sft_data/ext_en_C_ro.txt   (same ### / USER: / ASSISTANT: format)
Then:  python import_conversations.py sft_data/ext_en_C_ro.txt
A translated conversation is dropped if the number of ``` fences changed, a number disappeared, or the length looks wrong.
Safe to stop and run again: conversations already translated are skipped."""
import argparse
import os
import re

from gen_local import ask

PROMPT = """Translate the following chat message from English into natural, simple Romanian (correct diacritics: ă â î ș ț).
Rules: keep code, code blocks (```), inline `code`, Markdown symbols, numbers, URLs and names exactly as they are; translate only the human text
(and code comments). Do not add or remove anything. Output ONLY the translation.

MESSAGE:
{text}"""


def read_conversations(path):
    conversations, current = [], None
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if line.strip() == "###":
            if current:
                conversations.append(current)
            current = []
            continue
        match = re.match(r"^(USER|ASSISTANT):\s?(.*)$", line)
        if current is None:
            continue
        if match:
            current.append([match.group(1), match.group(2)])
        elif current:
            current[-1][1] += "\n" + line
    if current:
        conversations.append(current)
    return [[[role, text.strip()] for role, text in conversation] for conversation in conversations]


def acceptable(source, translation):
    if not translation or "<|" in translation:
        return False
    if source.count("```") != translation.count("```"):
        return False
    if sorted(re.findall(r"\d+", source)) != sorted(re.findall(r"\d+", translation)):
        return False
    return 0.5 <= len(translation) / max(len(source), 1) <= 2.5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    parser.add_argument("--model", default="hf.co/lmstudio-community/EuroLLM-9B-Instruct-GGUF:Q4_K_M")
    args = parser.parse_args()
    out_path = os.path.splitext(args.file)[0] + "_ro.txt"
    done = 0
    if os.path.exists(out_path):
        done = sum(1 for line in open(out_path, encoding="utf-8") if line.strip() == "###")
    conversations = read_conversations(args.file)
    print(f"{len(conversations)} conversations, {done} already translated")
    kept = dropped = 0
    for number, conversation in enumerate(conversations):
        if number < done:
            continue
        translated = []
        for role, text in conversation:
            result = ask(args.model, PROMPT.format(text=text), 0.2).strip()
            if not acceptable(text, result):
                translated = None
                break
            translated.append((role, result))
        with open(out_path, "a", encoding="utf-8") as file:
            file.write("###\n")                              # a dropped conversation is written EMPTY so the count stays aligned
            if translated:
                for role, text in translated:
                    file.write(f"{role}: {text}\n")
        if translated:
            kept += 1
        else:
            dropped += 1
        print(f"  {number + 1}/{len(conversations)}  kept {kept}  dropped {dropped}", flush=True)


if __name__ == "__main__":
    main()
