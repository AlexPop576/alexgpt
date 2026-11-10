"""Small helpers shared by the gen_data_*.py scripts (they write data/sft/gen_*.jsonl, one conversation per line).

Message roles:  user | assistant | system   (as always)
                tool_call  = what the model writes to use a tool, e.g. "calc: 245 * 18"   (becomes <|search|>...)
                context    = what the tool returned, put there by the app, e.g. "4410"     (becomes <|context|>...)
A tool conversation:  user -> tool_call -> context -> assistant.
"""
import json
import os

OUT_DIR = "data/sft"


def user(text):
    return {"role": "user", "content": text}


def assistant(text):
    return {"role": "assistant", "content": text}


def system(text):
    return {"role": "system", "content": text}


def tool_call(text):
    return {"role": "tool_call", "content": text}


def context(text):
    return {"role": "context", "content": text}


def format_number(value, decimals=10):
    """4410.0 -> "4410";  0.333333333333 -> "0.3333333333";  2.50 -> "2.5"  (the same rule as the calc tool)"""
    rounded = round(value, decimals)
    if rounded == int(rounded):
        return str(int(rounded))
    return f"{rounded:.{decimals}f}".rstrip("0").rstrip(".")


SMALL_TALK = {
    "en": [("hi", "Hi! What can I do for you?"), ("hello!", "Hello! What's up?"), ("hey", "Hey! Ask me anything.")],
    "ro": [("salut", "Salut! Spune, cu ce te ajut?"), ("bună!", "Bună! Ce faci?"), ("hei", "Hei! Te ascult.")],
}


def maybe_small_talk(rng, language, messages, chance=0.2):
    """Sometimes the request comes after a short greeting (so tool calls also work in the middle of a chat)."""
    if rng.random() < chance:
        question, answer = rng.choice(SMALL_TALK[language])
        return [user(question), assistant(answer)] + messages
    return messages


def write_jsonl(name, conversations, rng):
    """Shuffles and writes data/sft/<name>.jsonl, prints the count and 3 examples."""
    rng.shuffle(conversations)
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name + ".jsonl")
    with open(path, "w", encoding="utf-8") as file:
        for messages in conversations:
            file.write(json.dumps({"messages": messages, "source": name}, ensure_ascii=False) + "\n")
    print(f"{name}: {len(conversations)} conversations -> {path}")
    for messages in conversations[:3]:
        print("   ---")
        for message in messages:
            print(f"   {message['role']:9s} {message['content'][:150]!r}")
    return path
