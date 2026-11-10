"""
SFT v2, step 2 (after prepare_sft.py downloaded the raw conversations into data/sft/):
    python build_sft_mix.py                         -> data/sft_mix/supernova.jsonl   (for Supernova)
    python build_sft_mix.py --name Singularity --size "about 200 million" --size-ro "aproximativ 200 de milioane"

It reads every source, CLEANS it, FIXES its format, adds the identity conversations and writes ONE file:
    {"source": "ro_oasst", "repeat": 1, "messages": [{"role": "user", "content": "..."}, ...]}
plus data/sft_mix/<name>_preview.txt: a few conversations from every source, exactly as the model will see them.

Why each step exists (what went wrong in the first SFT):
  1. "Ca model de limbaj AI, ..."      27% of the ro_ultrachat answers start like that (a translation habit)
                                          -> conversations with these phrases are thrown out
  2. "Here's a revised version ..."    smol-rewrite / smol-summarize had the TASK in the system prompt and only
                                          a bare email in the user message, so the model learned:
                                          "a text without a question -> rewrite it" and "Hi! -> write an email"
                                          -> the task goes into the user's message, in many wordings
  3. "My name is Emily Nolen Patel"    42 identity conversations, always asked the same way
                                          -> ~300 different questions (sft_data/identity_qa.txt) + the system prompt
  4. weak small talk in English        smol_everyday was 1% of the data
                                          -> seen 3x + OpenAssistant (real people's conversations)
  5. loops ("Paris, Franța / Paris, Franța")   answers that repeat a sentence are thrown out
  6. people write Romanian without diacritics   -> 25% of Romanian user messages lose their diacritics
"""
import argparse
import glob
import json
import os
import random
import re
import unicodedata

from alexgpt.sft_format import system_prompt

parser = argparse.ArgumentParser()
parser.add_argument("--name", default="Supernova")
parser.add_argument("--size", default="about 100 million")
parser.add_argument("--size-ro", default="aproximativ 100 de milioane")
args = parser.parse_args()

RAW_DIR = "data/sft"
OUT_DIR = "data/sft_mix"
SEED = 0

# source: (max conversations kept after cleaning, how many times each one is seen per pass, language)
MIX = {
    "smol_everyday":    (3000,  3, "en"),
    "en_oasst":         (6000,  1, "en"),
    "smol_general":     (11000, 1, "en"),
    "smol_summarize":   (4000,  1, "en"),
    "smol_rewrite":     (1500,  1, "en"),
    "smol_constraints": (4000,  1, "en"),
    "ro_ultrachat":     (9000,  1, "ro"),     # long + translated: capped so it is ~1/4 of the text
    "ro_oasst":         (10000, 1, "ro"),
    "ro_norobots":      (9500,  1, "ro"),
    "ro_dolly":         (8000,  1, "ro"),
    "ro_alpaca":        (6000,  1, "ro"),
}
IDENTITY_COPIES = 3           # every identity question becomes 3 conversations (different answers / wrapping)
IDENTITY_TXT_REPEAT = 6       # sft_data/identity.txt: whole conversations, seen 6x
BANTER_REPEAT = 4             # sft_data/banter.txt: teasing about Alex, a small detail, seen 4x
SYSTEM_SHARE_IDENTITY = 0.5   # half of the identity conversations start with the system prompt
SYSTEM_SHARE_OTHERS = 0.15    # 15% of the others too: so the system prompt doesn't mean "talk about yourself"
NO_DIACRITICS_SHARE = 0.25    # Romanian user messages written without ă â î ș ț
MAX_EDITING = 600             # smol_general: at most this many "edit my text" conversations
MAX_ANSWER_CHARS = 2500       # a longer answer ends the conversation before it (~700 tokens)
TEXT_FIRST_SHARE = 0.7        # summarize / rewrite: the text first and the request at the end (like people do), 30% reversed
NAME_SOURCES = ("smol_summarize", "smol_rewrite", "smol_general")   # synthetic e-mails between the same few people

random.seed(SEED)


def plain(text):
    """lowercase, no diacritics: "Ca model de limbaj AI" -> "ca model de limbaj ai" """
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(char for char in text if not unicodedata.combining(char))


def strip_diacritics(text):
    """"Mă duc în oraș" -> "Ma duc in oras" (keeps the capital letters)"""
    text = unicodedata.normalize("NFKD", text)
    return unicodedata.normalize("NFC", "".join(char for char in text if not unicodedata.combining(char)))


# ====================================================================== filters
CLICHES = [  # compared in plain() form
    "as an ai language model", "as a language model", "as an ai assistant", "i'm just an ai", "i am just an ai",
    "as an artificial intelligence", "i am an ai language model", "i'm an ai language model",
    "model de limbaj ai", "ca model de limbaj", "ca inteligenta artificiala", "ca asistent ai", "ca un ai",
    "in calitate de model", "in calitate de ai", "in calitate de asistent", "ca asistent virtual",
]
OTHER_IDENTITIES = ["openassistant", "open assistant", "chatgpt", "openai", "smollm", "laion", "gpt-3", "gpt-4",
                    "meta ai", "i am llama", "i'm llama", "hugging face", "huggingface"]
# whole words only: "ca un ai" must not match "ca un aisberg"
CLICHE_PATTERN = re.compile(r"\b(" + "|".join(re.escape(phrase) for phrase in CLICHES) + r")\b")
OTHER_PATTERN = re.compile(r"\b(" + "|".join(re.escape(name) for name in OTHER_IDENTITIES) + r")\b")
EDITING_SIGNS = ["revised version", "rewritten version", "edited version", "suggested edits", "i made several changes",
                 "i made a few changes", "here's a revised", "here is a revised", "i've made some changes",
                 "i changed \"", "i replaced \""]
EN_WORDS = {"the", "and", "is", "to", "of", "it", "you", "that", "for", "are", "with", "this", "on", "i", "be",
            "my", "was", "have", "your", "what", "but", "can", "not", "do"}


def answers(messages):
    return [message["content"] for message in messages if message["role"] == "assistant"]


def repeats_a_sentence(text):
    """True if a sentence of 20+ characters appears twice in the same answer (the start of a loop)."""
    seen = set()
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", plain(text)):
        sentence = sentence.strip(" -*0123456789.")
        if len(sentence) < 20:
            continue
        if sentence in seen:
            return True
        seen.add(sentence)
    return False


def looks_english(messages):
    words = re.findall(r"[a-z']+", plain(" ".join(message["content"] for message in messages)))
    return len(words) > 0 and sum(word in EN_WORDS for word in words) / len(words) > 0.12


ARITHMETIC = re.compile(r"\d\s*[-+*/x×÷]\s*\d|\d\s*%\s*(of|din)\b|\bsquared\b|\bla pătrat\b|\bsquare root\b", re.I)


def problem(source, messages):
    """-> why this conversation is thrown out, or None if it's fine"""
    replies = [plain(text) for text in answers(messages)]
    if not replies:
        return "no answer"
    for reply in replies:
        if CLICHE_PATTERN.search(reply):
            return "cliche (as an AI language model)"
        if OTHER_PATTERN.search(reply):
            return "talks about another AI"
    if any(repeats_a_sentence(text) for text in answers(messages)):
        return "repeats a sentence"
    for message in messages:      # "what is 847 * 293?" must go to the calc tool, not be answered from "memory"
        if message["role"] == "user" and len(message["content"]) < 90 and ARITHMETIC.search(message["content"]):
            return "direct arithmetic (the calc tool does it)"
    if source == "en_oasst" and not looks_english(messages):
        return "not English"
    return None


def cut_long_answers(messages):
    """Keeps the conversation up to the first answer longer than MAX_ANSWER_CHARS (that answer and the rest go)."""
    kept = []
    for message in messages:
        if message["role"] == "assistant" and len(message["content"]) > MAX_ANSWER_CHARS:
            break
        kept.append(message)
    while kept and kept[-1]["role"] != "assistant":     # must end with an answer
        kept.pop()
    return kept


# ====================================================================== the task goes into the user message
SUMMARY_INSTRUCTIONS = [  # smol-summarize, system prompt "summary in up to three sentences"
    "Summarize this in a few sentences.", "Can you give me a short summary of this?", "Write a brief summary of the text.",
    "Summarize the following text.", "What's the gist of this? Two or three sentences, please.", "TL;DR?",
    "Give me a summary of this message in up to three sentences.", "Sum this up for me.",
    "Please summarize this email.", "Could you summarize this briefly?",
]
KEY_POINT_INSTRUCTIONS = [  # smol-summarize, system prompt "main key point in one very short sentence"
    "What's the main point of this, in one short sentence?", "In one sentence, what is this about?",
    "Give me the key point in one line.", "Summarize this in a single short sentence.",
    "What is the most important thing here? One sentence.",
]
REWRITE_INSTRUCTIONS = {  # smol-rewrite: which instruction fits which system prompt
    "professional": ["Rewrite this to sound more professional.", "Make this more formal, please.",
                     "Can you make this email sound more professional?", "Rewrite the text below in a formal tone."],
    "concise": ["Make this shorter and more concise.", "Can you cut this down? Keep the main points.",
                "Rewrite this to be more concise.", "Shorten this text, please."],
    "friendly": ["Rewrite this in a friendlier tone.", "Make this sound warmer and more approachable.",
                 "Can you make this message more friendly?", "Rewrite the text below so it sounds more casual."],
}


# ====================================================================== names
# smol-summarize / smol-rewrite are e-mails between the same few invented people: "Emily" is in 5,060 of 16,000,
# "Alex" in 3,337 (Alex = the model's creator!). The model learned "Emily" as a default name ("My sister's name is
# Emily"). Every conversation gets its own random names instead (the same name everywhere in that conversation).
FREQUENT_NAMES = ["Emily", "Alex", "Michael", "Sarah", "David", "John", "Jamie", "Jordan", "Liam", "Jenna", "Samantha",
                  "James", "Mark", "Emma", "Robert", "Olivia", "Ethan", "Sam", "Maria", "Mike", "Marcus", "Jessica",
                  "Oliver", "Amelia", "Lily", "Jane", "Daniel", "Rachel", "Eleanor", "Elena"]
NAME_POOL = ["Ana", "Mihai", "Ioana", "Andrei", "Elena", "Radu", "Irina", "Bogdan", "Cristina", "Vlad", "Diana", "Paul",
             "Sofia", "Matei", "Laura", "Tudor", "Bianca", "Ștefan", "Raluca", "Victor", "Hannah", "Lucas", "Chloe",
             "Noah", "Grace", "Mateo", "Zoe", "Isaac", "Nora", "Leo", "Aisha", "Kenji", "Fatima", "Diego", "Mei",
             "Tomás", "Ingrid", "Omar", "Clara", "Felix", "Nadia", "Hugo", "Leila", "Arjun", "Freya", "Marco",
             "Yuki", "Sven", "Amara", "Pablo", "Greta", "Ivan", "Rosa", "Theo", "Lina", "Caleb", "Maya", "Jonas",
             "Alina", "Sebastian", "Petra", "Kofi", "Lucia", "Henrik", "Selin", "Andrés", "Mira", "Owen", "Tara",
             "Erik", "Anya", "Nikolai", "Julia", "Rafael", "Elif", "Simon", "Dalia", "Adrian", "Carmen", "Paulo"]
NAME_PATTERN = re.compile(r"\b(" + "|".join(FREQUENT_NAMES) + r")\b")


def replace_names(messages):
    chosen = {}
    pool = [name for name in NAME_POOL]
    random.shuffle(pool)

    def new_name(match):
        old = match.group(1)
        if old not in chosen:
            chosen[old] = pool[len(chosen) % len(pool)]
        return chosen[old]
    return [dict(message, content=NAME_PATTERN.sub(new_name, message["content"])) for message in messages]


def put_task_in_user_message(source, messages):
    """smol_summarize / smol_rewrite: [system: task, user: bare text, assistant] -> [user: task + text, assistant]"""
    system = messages[0]["content"].lower() if messages[0]["role"] == "system" else ""
    rest = [message for message in messages if message["role"] != "system"]
    if source == "smol_summarize":
        options = KEY_POINT_INSTRUCTIONS if "one very short sentence" in system else SUMMARY_INSTRUCTIONS
    else:
        kind = "professional" if "professional" in system else ("concise" if "concise" in system else "friendly")
        options = REWRITE_INSTRUCTIONS[kind]
    instruction = random.choice(options)
    text = rest[0]["content"]
    if random.random() < TEXT_FIRST_SHARE:
        user = text + "\n\n" + instruction                  # the text, then the task (like the benchmark and most people)
    else:
        user = instruction + "\n\n" + text                  # ... or the task first
    return [{"role": "user", "content": user}] + rest[1:]


# ====================================================================== identity
def read_conversation_file(path):
    """sft_data/identity.txt, banter.txt (### / USER: / ASSISTANT:) -> list of conversations"""
    conversations, messages = [], []
    with open(path, encoding="utf-8") as file:
        for line in file:
            line = line.rstrip("\n")
            if line.startswith("#") and not line.startswith("###"):
                continue
            if line.startswith("###"):
                if messages:
                    conversations.append(messages)
                messages = []
            elif line.startswith("USER:"):
                messages.append({"role": "user", "content": line[5:].strip()})
            elif line.startswith("ASSISTANT:"):
                messages.append({"role": "assistant", "content": line[10:].strip()})
            elif messages:
                messages[-1]["content"] += "\n" + line
    if messages:
        conversations.append(messages)
    return conversations


def read_identity_groups(path):
    """sft_data/identity_qa.txt -> {group: {"questions": [...], "answers": [...]}}"""
    groups, current, last = {}, None, None
    with open(path, encoding="utf-8") as file:
        for line in file:
            line = line.rstrip("\n")
            if line.startswith("## "):
                current = line[3:].strip()
                groups[current] = {"questions": [], "answers": []}
            elif current and line.startswith("Q:"):
                groups[current]["questions"].append(line[2:].strip())
                last = None
            elif current and line.startswith("A:"):
                groups[current]["answers"].append(line[2:].strip())
                last = groups[current]["answers"]
            elif current and last is not None and not line.startswith("#"):
                last[-1] += "\n" + line                    # an answer that continues on the next line (a list)
    for group in groups.values():
        group["answers"] = [answer.strip() for answer in group["answers"]]
    return groups


def fill(text):
    return (text.replace("{NAME}", args.name).replace("{SIZE_RO}", args.size_ro).replace("{SIZE}", args.size))


def identity_conversations(benchmark_prompts):
    groups = read_identity_groups("sft_data/identity_qa.txt")
    greetings = {"en": groups["hello-en"], "ro": groups["hello-ro"]}
    conversations, dropped = [], 0
    for name, group in groups.items():
        language = "ro" if name.endswith("-ro") else "en"
        for question in group["questions"]:
            if plain(question).strip(" ?!.") in benchmark_prompts:
                dropped += 1                                   # the benchmark's own question: never train on it
                continue
            for copy in range(IDENTITY_COPIES):
                messages = [{"role": "user", "content": question},
                            {"role": "assistant", "content": fill(random.choice(group["answers"]))}]
                if copy == 2 and not name.startswith("hello"):      # asked in the middle of a chat, after a greeting
                    greeting = greetings[language]
                    messages = [{"role": "user", "content": random.choice(greeting["questions"])},
                                {"role": "assistant", "content": fill(random.choice(greeting["answers"]))}] + messages
                conversations.append(messages)
    return conversations, dropped


# ====================================================================== main
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    benchmark_prompts = set()
    if os.path.exists("benchmarks/alexgpt100.jsonl"):
        for line in open("benchmarks/alexgpt100.jsonl", encoding="utf-8"):
            benchmark_prompts.add(plain(json.loads(line)["prompt"]).strip(" ?!."))

    output = []                     # (source, repeat, messages)
    print(f"{'source':17s} {'raw':>7s} {'kept':>7s}   thrown out because ...")
    for source, (wanted, repeat, language) in MIX.items():
        path = os.path.join(RAW_DIR, source + ".jsonl")
        if not os.path.exists(path):
            print(f"{source:17s}  missing ({path}); run: python prepare_sft.py {source}")
            continue
        rows = [json.loads(line)["messages"] for line in open(path, encoding="utf-8") if line.strip()]
        random.shuffle(rows)
        reasons, kept, editing = {}, [], 0
        for messages in rows:
            if len(kept) >= wanted:
                break
            if source in ("smol_summarize", "smol_rewrite"):
                messages = put_task_in_user_message(source, messages)
            if source in NAME_SOURCES:
                messages = replace_names(messages)
            messages = cut_long_answers(messages)
            reason = problem(source, messages)
            if reason is None and source == "smol_general" and any(sign in plain(text) for text in answers(messages)
                                                                   for sign in EDITING_SIGNS):
                editing += 1
                if editing > MAX_EDITING:
                    reason = f"too many 'edit my text' (kept {MAX_EDITING})"
            if reason:
                reasons[reason] = reasons.get(reason, 0) + 1
                continue
            if language == "ro" and random.random() < NO_DIACRITICS_SHARE:
                messages = [dict(message, content=strip_diacritics(message["content"])) if message["role"] == "user"
                            else message for message in messages]
            kept.append(messages)
        why = ", ".join(f"{reason} {count}" for reason, count in sorted(reasons.items(), key=lambda pair: -pair[1]))
        print(f"{source:17s} {len(rows):7d} {len(kept):7d}   {why}" + (f"   (seen {repeat}x)" if repeat > 1 else ""))
        output += [(source, repeat, messages) for messages in kept]

    identity, dropped = identity_conversations(benchmark_prompts)
    output += [("identity_qa", 1, messages) for messages in identity]
    print(f"{'identity_qa':17s} {'':7s} {len(identity):7d}   ({dropped} questions dropped: same as the benchmark)")
    for source, path, repeat in (("identity", "sft_data/identity.txt", IDENTITY_TXT_REPEAT),
                                 ("banter", "sft_data/banter.txt", BANTER_REPEAT)):
        conversations = read_conversation_file(path)
        conversations = [[dict(message, content=fill(message["content"])) for message in messages]
                         for messages in conversations]
        output += [(source, repeat, messages) for messages in conversations]
        print(f"{source:17s} {'':7s} {len(conversations):7d}   (seen {repeat}x)")

    # extra sources: generated data (data/sft/gen_*.jsonl), persona and Alex's own conversations
    EXTRA_JSONL = (("gen_tools", 3), ("gen_facts", 3), ("gen_constraints", 2), ("gen_patch", 2), ("gen_markdown", 1), ("gen_idk", 2), ("gen_wiki", 2),
                   ("gen_grounded", 1), ("gen_local", 1))
    for source, repeat in EXTRA_JSONL:
        path = os.path.join(RAW_DIR, source + ".jsonl")
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            print(f"{source:17s}  missing or empty -> skipped")
            continue
        rows = [json.loads(line)["messages"] for line in open(path, encoding="utf-8") if line.strip()]
        output += [(source, repeat, messages) for messages in rows]
        print(f"{source:17s} {len(rows):7d} {len(rows):7d}   (seen {repeat}x)")
    for source, path, repeat in (("persona", "sft_data/persona_conversations.txt", 6),
                                 ("mine", "sft_data/my_conversations.txt", 6),
                                 ("mine", "sft_data/my_conversations_todo.txt", 6)):
        if not os.path.exists(path):
            print(f"{path} missing -> skipped")
            continue
        conversations = [m for m in read_conversation_file(path)
                         if not any(">>>" in x["content"] or not x["content"].strip() for x in m)]
        conversations = [[dict(x, content=fill(x["content"])) for x in m] for m in conversations]
        output += [(source, repeat, m) for m in conversations]
        print(f"{source:17s} {'':7s} {len(conversations):7d}   ({path}, seen {repeat}x)")

    # the system prompt in front of some conversations (the chat always sends it)
    identity_sources = ("identity_qa", "identity", "banter", "persona", "mine")
    with_system = 0
    final = []
    for source, repeat, messages in output:
        share = SYSTEM_SHARE_IDENTITY if source in identity_sources else SYSTEM_SHARE_OTHERS
        if messages[0]["role"] != "system" and random.random() < share:
            messages = [{"role": "system", "content": system_prompt(args.name)}] + messages
            with_system += 1
        final.append({"source": source, "repeat": repeat, "messages": messages})

    random.shuffle(final)
    out_path = os.path.join(OUT_DIR, args.name.lower() + ".jsonl")
    with open(out_path, "w", encoding="utf-8") as file:
        for row in final:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    seen = sum(row["repeat"] for row in final)
    characters = {}
    for row in final:
        characters[row["source"]] = characters.get(row["source"], 0) + row["repeat"] * sum(
            len(message["content"]) for message in row["messages"])
    total = sum(characters.values())
    print(f"\n{len(final)} conversations ({seen} with the repeats), {with_system} with the system prompt -> {out_path}")
    answer_characters = {}                # the loss is only on the answers: this is what the model learns from
    for row in final:
        answer_characters[row["source"]] = answer_characters.get(row["source"], 0) + row["repeat"] * sum(
            len(message["content"]) for message in row["messages"] if message["role"] == "assistant")
    answer_total = sum(answer_characters.values())
    print("share per source:          all text   answers only (= what the loss sees)")
    for source, count in sorted(answer_characters.items(), key=lambda pair: -pair[1]):
        print(f"   {source:17s}      {100 * characters[source] / total:5.1f}%     {100 * count / answer_total:5.1f}%")

    preview_path = os.path.join(OUT_DIR, args.name.lower() + "_preview.txt")
    with open(preview_path, "w", encoding="utf-8") as file:
        for source in list(MIX) + ["identity_qa", "identity", "banter", "persona", "mine", "gen_tools", "gen_facts", "gen_constraints", "gen_patch", "gen_markdown", "gen_idk", "gen_wiki", "gen_grounded", "gen_local"]:
            rows = [row for row in final if row["source"] == source][:3]
            for row in rows:
                file.write(f"\n{'=' * 30} {source}\n")
                for message in row["messages"]:
                    file.write(f"[{message['role']}] {message['content'][:700]}\n")
    print(f"a few of them, to read: {preview_path}")


if __name__ == "__main__":
    main()
