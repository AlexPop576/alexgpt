"""
A short name for a new chat, from its first message (the list of chats on the left).   Used by server.py  POST /api/title.

    1. Singularity tries first: a few examples "Message: ... / Title: ...", then it writes the title (greedy, at most 12 tokens).
    2. Code checks it: 1-5 words, not one of the examples, not the message itself, and at least one word in common with
       the message (a 126M model was never trained to name chats: it often copies an example or invents a topic).
    3. Not usable -> plain rules: the tool the router would call ("Weather in Cluj", "Math: 17*23"), a greeting,
       a question about the model, or the message itself, shortened.

Examples (tested on Singularity v4):
    "Write a short poem about the sea"   -> "The Sound of the Sea"   (model)
    "tell me about the Danube"           -> "The Danube River"       (model)
    "What's the weather in Cluj?"        -> "Weather in Cluj"        (rules: the model wrote "Capital of France")
    "Who made you?"                      -> "About Singularity"      (rules: the model wrote "Charles Darwin's theory...")
"""

import re

from alexgpt import tool_router, tools

EXAMPLES = [
    ("what is the capital of France?", "Capital of France"),
    ("cât face 12 ori 7?", "Înmulțire 12 × 7"),
    ("can you help me write an email to my boss", "Email to the boss"),
    ("ce înseamnă fotosinteza?", "Fotosinteza"),
    ("hey, how are you?", "Greeting"),
]
EXAMPLE_TITLES = {tools.fold(title) for _, title in EXAMPLES}
MAX_WORDS = 5


def content_stems(text):
    return {tools._stem(tools.fold(word)) for word in tools._terms(text)}


def clean(title):
    """the model's line -> a title: no "Title:", no quotes, no full stop at the end"""
    title = title.split("\n")[0].strip()
    title = re.sub(r"^(title|titlu)\s*:\s*", "", title, flags=re.I)
    title = title.strip(" \"'„”“«»*`")
    title = re.sub(r"[.!?:;,]+$", "", title).strip()
    return title[:1].upper() + title[1:]


def usable(title, message):
    words = title.split()
    if not 1 <= len(words) <= MAX_WORDS or len(title) > 48:
        return False
    if any(bad in title for bad in ("<|", "=", "http", "Singularity")):
        return False
    if tools.fold(title) in EXAMPLE_TITLES:                 # it copied an example
        return False
    if tools.fold(title).strip() == tools.fold(re.sub(r"[.!?]+$", "", message)).strip():
        return False                                        # it repeated the message
    return bool(content_stems(title) & content_stems(message))    # about the message, not an invented topic


def model_title(model, message):
    from alexgpt.sft_format import chat_prompt
    shots = "".join(f"Message: {text}\nTitle: {title}\n\n" for text, title in EXAMPLES)
    prompt = chat_prompt([{"role": "user", "content": "Name each message in 2-4 words.\n\n" + shots + "Message: " + message}]) + "Title:"
    ids = model.tokenizer.encode(prompt)
    written = []
    for _ in range(12):
        next_id = model.greedy(ids)
        if next_id == model.tokenizer.end_id:
            break
        ids.append(next_id)
        written.append(next_id)
        if "\n" in model.tokenizer.decode(written):
            break
    title = clean(model.tokenizer.decode(written))
    return title if usable(title, message) else None


def rule_title(message):
    romanian = tools.is_romanian(message)
    for call in tool_router.route(message) or []:
        name, _, argument = call.partition(":")
        name, argument = name.strip(), argument.strip()
        if name == "weather" and argument:
            return ("Vremea în " if romanian else "Weather in ") + argument
        if name == "calc" and argument:
            return ("Calcul: " if romanian else "Math: ") + argument
        if name == "convert" and argument:
            return ("Conversie: " if romanian else "Convert: ") + argument
        if name == "now":
            return "Ora exactă" if romanian else "Current time"
        if name in ("dice", "coin", "pick"):
            return "Zaruri" if romanian else "Dice"
        if name.startswith("memory"):
            return "Memorie" if romanian else "Memory"
    try:
        from alexgpt import identity_rag
        answer, group = identity_rag.reply(message, "Singularity")
    except Exception:
        answer, group = None, None
    if answer:
        if group.startswith(("hello", "how-are-you", "thanks", "bye")):
            return "Salut" if romanian else "Hello"
        return "Despre Singularity" if romanian else "About Singularity"
    words = re.sub(r"[.!?]+$", "", " ".join(message.split())).split()
    title = " ".join(words[:6]) + ("…" if len(words) > 6 else "")
    return title[:1].upper() + title[1:] if title else "New chat"


def chat_title(model, message):
    """-> (title, "model" or "rules")"""
    message = " ".join(str(message).split())[:300]
    if not message:
        return "New chat", "rules"
    if model is not None:
        try:
            title = model_title(model, message)
            if title:
                return title, "model"
        except Exception:
            pass                                            # a title must never break the chat
    return rule_title(message), "rules"
