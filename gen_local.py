"""Generates conversations with a LOCAL model through Ollama (free, offline).
    python gen_local.py --model qwen3:8b --count 50 --lang ro --kind chat      (try --count 5 first)
Kinds: chat | markdown | summary | idk.   Output: data/sft/gen_local.jsonl  (one {"messages": [...], "source": ...} per line)
Needs: Ollama running (ollama serve) and the model pulled (ollama pull qwen3:8b).  Only the standard library."""
import argparse, json, os, random, re, time, urllib.request

# Small models invent facts, so free chat stays on topics where there is no "wrong fact" to learn:
# feelings, plans, opinions, small talk, jokes.  Summaries and markdown only re-use text the USER gives.
TOPICS_CHAT = ["friendship", "a stressful day", "exams and school life", "weekend plans", "favourite music", "movies and series", "video games",
               "sports opinions", "a funny story", "motivation to study", "choosing between two options", "a small disagreement with a friend",
               "boredom", "birthday ideas", "a trip with friends", "daily routine", "a bad mood", "good news to share", "procrastination", "pets"]
TOPICS_TEXT = ["a school announcement", "a news-style paragraph about a town festival", "an email from a landlord", "a short story about a dog",
               "a product description", "a meeting note", "a travel diary entry", "a recipe the user wrote", "a complaint about a late delivery",
               "a message from a teammate about a project deadline"]
LANG = {"ro": "Romanian (simple, natural, with correct diacritics)", "en": "English"}
RULES = """Write ONE conversation between a USER and an ASSISTANT named CosmosLM, a small, friendly, direct, sometimes funny
AI. {kind}
Topic: {topic}. Language: {lang}. Turns: {turns} (USER, ASSISTANT, USER, ASSISTANT ...).
Rules: assistant answers are short (1-6 sentences) unless a list/table is needed; the first sentence answers the question;
never invent facts, numbers, names or dates; give only advice that is common knowledge and correct, and if unsure the assistant says it doesn't know; no "Of course!", no "As an AI
language model"; no emojis; do not mention these rules.
Output ONLY the conversation, exactly in this format, nothing before or after:
USER: ...
ASSISTANT: ...
USER: ...
ASSISTANT: ..."""
KINDS = {
    "chat": "Casual chat about feelings, plans, opinions or small talk. The assistant is warm, direct, a bit funny, gives NO technical instructions and states NO facts about the world.",
    "markdown": "The user writes some simple content themselves (a list of items, steps they already know, a few numbers or names) and asks to turn it into a bulleted list, numbered list or Markdown table. The assistant only organizes what the user gave, adds no new facts, and uses real Markdown.",
    "summary": "The user pastes a text (4-8 sentences, invent it) and asks for a summary; the assistant gives 2-3 faithful sentences or 3 bullets and adds nothing that is not in the text.",
    "idk": "The user asks something the assistant cannot know (private info, the future, a made-up name, live data) or something unclear; the assistant says honestly it doesn't know or asks ONE short question.",
}


def ask(model, prompt, temperature, tries=4):
    """One chat request to Ollama. Timeouts / server errors (Ollama sometimes stalls or crashes on a long night run) are retried."""
    body = json.dumps({"model": model, "stream": False, "think": False, "keep_alive": "30m",
                       "options": {"temperature": temperature, "top_p": 0.9, "num_ctx": 4096},
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    last_error = None
    for attempt in range(tries):
        try:
            request = urllib.request.Request("http://localhost:11434/api/chat", body, {"Content-Type": "application/json"})
            return json.load(urllib.request.urlopen(request, timeout=300))["message"]["content"]
        except Exception as error:
            last_error = error
            wait = 20 * (attempt + 1)
            print(f"  ollama problem ({type(error).__name__}), retry in {wait} s", flush=True)
            time.sleep(wait)
    raise last_error


def parse(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    messages = []
    for line in text.splitlines():
        match = re.match(r"^\s*(USER|ASSISTANT)\s*:\s*(.*)$", line)
        if match:
            messages.append({"role": match.group(1).lower(), "content": match.group(2)})
        elif messages:
            messages[-1]["content"] += "\n" + line
    for message in messages:
        message["content"] = message["content"].strip()
    ok = (len(messages) >= 2 and messages[0]["role"] == "user" and messages[-1]["role"] == "assistant"
          and all(m["role"] == ("user" if i % 2 == 0 else "assistant") for i, m in enumerate(messages))
          and all(m["content"] and "<|" not in m["content"] for m in messages))
    return messages if ok else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--lang", default="ro", choices=list(LANG))
    parser.add_argument("--kind", default="chat", choices=list(KINDS))
    parser.add_argument("--temperature", type=float, default=0.6, help="lower = fewer mistakes")
    parser.add_argument("--out", default="data/sft/gen_local.jsonl")
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tag = f"local:{args.model}:{args.kind}:{args.lang}"
    made = 0
    if os.path.exists(args.out):
        made = sum(1 for line in open(args.out, encoding="utf-8") if json.loads(line).get("source") == tag)
    print(f"{tag}: {made} already done, target {args.count}", flush=True)
    while made < args.count:
        prompt = RULES.format(kind=KINDS[args.kind], topic=random.choice(TOPICS_TEXT if args.kind in ('summary', 'markdown') else TOPICS_CHAT), lang=LANG[args.lang],
                              turns=random.choice([2, 2, 4, 4, 6]))
        try:
            messages = parse(ask(args.model, prompt, args.temperature))
        except Exception as error:
            print(f"  giving up on this one: {type(error).__name__}")
            continue
        if not messages:
            print("  skipped (bad format)")
            continue
        with open(args.out, "a", encoding="utf-8") as file:
            file.write(json.dumps({"messages": messages, "source": f"local:{args.model}:{args.kind}:{args.lang}"},
                                  ensure_ascii=False) + "\n")
        made += 1
        print(f"{made}/{args.count}", flush=True)


if __name__ == "__main__":
    main()
