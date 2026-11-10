"""FACT-SAFE conversations from REAL Wikipedia text, written by a local model (EuroLLM through Ollama). One click, no work from you.
    python gen_grounded.py                -> data/sft/gen_grounded.jsonl      (append-only, safe to stop and run again)
The user message CONTAINS a real passage; the model only answers from that passage, so it cannot invent facts. Three kinds:
  summary       "Summarize this text in 2 sentences" + passage   -> summary (checked: no new numbers, words come from the text)
  qa            passage + a question the text answers           -> short answer from the text (same checks)
  unanswerable  passage + a question the text does NOT answer   -> "The text doesn't say that." (honesty)
Passages come from the Wikipedia articles already in data/wiki_cache.jsonl (run gen_data_wiki.py first)."""
import argparse
import json
import os
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request

from gen_common import assistant, user
from gen_local import ask

CACHE = "data/wiki_cache.jsonl"
FULL = "data/wiki_full.jsonl"
DONE = "data/grounded_done.txt"
OUT = "data/sft/gen_grounded.jsonl"
VERSION = "#v3"
HEADERS = {"User-Agent": "CosmosLM-student-project/1.0 (educational; data generation)"}
FOREIGN_SCRIPT = re.compile(r"[\u0370-\u03ff\u0400-\u04ff\u0590-\u06ff\u0900-\u097f\u3040-\u30ff\u4e00-\u9fff]")
SENSITIVE = re.compile(r"suicid|sexual|castrat|genocid|\brape|violen|masacr|execut|torture|tortur|omor", re.I)
LANGUAGE_NAME = {"en": "English", "ro": "Romanian"}

SUMMARY_ASK = {"en": ["Summarize this text in {n} sentences:\n\n{p}", "Give me a {n}-sentence summary:\n\n{p}", "TL;DR in {n} sentences:\n\n{p}"],
               "ro": ["Rezumă textul în {n} propoziții:\n\n{p}", "Fă-mi un rezumat de {n} propoziții:\n\n{p}", "Pe scurt, în {n} propoziții:\n\n{p}"]}
QA_ASK = {"en": ["{p}\n\nQuestion: {q}", "Based on this text: {q}\n\n{p}", "Text:\n{p}\n\n{q}"],
          "ro": ["{p}\n\nÎntrebare: {q}", "Pe baza textului: {q}\n\n{p}", "Text:\n{p}\n\n{q}"]}
NO_ANSWER = {"en": ["The text doesn't say that.", "That isn't mentioned in the text, so I don't know. I'd rather not guess.",
                    "The text doesn't cover that. I can only tell you what it says."],
             "ro": ["Textul nu spune asta.", "Nu apare în text, deci nu știu. Prefer să nu ghicesc.", "Textul nu tratează asta. Îți pot spune doar ce scrie în el."]}


def request_json(url):
    for attempt in range(6):
        try:
            return json.load(urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=40))
        except urllib.error.HTTPError as error:
            print(f"  HTTP {error.code}, waiting {10 * (attempt + 1)} s", flush=True)
            time.sleep(10 * (attempt + 1))
        except Exception as error:
            print(f"  {type(error).__name__}: {str(error)[:80]}")
            return None
    return None


def load_full_articles():
    """-> list of {"language","title","text"}; downloads each article once (the whole text, cut at 9000 chars)."""
    have = {}
    if os.path.exists(FULL):
        for line in open(FULL, encoding="utf-8"):
            item = json.loads(line)
            have[(item["language"], item["title"])] = item
    if not os.path.exists(CACHE):
        raise SystemExit("data/wiki_cache.jsonl is missing: run gen_data_wiki.py first")
    for line in open(CACHE, encoding="utf-8"):
        entry = json.loads(line)
        key = (entry["language"], entry["title"])
        if key in have:
            continue
        query = urllib.parse.urlencode({"action": "query", "prop": "extracts", "explaintext": 1, "redirects": 1, "format": "json", "titles": entry["title"]})
        data = request_json(f"https://{entry['language']}.wikipedia.org/w/api.php?{query}")
        time.sleep(1.5)
        if not data:
            continue
        page = next(iter(data["query"]["pages"].values()))
        if not page.get("extract"):
            continue
        item = {"language": entry["language"], "title": entry["title"], "text": page["extract"][:9000]}
        have[key] = item
        with open(FULL, "a", encoding="utf-8") as file:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"  downloaded {entry['language']}:{entry['title']}", flush=True)
    return list(have.values())


def chunks_of(text, minimum=450, maximum=1100):
    paragraphs = [p.strip() for p in text.split("\n") if len(p.strip()) > 60 and not p.strip().startswith("==")]
    chunks, current = [], ""
    for paragraph in paragraphs:
        if len(current) + len(paragraph) > maximum and len(current) >= minimum:
            chunks.append(current)
            current = ""
        current = (current + "\n\n" + paragraph).strip() if current else paragraph
    if len(current) >= minimum:
        chunks.append(current)
    return [chunk for chunk in chunks if (chunk[0].isupper() or chunk[0].isdigit()) and not FOREIGN_SCRIPT.search(chunk) and not SENSITIVE.search(chunk)]


def words(text):
    return {w for w in re.findall(r"\w+", text.casefold()) if len(w) > 3}


def grounded(answer, source):
    """The answer may only use numbers that are in the source, and most of its words must come from the source."""
    if not answer or len(answer) > 700 or "<|" in answer:
        return False
    if not set(re.findall(r"\d+", answer)) <= set(re.findall(r"\d+", source)):
        return False
    answer_words = words(answer)
    return not answer_words or len(answer_words & words(source)) / len(answer_words) >= 0.5


def sentence_count(text):
    return len([s for s in re.split(r"(?<=[.!?])\s+(?=[A-ZĂÂÎȘȚ„\"])", text.strip()) if s])


def looks_answerable(question, passage):
    """Most content words of the question are in the passage -> the passage probably answers it."""
    question_words = [w for w in re.findall(r"\w+", question.casefold()) if len(w) > 4]
    return bool(question_words) and sum(w in passage.casefold() for w in question_words) / len(question_words) >= 0.5


def supported(model, passage, question, answer):
    verdict = clean(ask(model, f"TEXT:\n{passage}\n\nREQUEST: {question}\n\nANSWER: {answer}\n\n"
                               f"Check the ANSWER against the TEXT. Is EVERY claim in the ANSWER stated in the TEXT, with nothing added, nothing contradicted "
                               f"and no cause or consequence that the TEXT does not state? Reply with exactly one word: OK or BAD.", 0.0)).upper()
    return verdict.startswith("OK")


def clean(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip().strip('"').strip()
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="hf.co/lmstudio-community/EuroLLM-9B-Instruct-GGUF:Q4_K_M")
    parser.add_argument("--chunks-per-article", type=int, default=3)
    parser.add_argument("--seed", type=int, default=5)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    articles = load_full_articles()
    if os.path.exists(DONE) and not open(DONE, encoding="utf-8").readline().startswith(VERSION):
        first = open(DONE, encoding="utf-8").readline().strip()
        tag = first[1:] if first.startswith("#v") else "v1"
        os.replace(DONE, DONE + ".old_" + tag)                # older results had wrong labels / unsupported claims
        if os.path.exists(OUT):
            os.replace(OUT, OUT + ".old_" + tag)
        print(f"old {tag} results moved aside (not used)")
    if not os.path.exists(DONE):
        open(DONE, "w", encoding="utf-8").write(VERSION + "\n")
    done = set(open(DONE, encoding="utf-8").read().split("\n"))
    jobs = []
    for article in articles:
        for index, chunk in enumerate(chunks_of(article["text"])[:args.chunks_per_article]):
            for task in ("summary", "qa", "unanswerable"):
                key = f"{article['language']}|{article['title']}|{index}|{task}"
                if key not in done:
                    jobs.append((key, article["language"], chunk, task))
    rng.shuffle(jobs)
    print(f"{len(articles)} articles, {len(jobs)} jobs left", flush=True)
    made = 0
    for number, (key, language, passage, task) in enumerate(jobs, 1):
        name = LANGUAGE_NAME[language]
        messages = None
        try:
            if task == "summary":
                count = rng.choice([2, 3])
                answer = clean(ask(args.model, f"Write a summary of the text below in exactly {count} sentences, in {name}. Use ONLY information from the text; add nothing. "
                                                f"Output ONLY the summary.\n\nTEXT:\n{passage}", 0.3))
                if grounded(answer, passage) and sentence_count(answer) == count and supported(args.model, passage, f'summarize in {count} sentences', answer):
                    messages = [user(rng.choice(SUMMARY_ASK[language]).format(n=count, p=passage)), assistant(answer)]
            elif task == "qa":
                reply = clean(ask(args.model, f"Read the text. Write ONE question that the text clearly answers, and its answer in 1-3 sentences using only the text. Language: {name}. "
                                              f"Format exactly:\nQUESTION: ...\nANSWER: ...\n\nTEXT:\n{passage}", 0.3))
                match = re.search(r"QUESTION:\s*(.+?)\s*ANSWER:\s*(.+)", reply, flags=re.S)
                if match and grounded(match.group(2).strip(), passage) and supported(args.model, passage, match.group(1).strip(), match.group(2).strip()):
                    messages = [user(rng.choice(QA_ASK[language]).format(p=passage, q=match.group(1).strip())), assistant(match.group(2).strip())]
            else:
                question = clean(ask(args.model, f"Read the text. Write ONE question about the same topic that the text does NOT answer (a detail it never mentions). "
                                                 f"Language: {name}. Output ONLY the question.\n\nTEXT:\n{passage}", 0.5))
                verdict = clean(ask(args.model, f"TEXT:\n{passage}\n\nQUESTION: {question}\n\nCan the question be answered, fully or even partly, using ONLY the text? "
                                                f"Reply with exactly one word: YES or NO.", 0.0)).upper()
                if (10 < len(question) < 200 and "\n" not in question and question.endswith("?")
                        and not looks_answerable(question, passage) and verdict.startswith("NO")):
                    messages = [user(rng.choice(QA_ASK[language]).format(p=passage, q=question)), assistant(rng.choice(NO_ANSWER[language]))]
        except Exception as error:
            print(f"  error: {type(error).__name__} {str(error)[:80]}")
            continue
        with open(DONE, "a", encoding="utf-8") as file:
            file.write(key + "\n")
        if messages:
            made += 1
            with open(OUT, "a", encoding="utf-8") as file:
                file.write(json.dumps({"messages": messages, "source": "grounded:" + task}, ensure_ascii=False) + "\n")
        print(f"  {number}/{len(jobs)}  kept {made}  ({task})", flush=True)


if __name__ == "__main__":
    main()
