"""Wikipedia tool conversations with REAL passages (needs the internet, no other work from you).
    python gen_data_wiki.py             -> data/sft/gen_wiki.jsonl  (~600 conversations) + data/wiki_cache.jsonl (downloaded intros)
For every title it downloads the intro of the English or Romanian Wikipedia article, cuts it at a sentence end (~600 chars)
and builds:  question -> tool_call "wiki: Title" -> context "Title — passage (url)" -> short answer that COPIES the facts
from the passage + the source link.  The model learns: facts come from the context, not from its memory.
Wikipedia text is CC BY-SA 4.0: keep the source links (they are in the data) and list this in DATA.md."""
import argparse
import json
import os
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request

from gen_common import assistant, context, maybe_small_talk, tool_call, user, write_jsonl

CACHE = "data/wiki_cache.jsonl"
HEADERS = {"User-Agent": "CosmosLM-student-project/1.0 (educational; data generation)"}
TITLES = {
    "en": ["Eiffel Tower", "Albert Einstein", "Moon", "Python (programming language)", "Roman Empire", "Photosynthesis", "DNA", "Black hole", "Mount Everest",
           "Danube", "Transylvania", "Cluj-Napoca", "Bucharest", "Romania", "Linux", "Internet", "Pacific Ocean", "Leonardo da Vinci", "Isaac Newton",
           "World War II", "Great Wall of China", "Amazon River", "Sahara", "Computer", "Electricity", "Volcano", "Mars", "Jupiter", "Sun", "Earth",
           "Mozart", "Beethoven", "Shakespeare", "Marie Curie", "Nikola Tesla", "Alan Turing", "Ada Lovelace", "C (programming language)", "Binary search algorithm",
           "Transistor", "Central processing unit", "Artificial intelligence", "Neural network", "Mitochondrion", "Gravity", "Evolution", "Honey bee", "Blue whale",
           "Pyramids of Giza", "Colosseum", "Statue of Liberty", "Mona Lisa", "Olympic Games", "Football", "Chess", "Piano", "Guitar", "Coffee", "Tea", "Chocolate",
           "Bread", "Rainbow", "Thunderstorm", "Earthquake", "Tsunami", "Antarctica", "Iceland", "Japan", "Brazil", "Egypt", "Australia", "Carpathian Mountains",
           "Black Sea", "Vlad the Impaler", "Stephen the Great", "Nadia Comăneci", "George Enescu", "Constantin Brâncuși", "Henri Coandă", "Traian Vuia",
           "Hydrogen", "Oxygen", "Carbon", "Iron", "Gold", "Water", "Periodic table", "Atom", "Light", "Sound", "Magnet", "Telescope", "Microscope", "Penicillin",
           "Vaccine", "Heart", "Brain", "Sleep", "Printing press", "Steam engine", "Airplane", "Bicycle", "Railway", "Compass", "Clock", "Calendar", "Democracy"],
    "ro": ["Turnul Eiffel", "Cluj-Napoca", "București", "România", "Dunărea", "Transilvania", "Mihai Eminescu", "Ștefan cel Mare", "Munții Carpați", "Delta Dunării",
           "Nikola Tesla", "Constantin Brâncuși", "George Enescu", "Nadia Comăneci", "Henri Coandă", "Fotosinteză", "Luna", "Soarele", "Sistemul solar", "Internet", "Linux",
           "Marea Neagră", "Albert Einstein", "Isaac Newton", "Leonardo da Vinci", "Mozart", "Marte", "Pământ", "Apa", "Electricitatea", "Calculator", "Inteligență artificială",
           "Imperiul Roman", "Al Doilea Război Mondial", "Ion Creangă", "Ion Luca Caragiale", "Lucian Blaga", "Mircea Eliade", "Vlad Țepeș", "Mihai Viteazul",
           "Alexandru Ioan Cuza", "Brașov", "Sibiu", "Iași", "Timișoara", "Constanța", "Oradea", "Sinaia", "Castelul Bran", "Palatul Parlamentului", "Fotbal", "Șah",
           "Cafea", "Ciocolată", "Pâine", "Curcubeu", "Cutremur", "Vulcan", "Oceanul Pacific", "Antarctica", "Japonia", "Brazilia", "Egipt", "Australia",
           "Hidrogen", "Oxigen", "Aur", "Fier", "Atom", "Tabelul periodic", "Gravitație", "Evoluție", "Albină", "Balena albastră", "Inimă", "Creier", "Vaccin", "Penicilină"],
}
QUESTIONS = {"en": ["what is {t}?", "tell me about {t}", "who or what is {t}?", "give me a short summary of {t} from Wikipedia", "explain {t} briefly"],
             "ro": ["ce este {t}?", "spune-mi despre {t}", "cine sau ce este {t}?", "dă-mi un rezumat scurt despre {t} de pe Wikipedia", "explică pe scurt {t}"]}
PREFIX = {"en": ["According to Wikipedia: ", "Wikipedia says: ", ""], "ro": ["Conform Wikipediei: ", "Pe Wikipedia scrie: ", ""]}
SOURCE = {"en": "Source: ", "ro": "Sursă: "}


def fetch_intro(language, title):
    query = urllib.parse.urlencode({"action": "query", "prop": "extracts", "exintro": 1, "explaintext": 1, "redirects": 1, "format": "json", "titles": title})
    request = urllib.request.Request(f"https://{language}.wikipedia.org/w/api.php?{query}", headers=HEADERS)
    page = next(iter(json.load(urllib.request.urlopen(request, timeout=30))["query"]["pages"].values()))
    if "extract" not in page or not page["extract"].strip():
        return None
    return page["title"], page["extract"]


def clean_and_cut(text, limit=600):
    text = re.sub(r"\s*\([^)]*\)", "", text)               # (pronunciations, dates in brackets)
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-ZĂÂÎȘȚ„\"])", text)
    passage = ""
    for sentence in sentences:
        if len(passage) + len(sentence) > limit and passage:
            break
        passage += (" " if passage else "") + sentence
    return passage


def load_cache():
    cache = {}
    if os.path.exists(CACHE):
        for line in open(CACHE, encoding="utf-8"):
            item = json.loads(line)
            cache[(item["language"], item["asked"])] = item
    return cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=4)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    cache = load_cache()
    os.makedirs("data", exist_ok=True)
    for language, titles in TITLES.items():
        for asked in titles:
            if (language, asked) in cache:
                continue
            found = None
            for attempt in range(6):                          # Wikipedia answers 429 when we go too fast: wait and retry
                try:
                    found = fetch_intro(language, asked)
                    break
                except urllib.error.HTTPError as error:
                    wait = 10 * (attempt + 1)
                    print(f"  {language}:{asked}: HTTP {error.code}, waiting {wait} s", flush=True)
                    time.sleep(wait)
                except Exception as error:
                    print(f"  {language}:{asked}: {type(error).__name__} {str(error)[:80]}")
                    break
            if not found:
                continue
            title, text = found
            passage = clean_and_cut(text)
            if len(passage) < 80:
                continue
            item = {"language": language, "asked": asked, "title": title, "passage": passage,
                    "url": f"https://{language}.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"}
            cache[(language, asked)] = item
            with open(CACHE, "a", encoding="utf-8") as file:
                file.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(f"  {language}:{title} ({len(passage)} chars)", flush=True)
            time.sleep(1.5)
    conversations = []
    for item in cache.values():
        language = item["language"]
        sentences = re.split(r"(?<=[.!?])\s+", item["passage"])
        for _ in range(3):                                   # 3 different questions per article
            short = rng.random() < 0.5
            body = " ".join(sentences[:1 if short else 2])
            result = f"{item['title']} — {item['passage']} ({item['url']})"
            answer = rng.choice(PREFIX[language]) + body + "\n\n" + SOURCE[language] + item["url"]
            question = rng.choice(QUESTIONS[language]).format(t=item["title"] if rng.random() < 0.8 else item["asked"])
            conversations.append(maybe_small_talk(rng, language, [user(question), tool_call(f"wiki: {item['title']}"), context(result), assistant(answer)], 0.1))
    write_jsonl("gen_wiki", conversations, rng)


if __name__ == "__main__":
    main()
