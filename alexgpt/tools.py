"""
The tools Singularity can call.  Claude wrote calc, convert, wiki, now, memory_* and run_tool;
YOU write the rest (every `raise NotImplementedError`): coin.
The training data and this code agree on the same inputs and outputs.
Check your work any time:   python tests/test_tools.py            (offline tools)
                            python tests/test_tools.py --online   (also Wikipedia, weather, on-this-day)

How a tool is called (the model writes ONE line, the app runs it and puts the result back in front of the model):
    model writes:    <|search|>calc: 245 * 18
    app runs:        run_tool("calc: 245 * 18")  ->  "4410"
    model then sees: <|context|>4410   and writes the final answer ("245 x 18 = 4410").
The call line is always   <tool name>: <arguments>   (arguments are plain text, no JSON: a small model breaks JSON).

Rules for every tool:
  - input and output are plain str (memory is the only tool that keeps something between calls)
  - the result is SHORT: at most MAX_RESULT_CHARS characters (the model only sees 1024 tokens in total)
  - a tool never crashes: on any problem it returns a short sentence starting with "error:"
    ("error: division by zero"), so the model can tell the user honestly
  - no eval() / exec(): a message from a stranger must never run as code
Later the same logic is rewritten in JavaScript for the browser demo; this Python version is the reference.
"""

import ast
import json
import time
import math
import random
import re
import unicodedata
import urllib.error
import urllib.parse
import urllib.request

MAX_RESULT_CHARS = 900
MEMORY_FILE = "memory.json"          # where memory_save keeps the facts (a dict, saved as JSON)

TOOL_NAMES = ["calc", "convert", "wiki", "weather", "now", "memory_save", "memory_get",
              "dice", "coin", "pick", "today_in_history"]


def _format_number(value, decimals=10):
    """4410.0 -> "4410";  0.333333333333 -> "0.3333333333";  2.50 -> "2.5"  (same rule as the training data)"""
    rounded = round(value, decimals)
    if rounded == int(rounded):
        return str(int(rounded))
    return f"{rounded:.{decimals}f}".rstrip("0").rstrip(".")


UNITS = {   # unit -> (kind, factor to the base unit)
    "km": ("length", 1000), "m": ("length", 1), "cm": ("length", 0.01), "mm": ("length", 0.001), "miles": ("length", 1609.344),
    "feet": ("length", 0.3048), "inches": ("length", 0.0254),
    "kg": ("mass", 1), "g": ("mass", 0.001), "pounds": ("mass", 0.45359237), "ounces": ("mass", 0.028349523125),
    "liters": ("volume", 1), "ml": ("volume", 0.001), "gallons": ("volume", 3.785411784),
    "km/h": ("speed", 1 / 3.6), "mph": ("speed", 0.44704), "m/s": ("speed", 1),
    "hours": ("time", 3600), "minutes": ("time", 60), "seconds": ("time", 1),
    "celsius": ("temperature", None), "fahrenheit": ("temperature", None), "kelvin": ("temperature", None),
}
UNIT_ALIASES = {"kilometers": "km", "kilometres": "km", "kilometri": "km", "meters": "m", "metres": "m", "metri": "m",
                "centimeters": "cm", "centimetri": "cm", "millimeters": "mm", "milimetri": "mm", "mile": "miles", "mi": "miles",
                "ft": "feet", "foot": "feet", "picioare": "feet", "inch": "inches", "in": "inches", "inci": "inches",
                "kilograms": "kg", "kilograme": "kg", "grams": "g", "grame": "g", "lbs": "pounds", "lb": "pounds", "pound": "pounds",
                "livre": "pounds", "oz": "ounces", "ounce": "ounces", "uncii": "ounces", "l": "liters", "liter": "liters",
                "litre": "liters", "litri": "liters", "milliliters": "ml", "mililitri": "ml", "gallon": "gallons", "galoane": "gallons",
                "kph": "km/h", "kmh": "km/h", "hour": "hours", "ore": "hours", "h": "hours", "minute": "minutes", "min": "minutes",
                "second": "seconds", "secunde": "seconds", "s": "seconds", "c": "celsius", "°c": "celsius", "f": "fahrenheit",
                "°f": "fahrenheit", "k": "kelvin", "celcius": "celsius", "celsious": "celsius", "farenheit": "fahrenheit",
                "fahrenheight": "fahrenheit", "kalvin": "kelvin", "kelvins": "kelvin", "grade celsius": "celsius", "grade": "celsius"}


def _canonical_unit(name):
    name = name.strip().lower().replace("grade ", "")
    name = UNIT_ALIASES.get(name, name)
    return name if name in UNITS else None


def calc(expression):
    """Evaluates an arithmetic expression safely.
    Used by: run_tool("calc: ...").   Input: "245 * 18",  "(3 + 4) ** 2 / 7",  "sqrt(144)",  "15% of 80"
    Output: the number as text, without useless zeros:  "4410",  "7.0" -> "7",  "0.3333333333" (10 decimals at most)
    Allowed: numbers, + - * / ** %, parentheses, sqrt, abs, round, min, max, pi.   Nothing else (no names, no calls).
    Hint: parse it yourself into tokens and evaluate with two stacks, or walk Python's `ast` tree and accept only
    the node types above.  Errors: "error: division by zero", "error: cannot read the expression"."""
    text = expression.strip().replace("^", "**")
    if text.count(",") == 1 and "." not in text and "(" not in text:      # decimal comma: 2,5 * 4
        text = text.replace(",", ".")
    if not text or len(text) > 200:
        return "error: cannot read the expression"
    # "15% of 80" -> "(15/100)*(80)"
    text = re.sub(r"(\d+(?:\.\d+)?)\s*%\s*(?:of|din)\s*(\d+(?:\.\d+)?)", r"((\1/100)*\2)", text)
    operators = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b,
                 ast.Div: lambda a, b: a / b, ast.Pow: lambda a, b: a ** b, ast.Mod: lambda a, b: a % b}
    functions = {"sqrt": math.sqrt, "abs": abs, "round": round, "min": min, "max": max}

    def walk(node):
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.Name) and node.id == "pi":
            return math.pi
        if isinstance(node, ast.BinOp) and type(node.op) in operators:
            left, right = walk(node.left), walk(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 1000:
                raise OverflowError
            return operators[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            return -walk(node.operand) if isinstance(node.op, ast.USub) else walk(node.operand)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in functions and not node.keywords:
            return functions[node.func.id](*[walk(argument) for argument in node.args])
        raise ValueError("not allowed")

    try:
        value = walk(ast.parse(text, mode="eval"))
    except ZeroDivisionError:
        return "error: division by zero"
    except (SyntaxError, ValueError, TypeError, OverflowError, RecursionError, MemoryError):
        return "error: cannot read the expression"
    if isinstance(value, complex) or value != value or value in (float("inf"), float("-inf")):
        return "error: cannot read the expression"
    return _format_number(value, 4)          # "13.784": a long decimal gets garbled when the model copies it


def convert(value, from_unit, to_unit):
    """Converts units.  Used by: run_tool("convert: 5 km to miles").   Inputs: 5.0, "km", "miles"
    Output: "5 km = 3.107 miles" (3 decimals at most).
    Units to support: km, m, cm, mm, miles, feet, inches, kg, g, pounds, ounces, liters, ml, gallons,
    celsius, fahrenheit, kelvin, km/h, mph, m/s, hours, minutes, seconds.   (temperature is not a simple factor!)
    Errors: "error: unknown unit 'xyz'", "error: cannot convert km to kg"."""
    source, target = _canonical_unit(from_unit), _canonical_unit(to_unit)
    if source is None:
        return f"error: unknown unit '{from_unit}'"
    if target is None:
        return f"error: unknown unit '{to_unit}'"
    if UNITS[source][0] != UNITS[target][0]:
        return f"error: cannot convert {source} to {target}"
    if UNITS[source][0] == "temperature":
        celsius = value if source == "celsius" else (value - 32) * 5 / 9 if source == "fahrenheit" else value - 273.15
        result = celsius if target == "celsius" else celsius * 9 / 5 + 32 if target == "fahrenheit" else celsius + 273.15
    else:
        result = value * UNITS[source][1] / UNITS[target][1]
    return f"{_format_number(value, 3)} {source} = {_format_number(result, 3)} {target}"


_WIKI_PAGES = {}        # search -> (title, article text): the same question twice asks Wikipedia once
LAST_WIKI = {}          # details of the last good wiki() result, for the app: title, url, a longer text ("long")

STOP_WORDS = set("""a an the and or but of to in on at by for with from into about as is are was were be been being it its this that these those
there here what which who whom whose when where why how much many long far old does do did done can could should would will shall may might
must have has had having not no yes than then so such very just also only some any all each other more most less least own same too out up down
over under again further once tell me please search find look wikipedia google internet web info information something anything know
ce cine care cum unde cand când cat cât câte cati câți de la din pe in în si și sau un o niste niște este e sunt era au a fost fi pentru cu
despre spune spune-mi zi zi-mi mai mult multe ceva caut cauta caută vreau stii știi ma mă imi îmi lui lor ei el ea acest aceasta această asta
ăsta aia acel acea""".split())
SKIP_SECTIONS = re.compile(r"^(references|see also|external links|notes|further reading|bibliography|sources|footnotes|citations|"
                           r"note|referințe|referinte|vezi și|vezi si|legături externe|legaturi externe|bibliografie|lectură suplimentară)$", re.I)


def _terms(text):
    """'How long does the Moon take to orbit Earth?' -> ['moon', 'take', 'orbit', 'earth'] (no small words)"""
    words = re.findall(r"[0-9a-zăâîșțşţéèáíóúüöäçñ]+", text.lower())
    return [word for word in words if (len(word) >= 3 or word.isdigit()) and word not in STOP_WORDS]


def _stem(word):
    return word[:5] if len(word) > 5 else word          # "orbits" and "orbit", "românia" and "româniei" match


def _split_sentences(text):
    return [s for s in re.split(r"(?<=[.!?])\s+(?=[A-ZĂÂÎȘȚ„\"0-9])", text) if s.strip()]


# ---------------------------------------------------------------- strategy 4: the search must match the question's names
def fold(text):
    """'Ștefan cel Mare' -> 'stefan cel mare' (lower case, no diacritics), so 'Stefan' finds 'Ștefan'"""
    return "".join(c for c in unicodedata.normalize("NFKD", text.lower()) if not unicodedata.combining(c))


FOLDED_STOP_WORDS = {fold(word) for word in STOP_WORDS}
RO_VOTES = set("cine ce care cum unde cand cat cate cati este sunt fost au despre pentru din si sau lui de la pe cu mai al ale "
               "un o unei unui cel cea niste dupa intre fara catre".split())
EN_VOTES = set("the is are was were who what where when how which of and did does do to an in on it that by from with for "
               "about his her their has have".split())


def is_romanian(text):
    """A vote: Romanian function words + diacritics against English function words.
    'unde este Romania' -> True (unde, este);  'where is Romania' -> False;  'cine a scris Luceafarul' -> True."""
    words = re.findall(r"[^\W\d_]+", fold(text))
    romanian = sum(word in RO_VOTES for word in words) + min(len(re.findall(r"[ăâîșțşţ]", text.lower())), 2)
    english = sum(word in EN_VOTES for word in words)
    return romanian > english


COMMAND_WORDS = set("list name give show explain describe define enumera numeste explica descrie defineste da arata".split())


def key_entities(query):
    """The names the answer must be about: capitalized words that are not small words, and acronyms.
    'when did Napoleon visit Cluj?' -> ['Napoleon', 'Cluj'];  'who invented the telephone?' -> []"""
    found = []
    for word in re.findall(r"[^\W\d_]+", query):
        if word == "I" or not word[0].isupper() or fold(word) in FOLDED_STOP_WORDS or fold(word) in COMMAND_WORDS:
            continue
        if word not in found:
            found.append(word)
    return found


def entity_in(entity, folded_words, original_text):
    """Is the name in the text?  'Clujul' finds 'Cluj' (same start), 'EU' finds 'European Union' (initials)."""
    folded = fold(entity)
    if len(entity) > 1 and entity.isupper():                # an acronym: the word itself, or the words of its initials
        if folded in folded_words:
            return True
        initials = r"\b" + r"[\w-]*\s+(?:of\s+|the\s+|and\s+|for\s+)?".join(entity) + r"[\w-]*"
        return re.search(initials, original_text) is not None
    if len(folded) <= 3:
        return folded in folded_words
    start = folded[:max(4, min(6, len(folded) - 2))]
    return any(word.startswith(start) for word in folded_words)


def _search_candidates(get, language, query):
    """ONE request: the top 5 search results, each with its intro and its redirects (other names of the same
    article, e.g. 'Stefan cel Mare' -> 'Stephen the Great').  -> [{"title", "intro", "aliases", "rank"}], best rank first."""
    key = f"search|{language}|{query.lower()}"
    if key not in _WIKI_PAGES:
        pages = get({"action": "query", "generator": "search", "gsrsearch": query, "gsrlimit": 5,
                     "prop": "extracts|redirects", "exintro": 1, "explaintext": 1, "exlimit": 5,
                     "rdlimit": "max", "rdprop": "title"}).get("query", {}).get("pages", {})
        candidates = []
        for page in pages.values():
            candidates.append({"title": page.get("title", ""), "intro": page.get("extract", "").strip(),
                               "aliases": [r["title"] for r in page.get("redirects", [])], "rank": page.get("index", 99)})
        candidates.sort(key=lambda c: c["rank"])
        if len(_WIKI_PAGES) > 300:
            _WIKI_PAGES.clear()
        _WIKI_PAGES[key] = candidates
    return _WIKI_PAGES[key]


def names_together(entities, names, text):
    """The question's names that are NOT in the article's title or other names must appear TOGETHER in one paragraph.
    'Albert Einstein' + 'Bucharest' in the article 'Lucian Blaga': mentioned in different paragraphs -> False.
    'EU' for the article 'Romania and the euro': Romania is in the title, EU in some paragraph -> True."""
    name_words = set(re.findall(r"[^\W\d_]+", fold(names)))
    remaining = [entity for entity in entities if not entity_in(entity, name_words, names)]
    if not remaining:
        return True
    for paragraph in text.split("\n"):
        words = set(re.findall(r"[^\W\d_]+", fold(paragraph)))
        if all(entity_in(entity, words, paragraph) for entity in remaining):
            return True
    return False


def _choose_article(candidates, query, entities):
    """The best of the 5 articles.  An article is allowed only if its title + other names + intro contain EVERY
    name of the question ('Napoleon' and 'Cluj' -> 'Marie of Romania' is refused).  Among the allowed ones:
    the question's words found in it, + a bonus for a better search rank, + a bonus if the title is the question's name.
    -> the candidate, or None if none is allowed."""
    query_stems = {_stem(fold(word)) for word in _terms(query)}
    query_words = {fold(word) for word in _terms(query)}
    best, best_points = None, None
    for position, candidate in enumerate(candidates):
        names = candidate["title"] + " | " + " | ".join(candidate["aliases"][:60])
        haystack = names + " " + candidate["intro"][:1500]
        words = set(re.findall(r"[^\W\d_]+", fold(haystack)))
        if entities and not names_together(entities, names, candidate["intro"]):
            continue
        points = len(query_stems & {_stem(word) for word in words})
        points += (len(candidates) - position) * 0.5
        title_words = [fold(word) for word in _terms(candidate["title"])]
        if title_words and all(word in query_words for word in title_words):
            points += 2                                 # 'Napoleon' beats 'Napoleon III' for 'when was Napoleon born'
        if best_points is None or points > best_points:
            best, best_points = candidate, points
    return best


def _full_text(get, language, title):
    """the whole article as plain text (cached): one request"""
    key = f"page|{language}|{title}"
    if key not in _WIKI_PAGES:
        pages = get({"action": "query", "titles": title, "prop": "extracts", "explaintext": 1,
                     "redirects": 1}).get("query", {}).get("pages", {})
        _WIKI_PAGES[key] = next(iter(pages.values()), {}).get("extract", "").strip() if pages else ""
    return _WIKI_PAGES[key]


def wiki(query, language="en", max_chars=MAX_RESULT_CHARS):
    """Searches Wikipedia and returns the most relevant passage of the best article.
    Used by: run_tool("wiki: Turnul Eiffel") (language "ro") and run_tool("wiki: Eiffel Tower") ("en").
    Inputs: the search words, "en" or "ro", the maximum length of the result.
    Output: "<Title> — <passage of 1-4 sentences> (<url>)"   (what the model reads and copies from)
    Also fills LAST_WIKI = {"title", "url", "long"}: "long" is a longer piece of the article (intro + the paragraphs
    that match the question best), which the app shows after the model's own sentence.
    How it picks the passage: the search words that are NOT in the title ("how long ... orbit" for the article
    "Orbit of the Moon" -> "long") are the question; the sentences with the most of them win.  No such words -> the intro.
    Errors: "error: nothing found for 'xyz'", "error: no internet"."""
    LAST_WIKI.clear()
    headers = {"User-Agent": "CosmosLM-student-project/1.0 (educational)"}
    query = query.strip()
    if not query:
        return "error: nothing found for ''"

    def get(parameters):
        url = f"https://{language}.wikipedia.org/w/api.php?" + urllib.parse.urlencode({**parameters, "format": "json"})
        waits = [2, 5, 10]                              # too many requests (429): wait longer each time, 3 more tries
        for attempt in range(len(waits) + 1):
            try:
                return json.load(urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=15))
            except urllib.error.HTTPError as error:
                if error.code != 429 or attempt == len(waits):
                    raise
                time.sleep(min(float(error.headers.get("Retry-After") or waits[attempt]), 10))

    entities = key_entities(query)
    try:
        candidates = _search_candidates(get, language, query)
        if not candidates:
            return f"error: nothing found for '{query}'"
        chosen = _choose_article(candidates, query, entities)
        if chosen is None and entities:
            # no intro names everything: maybe the best article says it further down (one more request)
            best = _choose_article(candidates, query, [])
            text = _full_text(get, language, best["title"])
            if names_together(entities, best["title"] + " | " + " | ".join(best["aliases"][:60]), text):
                chosen = best
        if chosen is None:
            return "error: no article mentions " + " and ".join(entities)
        title, aliases = chosen["title"], chosen["aliases"]
        title_stems = {_stem(fold(word)) for word in _terms(title + " " + " ".join(aliases[:60]))}
        asks_more = [word for word in _terms(query) if _stem(fold(word)) not in title_stems]
        if asks_more or len(chosen["intro"]) < 300:
            text = _full_text(get, language, title)     # a real question: the answer may be further down
        else:
            text = chosen["intro"]                      # "tell me about NATO": the intro is enough, no second request
    except urllib.error.HTTPError as error:
        return "error: Wikipedia is busy, try again in a minute" if error.code == 429 else f"error: nothing found for '{query}'"
    except (urllib.error.URLError, OSError, TimeoutError):
        return "error: no internet"
    except Exception:
        return f"error: nothing found for '{query}'"
    if not text:
        return f"error: nothing found for '{query}'"

    # the article -> paragraphs (the intro is everything before the first "== Heading ==")
    paragraphs, in_intro, skipping = [], True, False
    for line in text.split("\n"):
        line = line.strip()
        heading = re.fullmatch(r"=+\s*(.*?)\s*=+", line)
        if heading:
            in_intro = False
            skipping = bool(SKIP_SECTIONS.match(heading.group(1)))
            continue
        if skipping or len(line) < 40:
            continue
        line = re.sub(r"\s*\([^()]*\)", "", line)                    # (pronunciations, dates in brackets)
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            paragraphs.append({"text": line, "intro": in_intro, "sentences": _split_sentences(line)})
    if not paragraphs:
        return f"error: nothing found for '{query}'"

    title_stems = {_stem(word) for word in _terms(title)}
    question = [word for word in dict.fromkeys(_terms(query)) if _stem(word) not in title_stems]
    question_stems = {_stem(word) for word in question}

    # the kind of answer the question wants: "how long" -> a time ("27.3 days"), "when" -> a year, "how many" -> a number
    low_query = query.lower()
    wanted = None
    if re.search(r"how long|how much time|cât timp|cat timp|cât durează|cat dureaza|în cât timp|in cat timp", low_query):
        wanted = r"\b\d+(?:[.,]\d+)?\s*(?:days?|years?|hours?|minutes?|seconds?|months?|weeks?|zile|ani|ore|minute|secunde|luni|săptămâni)\b"
    elif re.search(r"\bwhen\b|what year|\bcând\b|\bcand\b|în ce an|in ce an", low_query):
        wanted = r"\b(?:1\d{3}|20\d{2})\b"
    elif re.search(r"how many|how much|how far|how big|how tall|how old|câți|cati|câte|cate|cât de|cat de", low_query):
        wanted = r"\d"

    title_folded = fold(title)
    sentence_entities = [entity for entity in entities if fold(entity)[:4] not in title_folded]   # names NOT in the title

    def score(sentence):
        points = len(question_stems & {_stem(word) for word in _terms(sentence)})
        if sentence_entities:                           # "Alesso" in a song article: the sentence that names him wins
            sentence_words = set(re.findall(r"[^\W\d_]+", fold(sentence)))
            points += sum(entity_in(entity, sentence_words, sentence) for entity in sentence_entities)
        if wanted and re.search(wanted, sentence, re.I):
            points += 1.5
        return points

    url = f"https://{language}.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
    room = max_chars - len(title) - len(url) - 6
    first_sentence = paragraphs[0]["sentences"][0]
    ranked = []                                                       # (score, position, sentence)
    position = 0
    for paragraph in paragraphs:
        paragraph["score"] = 0
        for sentence in paragraph["sentences"]:
            points = score(sentence)
            paragraph["score"] += points
            ranked.append((points, position, sentence))
            position += 1
    best_points = max((points for points, _, _ in ranked), default=0)

    if question and best_points > 0:
        # a question: the first sentence + the sentences that answer it, in the article's order
        chosen = sorted(ranked, key=lambda item: (-item[0], item[1]))[:4]
        picked = [first_sentence] + [s for _, _, s in sorted(chosen, key=lambda item: item[1]) if s != first_sentence]
        best_paragraphs = sorted(paragraphs, key=lambda p: -p["score"])[:2]
        long_parts = [paragraphs[0]] + [p for p in paragraphs if p in best_paragraphs and p is not paragraphs[0]]
    else:
        # a topic ("NATO"): the intro
        picked = [s for p in paragraphs if p["intro"] for s in p["sentences"]] or paragraphs[0]["sentences"]
        long_parts = [p for p in paragraphs if p["intro"]] or paragraphs[:1]
        if sum(len(p["text"]) for p in long_parts) < 500 and len(paragraphs) > len(long_parts):
            long_parts = long_parts + [paragraphs[len(long_parts)]]

    passage = ""
    for sentence in picked:
        if len(passage) + len(sentence) > room and passage:
            break
        passage += (" " if passage else "") + sentence
    long_text = ""
    for paragraph in long_parts:
        if len(long_text) > 1600:
            break
        long_text += ("\n\n" if long_text else "") + paragraph["text"]
    relevant = len(question) < len(set(_terms(query))) or best_points > 0     # the title or a sentence matches the search
    if entities:
        relevant = True                                 # every name of the question is in this article (checked above)
    LAST_WIKI.update({"title": title, "url": url, "long": long_text[:2200], "language": language, "relevant": relevant,
                      "aliases": aliases[:20]})
    return f"{title} — {passage[:max(room, 0)]} ({url})"



def weather(city):
    """Current weather + today's forecast.   Used by: run_tool("weather: Cluj-Napoca").   Input: a city name
    Output: "Cluj-Napoca: 12°C, partly cloudy, wind 14 km/h; today min 6°C, max 15°C, rain chance 20%"
    Hint: Open-Meteo, no key needed: geocoding-api.open-meteo.com/v1/search?name=<city> gives latitude/longitude,
    then api.open-meteo.com/v1/forecast?latitude=..&longitude=..&current=temperature_2m,weather_code,wind_speed_10m
    &daily=temperature_2m_max,temperature_2m_min,precipitation_probability_max&timezone=auto.
    weather_code is a number (0 clear, 1-3 cloudy, 45 fog, 51-67 rain, 71-77 snow, 95 storm): turn it into words.
    Errors: "error: city 'xyz' not found", "error: no internet"."""
    headers = {"User-Agent": "CosmosLM-student-project/1.0 (educational)"}
    city = city.strip()
    if not city:
        return "error: city '' not found"

    def get(url, parameters):
        return json.load(urllib.request.urlopen(urllib.request.Request(url + "?" + urllib.parse.urlencode(parameters), headers=headers), timeout=15))

    try:
        places = get("https://geocoding-api.open-meteo.com/v1/search", {"name": city, "count": 1}).get("results")
        if not places:
            return f"error: city '{city}' not found"
        place = places[0]
        data = get("https://api.open-meteo.com/v1/forecast", {
            "latitude": place["latitude"], "longitude": place["longitude"], "timezone": "auto",
            "current": "temperature_2m,weather_code,wind_speed_10m",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max", "forecast_days": 1})
        current, daily = data["current"], data["daily"]
    except (urllib.error.URLError, OSError, TimeoutError):
        return "error: no internet"
    except Exception:
        return f"error: city '{city}' not found"
    code = current.get("weather_code", 0)
    if code == 0:
        sky = "clear"
    elif code in (1, 2):
        sky = "partly cloudy"
    elif code == 3:
        sky = "overcast"
    elif code in (45, 48):
        sky = "fog"
    elif 51 <= code <= 57:
        sky = "drizzle"
    elif 61 <= code <= 67 or 80 <= code <= 82:
        sky = "rain"
    elif 71 <= code <= 77 or code in (85, 86):
        sky = "snow"
    elif code >= 95:
        sky = "thunderstorm"
    else:
        sky = "mixed weather"
    rain = daily.get("precipitation_probability_max", [None])[0]
    text = (f"{place['name']}: {round(current['temperature_2m'])}°C, {sky}, wind {round(current['wind_speed_10m'])} km/h; "
            f"today min {round(daily['temperature_2m_min'][0])}°C, max {round(daily['temperature_2m_max'][0])}°C")
    return text + (f", rain chance {rain}%" if rain is not None else "")


TIME_ZONES = {"new york": "America/New_York", "miami": "America/New_York", "boston": "America/New_York", "washington": "America/New_York",
              "chicago": "America/Chicago", "los angeles": "America/Los_Angeles", "san francisco": "America/Los_Angeles",
              "toronto": "America/Toronto", "mexico city": "America/Mexico_City", "sao paulo": "America/Sao_Paulo",
              "london": "Europe/London", "londra": "Europe/London", "paris": "Europe/Paris", "madrid": "Europe/Madrid",
              "marbella": "Europe/Madrid", "malaga": "Europe/Madrid", "barcelona": "Europe/Madrid", "berlin": "Europe/Berlin",
              "roma": "Europe/Rome", "rome": "Europe/Rome", "viena": "Europe/Vienna", "vienna": "Europe/Vienna",
              "lisabona": "Europe/Lisbon", "lisbon": "Europe/Lisbon", "atena": "Europe/Athens", "athens": "Europe/Athens",
              "bucuresti": "Europe/Bucharest", "bucharest": "Europe/Bucharest", "cluj": "Europe/Bucharest",
              "cluj-napoca": "Europe/Bucharest", "cluj napoca": "Europe/Bucharest", "romania": "Europe/Bucharest", "moscova": "Europe/Moscow", "moscow": "Europe/Moscow",
              "istanbul": "Europe/Istanbul", "dubai": "Asia/Dubai", "mumbai": "Asia/Kolkata", "delhi": "Asia/Kolkata",
              "beijing": "Asia/Shanghai", "shanghai": "Asia/Shanghai", "tokyo": "Asia/Tokyo", "seul": "Asia/Seoul", "seoul": "Asia/Seoul",
              "singapore": "Asia/Singapore", "sydney": "Australia/Sydney", "cairo": "Africa/Cairo"}


def now(timezone="Europe/Bucharest"):
    """Current date and time.   Used by: run_tool("now: Europe/Bucharest"), run_tool("now:") (= Bucharest)
    Input: an IANA time zone name ("Europe/Bucharest", "America/New_York", "Asia/Tokyo")
    Output: "Saturday, 24 October 2026, 17:53 (Europe/Bucharest)"   (English day and month names)
    Hint: datetime + zoneinfo.ZoneInfo(timezone).   Errors: "error: unknown time zone 'xyz'"."""
    from datetime import datetime
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
    timezone = timezone.strip() or "Europe/Bucharest"
    timezone = TIME_ZONES.get(timezone.lower().replace("_", " "), timezone)
    try:
        moment = datetime.now(ZoneInfo(timezone))
    except (ZoneInfoNotFoundError, ValueError, OSError):
        return f"error: unknown time zone '{timezone}'"
    return f"{moment.strftime('%H:%M')}, {moment.strftime('%A')}, {moment.day} {moment.strftime('%B %Y')} ({timezone})"   # the hour first: the model copies the start


def _load_memory():
    try:
        with open(MEMORY_FILE, encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_memory(memory):
    try:
        with open(MEMORY_FILE, "w", encoding="utf-8") as file:
            json.dump(memory, file, ensure_ascii=False, indent=1)
    except OSError:
        pass


def memory_save(key, value):
    """Remembers a fact about the user (so it can use it later, even in a new chat).
    Used by: run_tool("memory_save: name=Alex"), run_tool("memory_save: age=20")
    Inputs: "name", "Alex"   (key: lowercase letters/_ ; value: text up to 60 characters)
    Output: "saved: name=Alex".   Keeps everything in MEMORY_FILE (load the dict, change it, write it back).
    Errors: "error: empty value"."""
    key = re.sub(r"[^a-z_]", "", key.strip().lower().replace(" ", "_"))
    value = value.strip()[:60]
    if not key or not value:
        return "error: empty value"
    memory = _load_memory()
    memory[key] = value
    _write_memory(memory)
    return f"saved: {key}={value}"


def memory_get(key=""):
    """Reads what was remembered.   Used by: run_tool("memory_get: name"), run_tool("memory_get:") (= everything)
    Input: a key, or "" for all.   Output: "name=Alex"  or  "name=Alex, age=20"  or  "nothing remembered"
    Errors: "error: nothing remembered about 'xyz'" (when a key is asked for and does not exist)."""
    memory = _load_memory()
    key = re.sub(r"[^a-z_]", "", key.strip().lower().replace(" ", "_"))
    if key:
        return f"{key}={memory[key]}" if key in memory else f"error: nothing remembered about '{key}'"
    return ", ".join(f"{name}={value}" for name, value in memory.items()) if memory else "nothing remembered"


def dice(spec="1d6"):
    """Rolls dice.   Used by: run_tool("dice: 2d6")   Input: "<count>d<sides>", 1-20 dice, 2-1000 sides
    Output: "2d6: 3 + 5 = 8"  (one die: "1d20: 17").   Hint: random.randint.   Errors: "error: write it like 2d6"."""
    parts = spec.strip().lower().split("d")
    if len(parts) != 2 or not parts[0].isdigit() or not parts[1].isdigit():
        return "error: write it like 2d6"
    count, sides = int(parts[0]), int(parts[1])
    if not (1 <= count <= 20) or not (2 <= sides <= 1000):
        return "error: write it like 2d6"
    rolls = [random.randint(1, sides) for _ in range(count)]
    label = f"{count}d{sides}"
    if count == 1:
        return f"{label}: {rolls[0]}"
    return f"{label}: " + " + ".join(str(roll) for roll in rolls) + f" = {sum(rolls)}"


def coin():
    return random.choice(["heads", "tails"])

def pick(options):
    """Picks one option at random.   Used by: run_tool("pick: pizza | pasta | sushi")
    Input: the options separated by |  (as one string).   Output: "pasta".   Errors: "error: need at least 2 options"."""
    options = [option.strip() for option in options.split("|") if option.strip()]
    if len(options) < 2:
        return "error: need at least 2 options"
    return random.choice(options)


def today_in_history(month=0, day=0, language="en"):
    """One event that happened on this day in history.   Used by: run_tool("today_in_history:") (= today)
    Inputs: month 1-12 and day 1-31 (0 = today), "en" or "ro".
    Output: "24 October 1929 — Black Thursday: the New York stock market crashes (https://en.wikipedia.org/...)"
    Hint: https://api.wikimedia.org/feed/v1/wikipedia/en/onthisday/events/MM/DD   (JSON list "events", each has "year",
    "text", "pages"); pick one with a short text.   Errors: "error: no internet"."""
    from datetime import date
    if not (1 <= month <= 12 and 1 <= day <= 31):
        today = date.today()
        month, day = today.month, today.day
    language = "ro" if language == "ro" else "en"
    url = f"https://api.wikimedia.org/feed/v1/wikipedia/{language}/onthisday/events/{month:02d}/{day:02d}"
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "CosmosLM-student-project/1.0 (educational)"})
        events = json.load(urllib.request.urlopen(request, timeout=15)).get("events", [])
    except (urllib.error.URLError, OSError, TimeoutError):
        return "error: no internet"
    except Exception:
        return "error: no events found"
    events = [event for event in events if event.get("text") and len(event["text"]) <= 160 and event.get("pages")]
    if not events:
        return "error: no events found"
    event = random.choice(events)
    names = {"en": ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"],
             "ro": ["ianuarie", "februarie", "martie", "aprilie", "mai", "iunie", "iulie", "august", "septembrie", "octombrie", "noiembrie", "decembrie"]}
    page = event["pages"][0].get("content_urls", {}).get("desktop", {}).get("page", "")
    return f"{day} {names[language][month - 1]} {event['year']} — {event['text']}" + (f" ({page})" if page else "")


def run_tool(call_text):
    """Runs one call line written by the model and returns the result text.  THIS is what the chat app calls.
    Input: "calc: 245 * 18"  /  "weather: Cluj-Napoca"  /  "convert: 5 km to miles"  /  "dice: 2d6"  /  "memory_save: name=Alex"
    Output: the tool's result (never longer than MAX_RESULT_CHARS), or "error: ..." (never raises).
    Steps: split at the first ":" -> tool name + argument text; check the name is in TOOL_NAMES; turn the argument text
    into the function's inputs (e.g. "5 km to miles" -> 5.0, "km", "miles"; "name=Alex" -> "name", "Alex");
    call it; cut the result to MAX_RESULT_CHARS.   Unknown tool -> "error: unknown tool 'xyz'"."""
    name, separator, argument = call_text.partition(":")
    name, argument = name.strip().lower(), argument.strip()
    name = {"memorysave": "memory_save", "memoryget": "memory_get", "todayinhistory": "today_in_history"}.get(name, name)   # the model sometimes drops "_"
    if not separator:
        return "error: write the call as 'tool: arguments'"
    if name not in TOOL_NAMES:
        return f"error: unknown tool '{name}'"
    try:
        if name == "calc":
            result = calc(argument)
        elif name == "convert":
            found = re.match(r"^(-?\d+(?:[.,]\d+)?)\s*([^\s]+(?: [^\s]+)?)\s+(?:to|in|în)\s+(.+)$", argument, re.I)
            if not found:
                return "error: write it like '5 km to miles'"
            result = convert(float(found.group(1).replace(",", ".")), found.group(2), found.group(3))
        elif name == "wiki":
            chosen = re.match(r"^(ro|en):\s*(.*)$", argument)          # "wiki: ro: inventat telefonul" (written by tool_router)
            if chosen:
                language, argument = chosen.group(1), chosen.group(2)
            else:
                language = "ro" if is_romanian(argument) else "en"
            result = wiki(argument, language)
            signs = ("may refer to", "se poate referi", "poate însemna", "may also refer", "poate fi:", "can refer to", "most commonly refers")
            if any(sign in result[:300] for sign in signs):
                kept = dict(LAST_WIKI)
                other = wiki(argument, "en" if language == "ro" else "ro")       # a disambiguation page: try the other language
                if not other.startswith("error:") and not any(sign in other[:300] for sign in signs):
                    result = other
                else:
                    LAST_WIKI.clear()
                    LAST_WIKI.update(kept)
        elif name == "weather":
            result = weather(argument)
        elif name == "now":
            result = now(argument)
        elif name == "memory_save":
            key, _, value = argument.partition("=")
            result = memory_save(key, value)
        elif name == "memory_get":
            result = memory_get(argument)
        elif name == "dice":
            result = dice(argument or "1d6")
        elif name == "coin":
            result = coin()
        elif name == "pick":
            result = pick(argument)
        else:
            parts = argument.split()
            numbers = [int(part) for part in parts[:2] if part.isdigit()]
            result = today_in_history(*(numbers if len(numbers) == 2 else (0, 0)))
    except NotImplementedError:
        return f"error: tool '{name}' is not available yet"
    except Exception:
        return f"error: tool '{name}' failed"
    return str(result)[:MAX_RESULT_CHARS]
