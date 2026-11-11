"""Small rules around the model's tool calls (used by the web interface and by singularity_chat/):
  find_expression:   an arithmetic expression the user wrote ("2+6", "15% of 240") -> always goes to the calc tool,
                     a 126M model is not reliable at deciding that for tiny sums;
  check_memory_call: a memory_save is kept only if the value really appears in the user's message
                     (the model once saved name=Tristan after "sunt trist azi")."""
import re

from alexgpt import tools

EXPRESSION = re.compile(r"(?<![\w.])\(*\d+(?:[.,]\d+)?\)*(?:\s*(?:[+*/x×÷^]|\s-\s)\s*\(*\d+(?:[.,]\d+)?\)*)+(?![\w])")
PERCENT = re.compile(r"(\d+(?:[.,]\d+)?)\s*%\s*(?:of|din|from)\s*(\d+(?:[.,]\d+)?)", re.I)


def find_expression(user_text):
    found = PERCENT.search(user_text)
    if found:
        candidate = f"{found.group(1)}% of {found.group(2)}".replace(",", ".")
    else:
        match = EXPRESSION.search(user_text)
        if not match:
            return None
        candidate = match.group(0).replace("×", "*").replace("÷", "/").replace("^", "**").replace(",", ".")
        candidate = re.sub(r"(?<=[\d)])\s*x\s*(?=[\d(])", " * ", candidate)
        candidate = re.sub(r"(?<=[\d)])\s*([+*/]|\*\*)\s*(?=[\d(])", r" \1 ", candidate)
        if re.fullmatch(r"\d{1,4}\s*-\s*\d{1,2}\s*-\s*\d{1,4}", candidate):      # a date
            return None
    return None if tools.calc(candidate).startswith("error") else candidate


# math written in words: "sum of 3 and 6", "7 times 8", "diferența dintre 10 și 4" -> "3 + 6", "7 * 8", "10 - 4"
# (without this the model decided alone and once wrote "calc: 3 ** 6" for "sum of 3 and 6")
_N = r"(-?\d+(?:[.,]\d+)?)"
WORD_MATH = [(re.compile(pattern.replace("N", _N)), operator) for pattern, operator in [
    (r"(?<![a-z])(?:sum|total|suma|adunarea)\s+(?:of\s+|lui\s+|dintre\s+)?N\s+(?:and|si|cu|plus)\s+N", "+"),
    (r"(?<![a-z])(?:add|aduna)\s+N\s+(?:and|to|si|cu|la)\s+N", "+"),
    (r"N\s+plus\s+N", "+"),
    (r"(?<![a-z])(?:difference|diferenta)\s+(?:between\s+|of\s+|dintre\s+|lui\s+)?N\s+(?:and|si)\s+N", "-"),
    (r"(?<![a-z])(?:subtract)\s+N\s+from\s+N", "from"),
    (r"N\s+minus\s+N", "-"),
    (r"(?<![a-z])(?:product|produsul)\s+(?:of\s+|lui\s+|dintre\s+)?N\s+(?:and|si|cu)\s+N", "*"),
    (r"(?<![a-z])(?:multiply|inmulteste)\s+N\s+(?:and|by|with|cu|si)\s+N", "*"),
    (r"N\s+(?:times|multiplied by|ori|inmultit cu)\s+N", "*"),
    (r"(?<![a-z])(?:divide|imparte)\s+N\s+(?:by|la)\s+N", "/"),
    (r"N\s+(?:divided by|impartit la|over)\s+N", "/"),
    (r"N\s+(?:to the power of|to the power|la puterea)\s+N", "**"),
]]


def word_expression(low_plain):
    """'what is the sum of 2 and 6' -> '2 + 6';  None if no math in words.  low_plain: lower case, no diacritics."""
    for pattern, operator in WORD_MATH:
        found = pattern.search(low_plain)
        if found:
            a, b = (number.replace(",", ".") for number in found.groups())
            candidate = f"{b} - {a}" if operator == "from" else f"{a} {operator} {b}"
            if not tools.calc(candidate).startswith("error"):
                return candidate
    return None


def check_memory_call(call, user_text):
    """-> the call (maybe corrected), or None if it must not run"""
    name, _, argument = call.partition(":")
    if name.strip().replace("_", "") != "memorysave":
        return call
    if not re.search(r"tine minte|ține minte|retine|reține|remember|sunt din|locuiesc|ma cheama|mă cheamă|my name is|i live|i am from|i'm from|"
                     r"imi place|îmi place|i like|call me|spune-mi\s+\w+$|name is", user_text.lower()):
        return None                                     # "ora in Paris" must not save city=Paris
    key, _, value = argument.partition("=")
    value = value.strip()
    if not value:
        return None
    if re.search(r"(?<!\w)" + re.escape(value.lower()) + r"(?!\w)", user_text.lower()):
        return f"memory_save: {key.strip()}={value}"
    for word in re.findall(r"\w+", user_text):          # "Alex" written by the user, "Alexandru" by the model
        if word[0].isupper() and len(word) >= 3 and value.lower().startswith(word.lower()):
            return f"memory_save: {key.strip()}={word}"
    return None


# ---------------------------------------------------------------- plain rules before the model
# A 126M model decides tools unreliably.  Clear requests (EN + RO) are routed by rules: route() -> list of call lines.
NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
                "un": 1, "o": 1, "doi": 2, "două": 2, "doua": 2, "trei": 3, "patru": 4, "cinci": 5, "șase": 6, "sase": 6,
                "șapte": 7, "sapte": 7, "opt": 8, "nouă": 9, "noua": 9, "zece": 10, "a": 1, "an": 1}
DAY_WORDS = r"(?:azi|astăzi|astazi|today|tonight|mâine|maine|tomorrow|joi|vineri|sâmbătă|sambata|duminică|duminica|luni|marți|marti|miercuri|" \
            r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|acum|now|diseară|diseara)"


def _clean_place(text):
    text = re.sub(r"\b" + DAY_WORDS + r"\b", " ", text, flags=re.I)
    return re.sub(r"[?!.]+", " ", text).strip(" ,-")


def _dice_call(text):
    low = text.lower()
    spec = re.search(r"\b(\d{1,2})\s*d\s*(\d{1,3})\b", low)
    if spec:
        return f"dice: {int(spec.group(1))}d{int(spec.group(2))}"
    if not re.search(r"\b(dice|die|zar|zaruri|zarul|zaruri)\b", low):
        return None
    words = "|".join(sorted(NUMBER_WORDS, key=len, reverse=True))
    found = re.search(r"\b(\d{1,2}|" + words + r")\s+(?:dice|die|dices|zar|zaruri)\b", low)
    count = 1
    if found:
        token = found.group(1)
        count = int(token) if token.isdigit() else NUMBER_WORDS[token]
    return f"dice: {max(1, min(count, 20))}d6"


def route(user_text):
    """-> a list of call lines ("calc: sqrt(9)", "wiki: Cluj-Napoca"), [] if no rule fits."""
    text = user_text.strip()
    low = text.lower()
    low_plain = low.replace("ă", "a").replace("â", "a").replace("î", "i").replace("ș", "s").replace("ş", "s").replace("ț", "t").replace("ţ", "t")

    square_root = re.search(r"(?:radical|radacina(?: patrata)?|square root)\s*(?:din|of)?\s*(\d+(?:[.,]\d+)?)"
                            r"(?:\s*(?:la puterea|to the power(?: of)?|la|\^)\s*(\d+))?(?:\s*([x*+/-])\s*(\d+(?:[.,]\d+)?))?", low_plain)
    if square_root:
        call = f"sqrt({square_root.group(1).replace(',', '.')})"
        if square_root.group(2) or square_root.group(3):         # "radical din 16 la puterea 3": the model would only
            root = tools.calc(call)                              # say "radical din 16 = 64", so it gets the plain sum "4 ** 3"
            call = root
            if square_root.group(2):
                call += f" ** {square_root.group(2)}"
            if square_root.group(3):
                call += f" {'*' if square_root.group(3) == 'x' else square_root.group(3)} {square_root.group(4).replace(',', '.')}"
        if not tools.calc(call).startswith("error"):
            return ["calc: " + call]

    expression = find_expression(text) or word_expression(low_plain)
    if expression:
        return ["calc: " + expression]

    if re.search(r"flip a coin|toss a coin|heads or tails|coin flip|arunca? (?:o )?moneda|cap sau pajura", low_plain):
        return ["coin:"]
    if re.search(r"\b(roll|throw|toss|arunca|dau|da|rostogoleste)\b", low_plain) or re.search(r"\b\d{1,2}d\d{1,3}\b", low_plain):
        call = _dice_call(text)
        if call:
            return [call]

    if re.search(r"today in history|on this day|what happened today|ce s-a intamplat (?:azi|astazi|in aceasta zi)|azi in istorie|in istorie azi|"
                 r"ceva ce s-a intamplat (?:azi|astazi)|intamplat azi in istorie", low_plain):
        return ["today_in_history:"]

    # units: "10 km to miles", "0 celsius to fahrenheit", "55 celcius in kelvin"
    unit = re.search(r"(-?\d+(?:[.,]\d+)?)\s*°?\s*([a-zA-Z/°]+(?: [a-z]+)?)\s+(?:to|in|into|în|la|spre)\s+([a-zA-Z/°]+)", text, re.I)
    if unit and tools.convert(float(unit.group(1).replace(",", ".")), unit.group(2), unit.group(3)).count("=") == 1:
        return [f"convert: {unit.group(1)} {unit.group(2)} to {unit.group(3)}"]

    # the time: "what time is it in New York", "ora in Paris", "ce ora e"
    if re.search(r"what(?:'s| is)? the (?:time|hour)|what(?:'s| s)? the time|what time|current time|the hour\b|\bce ora\b|\bcat e ora\b|\bcat e ceasul\b|\bora (?:in|din|la|exacta)\b|^\s*(?:spune-mi )?ora\b|\bspune-mi ora\b|time in\b", low_plain) \
            and not re.search(r"(?<!nu )vrem|weather|temperatur", low_plain):
        place = re.search(r"(?:\bin|\bat|\bdin|\bla)\s+([a-z][a-z .'-]+?)\s*[?!.]*\s*$", low_plain)
        if place:
            zone = tools.TIME_ZONES.get(_clean_place(place.group(1)).lower()) or tools.TIME_ZONES.get(_clean_place(place.group(1)).lower().replace(" ", "-"))
            if zone:
                return ["now: " + zone]
            return []
        return ["now: Europe/Bucharest"]

    # the weather: "weather in Malaga", "vremea in Madrid si Marbella"
    weather = re.search(r"(?:weather|vremea|temperatura|meteo)\b.*?\b(?:in|at|for|la|din|pentru)\s+(.+)$", text, re.I)
    if weather or re.search(r"^\s*(?:weather|vremea)\s+\w", low_plain):
        place = _clean_place(weather.group(1)) if weather else _clean_place(re.sub(r"^\s*(?:weather|vremea)\s+", "", text, flags=re.I))
        cities = [c.strip() for c in re.split(r"\s+(?:si|și|and|sau|or)\s+|,", place) if c.strip()]
        if cities and len(cities) <= 3 and all(len(c.split()) <= 3 for c in cities):
            return ["weather: " + c for c in cities]

    # pick: "alege dintre pizza si burgeri"
    choice = re.search(r"\b(?:alege|choose|pick|decide)\s+(?:dintre|intre|between|from|one of|din)?\s*(.+)$", low_plain)
    if choice:
        options = [o.strip(" .?!") for o in re.split(r"\s+(?:si|or|sau|and)\s+|,|\|", choice.group(1)) if o.strip(" .?!")]
        if len(options) >= 2:
            return ["pick: " + " | ".join(options)]

    # memory: only a clear sentence is saved ("ține minte că mă cheamă Alex", "my name is Alex", "locuiesc în Cluj")
    name = re.search(r"(?:ma cheama|numele meu (?:e|este)|my name is|call me|i am called)\s+([A-Za-zĂÂÎȘȚăâîșț-]{2,20})\s*[.!]*\s*$", text.replace("mă", "ma"), re.I)
    if name and re.search(r"tine minte|remember|ma cheama|my name is|call me", low_plain):
        return [f"memory_save: name={name.group(1)}"]
    city = re.search(r"(?:locuiesc (?:in|la)|sunt din|i live in|i am from|i'm from)\s+([A-Za-zĂÂÎȘȚăâîșț -]{2,30}?)\s*[.!]*\s*$", text, re.I)
    if city:
        return [f"memory_save: city={city.group(1).strip()}"]

    # memory questions
    if re.search(r"ce stii despre mine|what do you know about me|cum ma cheama|care e numele meu|what(?:'s| is) my name|ce ti-am spus", low_plain):
        return ["memory_get:"]

    # Wikipedia: "tell me about X", "spune-mi despre X", "search on wikipedia about X", "who is X", "ce este X"
    topic = None
    for pattern in (r"wikipedia\s*(?:about|for|despre|pentru)?\s+(.+)$", r"(?:cauta|search|find)\b.*?\b(?:about|despre)\s+(.+)$",
                    # "write a text about Ultra Miami", "scrie-mi un eseu despre Eminescu": the facts come from Wikipedia
                    r"(?:write|compose|give me|scrie|scrie-mi|fa-mi|compune)\s+(?:me\s+)?(?:a|an|un|o)?\s*(?:short\s+|long\s+|scurt\s+|lung\s+)?"
                    r"(?:text|essay|paragraph|article|summary|description|presentation|eseu|compunere|paragraf|articol|rezumat|"
                    r"descriere|prezentare)\s+(?:about|on|despre|cu|pe)\s+(.+)$",
                    r"(?:tell me (?:more )?about|spune-mi (?:mai mult )?despre|povesteste-mi despre|vorbeste-mi despre|"
                    r"explain|explica-mi|who is|who was|cine este|cine e|cine a fost|what is|what are|ce este|ce e|ce sunt|"
                    r"descrie|^despre)\s+(.+)$"):       # a bare "despre" only at the start: "ce a spus Eminescu despre Japonia" is a question
        found = re.search(pattern, low_plain)
        if found:
            topic = text[found.start(1):]       # same length as low_plain: keep the user's own capitals and diacritics
            break
    if topic:
        words = re.sub(r"[?!.]+\s*$", "", topic).strip()
        words = re.sub(r"^(?:the\s+)?(?:city|town|country|village|orasul|orașul|tara|țara|satul)\s+(?:of\s+|de\s+)?", "", words, flags=re.I).strip()
        words = re.sub(r"^(?:the|a|an|un|o|niste)\s+", "", words, flags=re.I).strip()
        blocked = re.search(r"\b(my|your|me|you|tau|tale|meu|mea|tu|name|age|joke|story|poveste|gluma|time|ora|weather|vremea|"
                            r"today|azi|this|that|asta|aia|it|more|mai mult|aceasta)\b", words.lower())
        if words and not blocked and len(words.split()) <= 5:
            language = "ro" if tools.is_romanian(text) else "en"     # the language of the whole message, not of the name
            return [f"wiki: {language}: {words}"]
    return []


# ---------------------------------------------------------------- questions that need facts -> Wikipedia
# A 126M model invents facts.  An ordinary question ("how long does the Moon take to orbit the Earth?") is looked up
# first; route_question() -> (call, explicit).  explicit = the user asked for a search: the result is used even if weak.
QUESTION_START = re.compile(r"^(?:and |but |so |si |și |dar )?(?:who|whom|what|when|where|which|why|how|is|are|was|were|did|does|do|"
                            r"cine|ce|cand|când|unde|care|de ce|cum|cat|cât|câte|câți|cati|cate|in ce|în ce)\b", re.I)
SEARCH_START = re.compile(r"^(?:please |te rog )?(?:search|look up|lookup|google|find out|find|cauta|caută|gaseste|găsește)"
                          r"(?:\s+(?:on|pe|in|în)\s+(?:wikipedia|wiki|the internet|internet|the web|web|google))?"
                          r"(?:\s+(?:for|about|despre|după|dupa|info about|information about))?\s*(.*)$", re.I)
NOT_FACTS = re.compile(r"\b(you|your|yours|yourself|i|me|my|mine|we|us|our|tu|te|tău|tau|ta|tale|eu|meu|mea|mele|noi|nostru|"
                       r"joke|story|poem|essay|write|code|function|program|script|recommend|should|would|could|opinion|think|favorite|"
                       r"favourite|feel|weather|time|today|tomorrow|gluma|glumă|poveste|poezie|eseu|scrie|cod|funcție|functie|program|"
                       r"recomanzi|recomanda|crezi|parere|părere|preferat|preferata|simti|simți|faci|esti|ești|vremea|vreme|ora|azi|mâine|maine|"
                       r"going|up|doing|hello|hi|hey|salut|buna|bună|"
                       r"pot|poți|poti|putem|tonight|diseară|diseara)\b", re.I)     # "ce pot să gătesc diseară?" = advice, not a fact
RO_WORDS = re.compile(r"[ăâîșțşţ]|\b(cine|ce|care|cum|unde|cand|cât|cat|este|sunt|despre|pentru|din|si|sau|lui|de ce|a fost|inventat|"
                      r"capitala|tara|țara|orasul|orașul)\b", re.I)


def route_question(user_text, previous_user_text=""):
    """-> (call, explicit) or (None, False)"""
    text = user_text.strip()
    search = SEARCH_START.match(text)
    if search:
        topic = search.group(1).strip(" ?!.")
        if not topic or re.fullmatch(r"(it|that|this|asta|aia|acest lucru|that up|it up)", topic, re.I):
            topic = previous_user_text.strip()                 # "search on wikipedia" right after a question: that question
        if not topic:
            return None, False
        return wiki_call(topic), True
    if LIST_REQUEST.match(text) and not re.search(r"\b(you|your|my|tu|tale|mele)\b", text, re.I):
        return wiki_call(text), False                   # "tell me songs by Calvin Harris": lists get invented without a source
    if LIST_THINGS.search(text) and named_entities(text) and not re.search(r"\b(you|your|my|tu|tale|mele)\b", text, re.I):
        return wiki_call(text), False                   # "Martin Garrix most famous songs": a name + songs, no verb
    if not QUESTION_START.match(text) or is_not_factual(text):
        return None, False
    from alexgpt.tools import _terms
    if not _terms(text):
        return None, False
    return wiki_call(text), False


def is_not_factual(text):
    """True if the question is about the user/the bot or asks for something creative.
    A word written in capitals is an acronym, not a pronoun: "EU" (European Union) is not "eu", "US" is not "us"."""
    for match in NOT_FACTS.finditer(text):
        word = match.group(0)
        if len(word) > 1 and word.isupper():
            continue
        return True
    return False


def wiki_call(question):
    """'cine a inventat telefonul?' -> 'wiki: ro: cine a inventat telefonul'"""
    question = re.sub(r"[?!.]+\s*$", "", question.strip())
    language = "ro" if tools.is_romanian(question) else "en"
    return f"wiki: {language}: {question}"          # the whole question: Wikipedia's search ranks it better than keywords


# ---------------------------------------------------------------- follow-up questions
# "what is the weather in Spain?" ... "but in Cluj?"  -> the same tool, the new place.
# "who is Alesso?" ... "what songs does he have?"     -> "he" becomes "Alesso" before searching.
FOLLOW_UP = re.compile(r"^(?:and|but|what about|how about|and what about|si|și|dar|iar|ce zici de|ce zici despre)\s+"
                       r"(?:in|at|for|la|în|din|pentru|despre|about)?\s*([^\s?!.,][^?!.]{0,40}?)\s*[?!.]*$", re.I)
PRONOUNS = re.compile(r"\b(he|she|they|him|them|his|her|their|el|ea|ei|lui|lor|dânsul|dansul)\b", re.I)
LIST_REQUEST = re.compile(r"^(?:tell me|give me|name|list|show me|spune-mi|zi-mi|dă-mi|da-mi|numește|numeste|enumeră|enumera)\b.*\b"
                          r"(songs?|albums?|movies?|films?|books?|works|games|singles?|tracks?|melodii|melodie|melodiile|piese|piesă|piesa|piesele|albume|albumele|"
                          r"filme|filmele|cărți|carti|cărțile|jocuri|opere)\b", re.I)
LIST_THINGS = re.compile(r"\b(songs?|albums?|singles?|hits|melodii|melodiile|piese|piesele|albume|albumele)\b", re.I)


def route_followup(user_text, last_call):
    """'but in cluj?' after 'weather: Spain' -> 'weather: cluj'.  None if it is not such a question."""
    found = FOLLOW_UP.match(user_text.strip())
    if not found or not last_call:
        return None
    tool = last_call.partition(":")[0].strip()
    place = found.group(1).strip()
    if tool == "weather":
        return "weather: " + place
    if tool == "now":
        zone = tools.TIME_ZONES.get(place.lower())
        return "now: " + zone if zone else None
    if tool == "wiki":
        return wiki_call(place)
    return None


def with_topic(user_text, topic):
    """'what songs does he have' + 'Alesso' -> 'what songs does Alesso have'  (the last Wikipedia title)"""
    if not topic:
        return user_text
    about_it = re.search(r"\b(about|on|despre)\s+(it|this|that|asta|aia|el|ea)\s*[?.!]*\s*$", user_text, re.I)
    if about_it:                                       # "write an essay about it" -> "... about Ultra Music Festival"
        return user_text[:about_it.start(2)] + topic + user_text[about_it.end(2):]
    if not PRONOUNS.search(user_text):
        return user_text

    def replace(match):
        word = match.group(1).lower()
        return topic + "'s" if word in ("his", "her", "their") else topic
    return PRONOUNS.sub(replace, user_text, count=1)


# ---------------------------------------------------------------- no source -> no guess
# A factual question that Wikipedia could not answer is NOT handed to the model: at 126M it invents
# ("most visited country -> China").  The code writes an honest fixed reply instead.
NO_SOURCE_REPLY = {
    "en": "I couldn't find a reliable source for that, so I'd rather not guess.",
    "ro": "Nu am găsit o sursă sigură pentru asta, așa că prefer să nu ghicesc.",
}


BUSY_REPLY = {
    "en": "Wikipedia is busy right now, so I can't check that. Try again in a minute.",
    "ro": "Wikipedia e ocupată acum, așa că nu pot verifica. Încearcă din nou peste un minut.",
}


def no_source_reply(user_text, busy=False):
    """the fixed reply in the user's language ('unde e Romania?' -> Romanian); busy = Wikipedia answered 429"""
    language = "ro" if tools.is_romanian(user_text) else "en"
    return (BUSY_REPLY if busy else NO_SOURCE_REPLY)[language]


# ---------------------------------------------------------------- history gate (strategy 3, docs/fable_plan.md)
# A 126M model copies the style of whatever is in its context (a long wiki answer -> long lists everywhere).
# So the chat history is sent ONLY when the new message clearly continues the last one.
CONTINUES = re.compile(r"^(?:and|but|so|also|then|or|ok|okay|what about|how about|dar|și|si|iar|deci|apoi|sau|ce zici|"
                       r"the one|that one|the other|the same|cel din|cea din|cel de|cea de|ăla|aia|ala)\b", re.I)    # "the one in Jibou"
SHORT_FOLLOW_UP = re.compile(r"^(?:why|how come|really|are you sure|is that true|more|tell me more|go on|continue|explain|"
                             r"for example|an example|de ce|serios|chiar|esti sigur|ești sigur|mai mult|spune-mi mai mult|"
                             r"continua|continuă|explica|explică|adica|adică|de exemplu|un exemplu)\b", re.I)
REFERENCE_WORDS = re.compile(r"\b(he|she|it|they|him|them|his|her|their|its|this|that|these|those|there|the one|"
                             r"el|ea|ei|ele|lui|lor|asta|aia|ăsta|ăla|acesta|aceasta|acela|aceea|acolo|dânsul|dansul)\b", re.I)


def named_entities(text):
    """Capitalized words that are not the first word ('who is Alesso?' -> ['Alesso']) and acronyms ('EU').
    The first word is skipped: it is capitalized only because it starts the sentence."""
    words = re.findall(r"[\wăâîșțĂÂÎȘȚ'-]+", text)
    found = []
    for position, word in enumerate(words):
        if word == "I":
            continue
        if (position > 0 and word[0].isupper()) or (len(word) > 1 and word.isupper()):
            found.append(word)
    return found


def is_follow_up(user_text):
    """-> (True/False, reason).  True = the message needs the earlier messages to make sense."""
    text = user_text.strip()
    if CONTINUES.match(text):
        return True, "starts with a connector"
    if SHORT_FOLLOW_UP.match(text) and len(text.split()) <= 6:
        return True, "short follow-up phrase"
    reference = REFERENCE_WORDS.search(text)
    if reference and not named_entities(text):
        return True, f"refers back ('{reference.group(0)}') and names nothing"
    return False, "stands on its own"


ENTITY_SPAN = re.compile(r"[A-ZĂÂÎȘȚ][\wăâîșț'-]*(?:\s+(?:(?:cel|cea|de|of|the|din|la|von|van|da|del)\s+)?[A-ZĂÂÎȘȚ][\wăâîșț'-]*)*")


def main_entity(text):
    """The longest capitalized name in the message, for the topic memory:
    'cine a fost Stefan cel Mare?' -> 'Stefan cel Mare'.  A single capital word at the very start is skipped
    (it is just the start of the sentence), unless it is an acronym ('EU')."""
    best = None
    for match in ENTITY_SPAN.finditer(text.strip()):
        span = match.group(0)
        if match.start() == 0 and " " not in span and not (len(span) > 1 and span.isupper()):
            continue
        if span == "I":
            continue
        if best is None or len(span) > len(best):
            best = span
    return best


THE_ONE = re.compile(r"^(?:the one|that one|the ones|cel|cea|cele|ăla|aia|ala)\s+(?:in|from|at|near|din|de la|de lângă|de langa|de)\s+"
                     r"([^?!.]{2,40}?)\s*[?!.]*$", re.I)


def the_one_in(user_text, previous_user_text):
    """'the one in Jibou' after 'what is the Wesselenyi castle?' -> 'what is the Wesselenyi castle in Jibou'.
    None if the message is not like that or there is no earlier question."""
    found = THE_ONE.match(user_text.strip())
    if not found or not previous_user_text:
        return None
    place = found.group(1).strip()
    linker = "din" if tools.is_romanian(previous_user_text) else "in"
    return re.sub(r"[?!.]+\s*$", "", previous_user_text.strip()) + f" {linker} {place}"


def who_do_you_mean(user_text):
    """'what songs does he have?' with nobody named before -> ask instead of searching"""
    if tools.is_romanian(user_text):
        return "La cine te referi? Spune-mi numele și caut."
    return "Who do you mean? Tell me the name and I'll look it up."
