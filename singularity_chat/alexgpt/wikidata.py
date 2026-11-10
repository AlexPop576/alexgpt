"""
Wikidata: one TRUE fact for the Wikipedia article the search picked, written as a sentence the model can copy.
Free, no key, the same project as Wikipedia (query.wikidata.org).   Used by tools.wiki().

    wiki() picks the article  ->  its Wikidata id (from the same search request, e.g. "Romeo and Juliet" = Q83186)
    the question picks WHICH fact (FACTS below, first match wins)  ->  one SPARQL request (5 s at most)
        "Who wrote Romeo and Juliet"         -> author P50          -> "Romeo and Juliet was written by William Shakespeare."
        "Martin Garrix's most famous songs"  -> songs he performed  -> "Martin Garrix's best-known songs include Animals, ..."
        "When was Martin Garrix born"        -> date of birth P569  -> "Martin Garrix was born on 14 May 1996."
    No pattern, no answer, too slow, no internet  ->  None: the answer uses Wikipedia alone, as before.

Why: Wikipedia gives a paragraph and a 126M model must find the fact in it (it often picks the wrong name);
Wikidata gives the fact itself.  Lists (songs, albums) are in tables that the Wikipedia text does not contain.
"""

import json
import re
import urllib.parse
import urllib.request

ENDPOINT = "https://query.wikidata.org/sparql"
TIMEOUT = 5
MONTHS = {"en": "January February March April May June July August September October November December".split(),
          "ro": "ianuarie februarie martie aprilie mai iunie iulie august septembrie octombrie noiembrie decembrie".split()}

# (pattern on the folded question, properties tried in order, kind of value, English sentence, Romanian sentence)
# {name} = the article's name, {value} = the value(s).  Patterns are on lower case text without diacritics.
FACTS = [
    (r"\b(songs?|hits?|singles?|tracks?|piese|piesele|melodii|melodiile|cantece|cantecele)\b", ["songs"], "list",
     "{name}'s best-known songs include {value}.", "Cele mai cunoscute piese ale lui {name}: {value}."),
    (r"\b(albums?|albume|albumele)\b", ["albums"], "list",
     "{name}'s best-known albums include {value}.", "Cele mai cunoscute albume ale lui {name}: {value}."),
    (r"\bwho (wrote|authored)\b|\bauthor\b|\bcine a scris\b|\bautorul\b", ["P50", "P86", "P170"], "entity",
     "{name} was written by {value}.", "Autorul lui {name} este {value}."),
    (r"\bwho (composed)\b|\bcomposer\b|\bcine a compus\b|\bcompozitorul\b", ["P86"], "entity",
     "{name} was composed by {value}.", "Compozitorul lui {name} este {value}."),
    (r"\bwho (invented|discovered)\b|\binventor\b|\bcine a (inventat|descoperit)\b|\binventatorul\b", ["P61"], "entity",
     "{name} was invented by {value}.", "{name}: inventat de {value}."),
    (r"\bwho (founded|created|started)\b|\bfounder\b|\bcine a (fondat|infiintat|creat)\b|\bfondatorul\b", ["P112", "P170"], "entity",
     "{name} was founded by {value}.", "{name} a fost fondat de {value}."),
    (r"\bwho directed\b|\bdirector of\b|\bcine a regizat\b|\bregizorul\b", ["P57"], "entity",
     "{name} was directed by {value}.", "{name} a fost regizat de {value}."),
    (r"\bwhere was\b.*\bborn\b|\bbirthplace\b|\bunde s-a nascut\b|\blocul nasterii\b", ["P19"], "entity",
     "{name} was born in {value}.", "{name} s-a născut în {value}."),
    (r"\bwhen was\b.*\bborn\b|\bborn when\b|\bbirthday\b|\bdate of birth\b|\bhow old\b|\bcand s-a nascut\b|\bdata nasterii\b|\bce varsta\b",
     ["P569"], "date", "{name} was born on {value}.", "{name} s-a născut pe {value}."),
    (r"\bwhen did\b.*\bdie\b|\bdate of death\b|\bcand a murit\b|\bdata mortii\b", ["P570"], "date",
     "{name} died on {value}.", "{name} a murit pe {value}."),
    (r"\bwhen was\b.*\b(founded|created|built|established|formed)\b|\bcand a fost (fondat|infiintat|construit|creat)",
     ["P571"], "year", "{name} was founded in {value}.", "{name} a fost fondat în {value}."),
    (r"\bcapital\b|\bcapitala\b", ["P36"], "entity",
     "The capital of {name} is {value}.", "Capitala: {value}."),
    (r"\bpopulation\b|\bhow many (people|inhabitants)\b|\bpopulatia\b|\bcati locuitori\b", ["P1082"], "number",
     "{name} has a population of {value}.", "{name} are o populație de {value} de locuitori."),
    (r"\bcurrency\b|\bmoneda\b|\bvaluta\b", ["P38"], "entity",
     "The currency of {name} is {value}.", "Moneda: {value}."),
    (r"\bofficial language\b|\blimba oficiala\b|\blimbile oficiale\b", ["P37"], "entity",
     "The official language of {name}: {value}.", "Limba oficială: {value}."),
    (r"\bpresident\b|\bpresedinte\b|\bhead of state\b|\bseful statului\b", ["P35", "P1308"], "entity",
     "The head of state of {name}: {value}.", "Șeful statului: {value}."),
    (r"\bprime minister\b|\bprim-ministru\b|\bpremierul\b", ["P6", "P1308"], "entity",
     "The head of government of {name}: {value}.", "Șeful guvernului: {value}."),
    (r"\b(married|wife|husband|spouse)\b|\bsotia\b|\bsotul\b|\bcasatorit\b", ["P26"], "entity",
     "{name} is or was married to {value}.", "{name}: căsătorit(ă) cu {value}."),
    (r"\bhow (tall|high)\b|\bheight\b|\bcat de (inalt|mare)\b|\binaltimea\b|\baltitudinea\b", ["P2048", "P2044"], "metres",
     "{name} is {value} high.", "{name} are {value}."),
]

# songs / albums: what the artist performed (P175), the most linked first (sitelinks = how many Wikipedias have it)
LIST_TYPES = {"songs": ["Q134556", "Q7366", "Q105543609"], "albums": ["Q482994", "Q208569"]}


def fold(text):
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFKD", text.lower()) if not unicodedata.combining(c))


def sparql(query, user_agent):
    url = ENDPOINT + "?" + urllib.parse.urlencode({"query": query, "format": "json"})
    request = urllib.request.Request(url, headers={"User-Agent": user_agent, "Accept": "application/sparql-results+json"})
    return json.load(urllib.request.urlopen(request, timeout=TIMEOUT))["results"]["bindings"]


def join(values, language):
    word = " și " if language == "ro" else " and "
    return values[0] if len(values) == 1 else ", ".join(values[:-1]) + word + values[-1]


def show_date(text, language, year_only=False):
    found = re.match(r"(-?\d{4})-(\d{2})-(\d{2})", text)
    if not found:
        return None
    year, month, day = int(found.group(1)), int(found.group(2)), int(found.group(3))
    if year_only or (month == 1 and day == 1):                   # "1876-01-01" often means "only the year is known"
        return str(year)
    return f"{day} {MONTHS[language][month - 1]} {year}"


def show_number(text, language):
    try:
        number = float(text)
    except ValueError:
        return None
    whole = f"{round(number):,}"
    return whole.replace(",", ".") if language == "ro" else whole


def labels(ids, language, user_agent):
    """{"Q25191": "Christopher Nolan"} for up to 20 ids (one request to www.wikidata.org, no key)"""
    url = "https://www.wikidata.org/w/api.php?" + urllib.parse.urlencode({
        "action": "wbgetentities", "ids": "|".join(ids[:20]), "props": "labels", "languages": "ro|en" if language == "ro" else "en",
        "format": "json"})
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    entities = json.load(urllib.request.urlopen(request, timeout=TIMEOUT)).get("entities", {})
    names = {}
    for item, entity in entities.items():
        found = entity.get("labels", {})
        label = found.get(language) or found.get("en")
        if label:
            names[item] = label["value"]
    return names


def find_item(name, language, user_agent):
    """the Wikidata id of a name ("Taylor Swift" -> "Q26876"), the first search result"""
    url = "https://www.wikidata.org/w/api.php?" + urllib.parse.urlencode({
        "action": "wbsearchentities", "search": name, "language": language, "limit": 1, "format": "json"})
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    found = json.load(urllib.request.urlopen(request, timeout=TIMEOUT)).get("search", [])
    return found[0]["id"] if found else None


def values_for(item, prop, kind, language, user_agent):
    languages = "ro,en" if language == "ro" else "en"
    label = 'SERVICE wikibase:label { bd:serviceParam wikibase:language "' + languages + '". }'
    if prop in LIST_TYPES:
        types = " ".join("wd:" + q for q in LIST_TYPES[prop])
        rows = sparql("SELECT ?value ?valueLabel ?links WHERE { ?value wdt:P175 wd:" + item + " ; wdt:P31 ?type ; wikibase:sitelinks ?links . "
                      "VALUES ?type { " + types + " } " + label + " } ORDER BY DESC(?links) LIMIT 8", user_agent)
    else:
        rows = sparql("SELECT ?value ?valueLabel WHERE { wd:" + item + " wdt:" + prop + " ?value . " + label + " } LIMIT 6", user_agent)
    # the label service sometimes answers with the bare id ("Q25191"): ask the normal Wikidata API for those names
    missing = [row["valueLabel"]["value"] for row in rows if re.fullmatch(r"Q\d+", row.get("valueLabel", {}).get("value", ""))]
    names = labels(missing, language, user_agent) if missing else {}
    values = []
    for row in rows:
        raw = row.get("value", {}).get("value", "")
        shown = row.get("valueLabel", {}).get("value", raw)
        shown = names.get(shown, shown)
        if kind in ("date", "year"):
            shown = show_date(raw, language, year_only=kind == "year")
        elif kind == "number":
            shown = show_number(raw, language)
        elif kind == "metres":
            number = show_number(raw, language)
            shown = number + " m" if number else None
        elif re.fullmatch(r"Q\d+", shown or ""):                  # an item with no name in this language
            shown = None
        if shown and shown not in values:
            values.append(shown)
    return values


def fact_for(item, title, question, language="en", user_agent="CosmosLM", names=None):
    """-> {"sentence", "property", "values", "item"} or None.   item: "Q83186", title: the Wikipedia article,
    names: the names written in the question (["Taylor", "Swift"]): for songs / albums, when the article is a list
    ("List of songs by Taylor Swift"), the artist is looked up by that name."""
    if not item or not re.fullmatch(r"Q\d+", item):
        return None
    folded = fold(question)
    name = re.sub(r"\s*\([^)]*\)$", "", title).strip()           # "Romeo and Juliet (1968 film)" -> "Romeo and Juliet"
    for pattern, props, kind, english, romanian in FACTS:
        if not re.search(pattern, folded):
            continue
        for prop in props:
            try:
                values = values_for(item, prop, kind, language, user_agent)
                if not values and kind == "list" and names:
                    artist = find_item(" ".join(names), language, user_agent)
                    if artist and artist != item:
                        values = values_for(artist, prop, kind, language, user_agent)
                        name = " ".join(names)
            except Exception:
                return None                                      # slow or no internet: Wikipedia alone
            if values:
                limit = 5 if kind == "list" else 3
                if kind in ("date", "year", "number", "metres"):
                    values = values[:1]
                sentence = (romanian if language == "ro" else english).format(name=name, value=join(values[:limit], language))
                return {"sentence": sentence, "property": prop, "values": values[:limit], "item": item}
        return None                                              # the right question, but Wikidata has no value
    return None
