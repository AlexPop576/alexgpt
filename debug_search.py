"""Compares Wikipedia search phrasings:  python debug_search.py"""
import json, time, urllib.parse, urllib.request
tests = [("en", "why is the sky blue"), ("en", "sky blue why"), ("en", "who invented the telephone"), ("en", "invented telephone"),
         ("en", "how long does the moon take to orbit the earth"), ("ro", "cine a inventat telefonul"), ("ro", "telefonul inventat"),
         ("ro", "care e capitala Mongoliei"), ("ro", "Mongolia capitala"), ("en", "how does a CPU work")]
for language, query in tests:
    url = f"https://{language}.wikipedia.org/w/api.php?" + urllib.parse.urlencode({"action": "query", "list": "search", "srsearch": query, "srlimit": 3, "format": "json"})
    try:
        hits = json.load(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "CosmosLM-student-project/1.0 (educational)"}), timeout=15))["query"]["search"]
        print(f"{language} {query!r:55} ->", [hit["title"] for hit in hits])
    except Exception as error:
        print(query, "FAILED", error)
    time.sleep(1.5)
