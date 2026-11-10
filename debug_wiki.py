"""Checks the wiki tool on a few searches and prints the real errors:  python debug_wiki.py"""
import json, time, urllib.parse, urllib.request
from alexgpt import tools
for language, query in [("en", "sky blue"), ("en", "CPU work"), ("ro", "capitala Mongoliei"), ("en", "Martin Garrix")]:
    url = f"https://{language}.wikipedia.org/w/api.php?" + urllib.parse.urlencode({"action": "query", "list": "search", "srsearch": query, "srlimit": 1, "format": "json"})
    start = time.time()
    try:
        data = json.load(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "CosmosLM-student-project/1.0 (educational)"}), timeout=15))
        print(query, "search ok", data["query"]["search"][0]["title"], round(time.time() - start, 1), "s")
    except Exception as error:
        print(query, "search FAILED", type(error).__name__, error, round(time.time() - start, 1), "s")
    start = time.time()
    print("   tool:", tools.wiki(query, language)[:150], round(time.time() - start, 1), "s")
