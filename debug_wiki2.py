"""One Wikipedia request, prints the answer headers:  python debug_wiki2.py"""
import urllib.parse, urllib.request
for agent in ["CosmosLM-student-project/1.0 (educational)", "CosmosLM/1.0 (student project; https://github.com) python-urllib"]:
    url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode({"action": "query", "list": "search", "srsearch": "NATO", "srlimit": 1, "format": "json"})
    try:
        answer = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": agent}), timeout=15)
        print(agent, "->", answer.status)
    except urllib.error.HTTPError as error:
        print(agent, "->", error.code, dict(error.headers).get("Retry-After"), error.read()[:300])
