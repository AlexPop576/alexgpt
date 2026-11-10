"""Strategy 4 test (docs/fable_plan.md): does the search find the RIGHT article, and say "no source" for false premises?
Only the tool step runs (SingularityChat.run_tools), no answer is generated.  Live Wikipedia.
    .venv\\Scripts\\python.exe test_search.py
FALSE premise: these two things never go together -> must end with NO SOURCE (abstain).
TRUE: the article title must contain one of the expected words."""
import re
import time
import unicodedata

from alexgpt.chat import SingularityChat

FALSE_PREMISE = [
    "when did Napoleon visit Cluj?", "what did Mihai Eminescu say about Tokyo?", "when did Albert Einstein live in Bucharest?",
    "who did Stefan cel Mare fight in Brazil?", "when did the Beatles play in Jibou?", "how did Shakespeare describe Romania?",
    "când a vizitat Napoleon Clujul?", "when did Elon Musk buy Dacia?", "who built the Eiffel Tower in Cluj?",
    "when did Mozart visit Sibiu?", "when did Nikola Tesla live in Iasi?", "why did Dracula build the Eiffel Tower?",
    "when did Taylor Swift perform in Jibou?", "când a cântat Michael Jackson în Jibou?", "ce a spus Eminescu despre Japonia?",
]
TRUE = [
    ("who invented the telephone?", "telephone|bell"), ("where is Romania?", "romania"),
    ("when did Romania join the EU?", "romania|european|enlargement"), ("who wrote Hamlet?", "hamlet|shakespeare"),
    ("how tall is Mount Everest?", "everest"), ("who painted the Mona Lisa?", "mona"),
    ("what is the capital of Australia?", "australia|canberra"), ("cine a scris Luceafarul?", "luceaf|eminescu"),
    ("unde este Cluj-Napoca?", "cluj"), ("când a murit Stefan cel Mare?", "stefan|stephen"),
    ("who discovered penicillin?", "penicillin|fleming"), ("when was Napoleon born?", "napoleon"),
    ("what is the longest river in Europe?", "river|volga|danube"), ("ce este NATO?", "nato|atlantic|tratat"),
    ("who is Alesso?", "alesso"),
]


def fold(text):
    return "".join(c for c in unicodedata.normalize("NFKD", text.lower()) if not unicodedata.combining(c))


model = SingularityChat("runs/singularity_sft4/final.pt")


def search(question):
    calls = []
    for event in model.run_tools(question, []):
        if event["type"] == "tool":
            calls.append(event["call"] + (" [unused]" if event.get("unused") else "")
                         + (" [" + event["result"][:45] + "]" if event["result"].startswith("error") else ""))
    title = model.wiki_details["title"] if model.wiki_details and not model.abstain else None
    return title, calls


false_ok, true_ok = 0, 0
print("=== FALSE PREMISE (must say no source) ===")
for question in FALSE_PREMISE:
    time.sleep(3)
    title, calls = search(question)
    false_ok += title is None
    print(f"  {'OK   ' if title is None else 'WRONG'} {question:45s} -> {title!r}  {calls}")
print("\n=== TRUE (right article) ===")
for question, expected in TRUE:
    time.sleep(3)
    title, calls = search(question)
    good = title is not None and re.search(expected, fold(title)) is not None
    true_ok += good
    print(f"  {'OK   ' if good else 'WRONG'} {question:45s} -> {title!r}  {calls}")
print(f"\nfalse premise -> no source: {false_ok}/{len(FALSE_PREMISE)}   true -> right article: {true_ok}/{len(TRUE)}")
