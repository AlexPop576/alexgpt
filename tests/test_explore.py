"""Checks the Explore backend (alexgpt/explore.py + the tracer in alexgpt/chat.py).

    .venv\\Scripts\\python.exe tests/test_explore.py            # needs the GPU (loads Singularity), no internet
    .venv\\Scripts\\python.exe tests/test_explore.py --online   # also a Wikipedia question

What it checks:
  1. identity_rag.match() still gives the same answer after the ranked() refactor
  2. tools.explain_candidates() keeps / rejects search results like _choose_article
  3. on the real model: the router checks come in the code's order, a calculator message gives a calc tool,
     the steps have 16 blocks x 12 heads, the chat state is restored, and the normal chat is unchanged (tracer = None)
"""
import json
import os
import sys

sys.path.insert(0, ".")
import numpy as np  # noqa: E402

from alexgpt import identity_rag, tools  # noqa: E402

ONLINE = "--online" in sys.argv
results = {"pass": 0, "fail": 0}


def check(name, ok, detail=""):
    results["pass" if ok else "fail"] += 1
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}" + ("" if ok else f"   {detail}"))


print("identity_rag")
index = identity_rag.index()
for text in ["Who made you?", "what is your name", "cine te-a creat?", "what is the capital of France?"]:
    group, score, question = index.match(text)
    ranked = index.ranked(text)
    check(f"match == ranked()[0] for {text!r}", ranked and (group, score) == (ranked[0][0], ranked[0][1]), f"{group} {score} vs {ranked[:1]}")
check("'Who made you?' is accepted", identity_rag.reply("Who made you?", "Singularity")[0] is not None)

print("explain_candidates")
candidates = [
    {"title": "Marie of Romania", "intro": "Queen of Romania, born in England.", "aliases": [], "rank": 0},
    {"title": "Napoleon", "intro": "Napoleon visited many cities, among them Cluj in 1810.", "aliases": [], "rank": 1},
]
explained = tools.explain_candidates(candidates, "when did Napoleon visit Cluj", ["Napoleon", "Cluj"])
check("an article without the names is rejected", explained[0]["kept"] is False and "Napoleon" in explained[0]["why"], explained[0])
check("an article with both names is kept", explained[1]["kept"] is True, explained[1])
check("the same article wins as before", tools._choose_article(candidates, "when did Napoleon visit Cluj", ["Napoleon", "Cluj"])["title"] == "Napoleon")

print("the model (loads Singularity)")
from alexgpt.chat import MODEL_REGISTRY, generate_stream  # noqa: E402
from alexgpt.explore import explore  # noqa: E402

model_class, paths, _ = MODEL_REGISTRY["Singularity"]
path = next((p for p in paths if os.path.exists(p)), None)
if path is None:
    print("  skipped: no Singularity checkpoint")
else:
    model = model_class(path)
    model.last_call, model.topic = "weather: Paris", "Paris"           # pretend the chat had a weather question
    rng = np.random.default_rng(0)

    record = explore(model, "What is 17*23?", [], 0.7, rng)
    gates = [gate["gate"] for gate in record["gates"]]
    check("router checks in the code's order", gates == ["followup", "rules"], gates)
    check("the rules check says yes", record["gates"][1]["hit"] is True, record["gates"])
    check("one calc tool with 391", len(record["tools"]) == 1 and record["tools"][0]["tool"] == "calc" and "391" in record["tools"][0]["result"], record["tools"])
    check("tokens with kinds", record["tokens"] and {"special", "system", "text", "ctx"} <= {t["kind"] for t in record["tokens"]},
          {t["kind"] for t in record["tokens"]})
    check("a vector for every token id", all(str(t["id"]) in record["vectors"] for t in record["tokens"]))
    first = record["steps"][0] if record["steps"] else {}
    check("at least one step", len(record["steps"]) > 0)
    check("16 blocks x 12 heads in the first step", len(first.get("blocks", [])) == 16 and len(first["blocks"][0]["heads"]) == 12,
          len(first.get("blocks", [])))
    check("a lens guess for every block", len(first.get("lens", [])) == 16)
    check("8 embedding numbers of the newest token", len(first.get("embedding", [])) == 8)
    check("later steps are light (no heads)", all("blocks" not in step for step in record["steps"][3:]))
    check("the JSON is not too big", len(json.dumps(record)) < 400_000, len(json.dumps(record)))
    check("the chat's state is restored", (model.last_call, model.topic) == ("weather: Paris", "Paris"), (model.last_call, model.topic))
    check("the tracer is off after the run", model.tracer is None)

    record = explore(model, "Who made you?", [], 0.7, rng)
    check("identity: a fixed reply, no steps", record["direct"] and record["direct"]["kind"] == "identity" and not record["steps"], record["direct"])
    check("identity: the closest questions with scores", record["direct"] and len(record["direct"]["matches"]) == 4)
    check("identity: checks followup, rules, identity", [g["gate"] for g in record["gates"]] == ["followup", "rules", "identity"],
          [g["gate"] for g in record["gates"]])

    record = explore(model, "And in Cluj?", [{"role": "user", "text": "What's the weather in Bucharest?"},
                                              {"role": "assistant", "text": "In Bucharest it is 9°C."}], 0.7, rng)
    check("follow-up after Explore's own weather call is seen as follow-up only if Explore called weather",
          record["gates"][0]["gate"] == "followup", record["gates"][:1])

    # the normal chat must not change: same events with the tracer off
    model.tracer = None
    events = [e for e in generate_stream(model, "What is 17*23?", 40, 0.7, np.random.default_rng(1), [])]
    check("normal chat: still a calc tool event", any(e.get("type") == "tool" and e["tool"] == "calc" for e in events))
    check("normal chat: no 'galaxy' data without visualize", not any("galaxy" in e for e in events))

    if ONLINE:
        record = explore(model, "Who invented the telephone?", [], 0.7, rng)
        wiki = [t for t in record["tools"] if t["tool"] == "wiki"]
        check("wiki: a search with results and sentences", wiki and wiki[0]["search"]["results"] and wiki[0]["search"]["sentences"], wiki[:1])
        check("wiki: a source", record["source"] is not None, record["source"])
        record = explore(model, "Who is Zorblax Quint?", [], 0.7, rng)
        check("no source: abstain", record["direct"] and record["direct"]["kind"] == "abstain", record["direct"])

print(f"\n{results['pass']} ok, {results['fail']} failed")
sys.exit(1 if results["fail"] else 0)
