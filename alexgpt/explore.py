"""
Explore mode (web/explore/): runs Singularity on ONE message, exactly like the chat does, and records
everything the 3D page draws.   Used by server.py  POST /api/explore  ->  one JSON (not a stream: the page
plays it back at its own speed, and the same JSON can be saved as a "recorded run" for a demo without a GPU).

    message
      -> router checks          gates:  [{gate, hit, detail}]        (SingularityChat.note_gate, in order)
      -> tools                  tools:  [{tool, call, result, unused, search}]   (search = tools.LAST_SEARCH for wiki)
      -> fixed reply?           direct: {kind: "identity"/"abstain", text, matches}
      -> the model's input      tokens: [{text, id, kind}]           kind: special / system / text / call / ctx
                                vectors: {id: 6 numbers}              (rows of the embedding table)
      -> one step per token     steps:  [{token, prob, top, before_guard, banned, ...}]
                                         the first DETAILED steps also have blocks (heads, lens, attn, mlp), window, embedding
      -> grounded text + source

The model itself is not changed: the hooks are the ones the Visualisation already uses (FlowInspector).
"""

import threading

import numpy as np

from alexgpt import identity_rag, tools
from alexgpt.chat import generate_stream

DETAILED_STEPS = 3          # the page explains the first token slowly and shows 2 more quickly: only those need the heads
MAX_TOKENS = 120
SPECIAL_KINDS = {"<|system|>": "system", "<|user|>": "text", "<|assistant|>": "answer", "<|context|>": "ctx", "<|search|>": "call"}

_lock = threading.Lock()                   # one Explore run at a time (the model object holds per-run state)
_explore_state = {"last_call": None, "last_wiki_title": None, "topic": None}   # Explore's own follow-up memory


def token_list(model, ids):
    """ids -> [{"text", "id", "kind"}]: the kind is decided by the last special token before it"""
    pieces = model.chars
    result, kind = [], "text"
    for token_id in ids:
        piece = pieces[token_id]
        if piece in SPECIAL_KINDS:
            kind = SPECIAL_KINDS[piece]
            result.append({"text": piece, "id": int(token_id), "kind": "special"})
        elif piece == "<|endoftext|>":
            result.append({"text": piece, "id": int(token_id), "kind": "special"})
        else:
            result.append({"text": piece, "id": int(token_id), "kind": kind})
    return result


def embedding_rows(model, ids, size=6):
    """the first `size` numbers of the embedding table row of every id (what the "Embeddings" board shows)"""
    torch = model.torch
    unique = sorted(set(int(i) for i in ids))
    with torch.no_grad():
        rows = model.model.token_table.weight[unique, :size].float().cpu().numpy()
    return {str(token_id): [round(float(v), 3) for v in row] for token_id, row in zip(unique, rows)}


def identity_matches(text, count=4):
    """the closest questions of identity_qa.txt with their score (for the "Identity file" board)"""
    found = identity_rag.index()
    return [[question, round(score, 3), group] for group, score, question in found.ranked(text)[:count]]


def step_record(event, detailed):
    """one generated token -> what the page needs.  event["galaxy"] = FlowInspector.report() + generate_core's extras"""
    inside = event.get("galaxy") or {}
    record = {"token": event["char"], "prob": round(event.get("prob", 0.0), 4),
              "model_top": inside.get("model_top", []),                                   # the model's own chances (after fact_guard)
              "top": [[row[0], row[3]] for row in inside.get("output", []) if row[3] > 0.0005][:8],   # what is left after temperature / top-p
              "before_guard": inside.get("before_guard"), "banned": inside.get("banned", [])}
    blocks = inside.get("blocks", [])
    if blocks:
        record["lens"] = [block["lens"][0][0] for block in blocks]
    if detailed and blocks:
        record["window"] = inside.get("tokens", [])
        record["embedding"] = inside.get("embedding", [])
        record["blocks"] = [{"heads": block["heads"], "attn": block["attn"], "mlp": block["mlp"],
                             "lens": block["lens"]} for block in blocks]
    return record


def explore(model, prompt, history=None, temperature=0.7, rng=None, max_tokens=MAX_TOKENS):
    """Runs the message and returns the record (a dict, ready for json.dumps)."""
    rng = rng or np.random.default_rng()
    history = history or []
    record = {"prompt": prompt, "model": getattr(model, "name", "?"), "gates": [], "tools": [], "direct": None,
              "rewritten": None, "tokens": [], "vectors": {}, "steps": [], "grounded": "", "source": None, "answer": ""}
    with _lock:
        # Explore has its own follow-up memory: the chat's "last tool call" and topic stay as they were
        chat_state = {key: getattr(model, key, None) for key in _explore_state}
        for key, value in _explore_state.items():
            setattr(model, key, value)
        model.tracer = record["gates"]
        input_recorded = False
        try:
            for event in generate_stream(model, prompt, max_tokens, temperature, rng, history, visualize=True):
                kind = event.get("type")
                if kind == "tool":
                    entry = {"tool": event["tool"], "call": event["call"], "result": event["result"][:700],
                             "unused": bool(event.get("unused"))}
                    if event["tool"] == "wiki":
                        entry["search"] = {key: value for key, value in tools.LAST_SEARCH.items()}
                    record["tools"].append(entry)
                    continue
                if kind != "token":
                    if kind == "source":
                        record["source"] = {"title": event["title"], "url": event["url"]}
                    continue
                if event.get("abstain") or event.get("identity"):
                    record["direct"] = {"kind": "abstain" if event.get("abstain") else "identity", "text": event["char"]}
                    if event.get("identity"):
                        record["direct"]["matches"] = identity_matches(prompt)
                    continue
                if event.get("grounded"):
                    record["grounded"] += event["char"]
                    continue
                if not input_recorded:                  # the tools are done: this is the input the model really got
                    input_recorded = True
                    used_prompt = model.rewritten_prompt if (model.tool_messages and model.rewritten_prompt) else prompt
                    record["rewritten"] = model.rewritten_prompt
                    ids = model.encode(used_prompt, [], model.tool_messages) if model.tool_messages else model.encode(used_prompt, history)
                    record["tokens"] = token_list(model, ids)
                    record["vectors"] = embedding_rows(model, ids)
                step = step_record(event, len(record["steps"]) < DETAILED_STEPS)
                record["steps"].append(step)
                if "<|" not in event["char"]:           # <|endoftext|> is a real step (the model chose to stop), not text
                    record["answer"] += event["char"]
        finally:
            model.tracer = None
            model.visualize = False
            for key in _explore_state:
                _explore_state[key] = getattr(model, key, None)
            for key, value in chat_state.items():
                setattr(model, key, value)
    record["answer"] = record["answer"].strip()
    return record
