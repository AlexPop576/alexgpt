"""
Singularity chat: a small web server + page. Runs on the CPU (a laptop is enough) or on a GPU if there is one.
    python chat_server.py                 -> open http://localhost:8000
    python chat_server.py --lan           -> other devices on the same Wi-Fi can open http://<this PC's IP>:8000
    python chat_server.py --selftest      -> asks 3 questions in the console and exits (to check everything works)
Only the Python standard library + torch.
"""
import argparse
import json
import os
import re
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from alexgpt import tools                          # noqa: E402
from alexgpt import tool_router                    # noqa: E402
from alexgpt.bpe_fast import FastBPE               # noqa: E402
from alexgpt.sft_format import system_prompt       # noqa: E402
from alexgpt.supernova import Supernova, apply_rope  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--model", default=os.path.join(HERE, "singularity_v4.pt"))
parser.add_argument("--port", type=int, default=8000)
parser.add_argument("--lan", action="store_true", help="also reachable from other devices on the same network")
parser.add_argument("--selftest", action="store_true")
args = parser.parse_args()

NAME = "Singularity"
MAX_NEW_TOKENS = 400
TEMPERATURE = 0.6
TOP_P = 0.9
REPETITION_PENALTY = 1.15
tools.MEMORY_FILE = os.path.join(HERE, "memory.json")

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(max(1, os.cpu_count() or 1))
saved = torch.load(args.model, map_location="cpu", weights_only=False)
model = Supernova(**saved["config"])
weights = {name: tensor.float() for name, tensor in saved["weights"].items()}
weights.setdefault("out.weight", weights["token_table.weight"])     # tied: the export keeps only one copy
model.load_state_dict(weights)
model = model.to(device).eval()
tokenizer = FastBPE.load(os.path.join(HERE, "tokenizer.json"))
SPECIAL = tokenizer.special_ids
END = tokenizer.end_id
print(f"{NAME} loaded on {device} ({sum(p.numel() for p in model.parameters()) / 1e6:.0f}M parameters)")


# ====================================================================== the prompt
def message_ids(role, content):
    """one message -> ids.  The text is encoded WITHOUT special tokens: a user typing "<|system|>" stays plain text."""
    text_ids = tokenizer.encode(content, allow_special=False)
    if role == "system":
        return [SPECIAL["<|system|>"]] + text_ids
    if role == "user":
        return [SPECIAL["<|user|>"]] + text_ids
    if role == "tool_call":
        return [SPECIAL["<|assistant|>"], SPECIAL["<|search|>"]] + text_ids + [END]
    if role == "context":
        return [SPECIAL["<|context|>"]] + text_ids
    return [SPECIAL["<|assistant|>"]] + text_ids + [END]


def prompt_ids(messages):
    """system prompt (+ memory) + the conversation + <|assistant|>.  Old turns are dropped if it gets too long."""
    system = system_prompt(NAME)
    memory = tools.memory_get("")
    if not memory.startswith("error") and memory != "nothing remembered":
        system += "\nMemory: " + memory
    head = message_ids("system", system)
    turns = [message_ids(message["role"], message["content"]) for message in messages]
    room = model.context_size - MAX_NEW_TOKENS - len(head) - 1
    while len(turns) > 1 and sum(len(turn) for turn in turns) > room:
        turns.pop(0)                                   # forget the oldest message
    ids = head + [token for turn in turns for token in turn] + [SPECIAL["<|assistant|>"]]
    return ids[-(model.context_size - MAX_NEW_TOKENS):]


# ====================================================================== generation with a KV cache
@torch.no_grad()
def forward_with_cache(new_ids, cache, start):
    """Runs only the NEW tokens; keys and values of the earlier tokens come from the cache.
    Without the cache every new token would recompute the whole conversation (too slow on a laptop)."""
    x = model.token_table(torch.tensor([new_ids], device=device))
    length = len(new_ids)
    cos, sin = model.rope_cos[start:start + length], model.rope_sin[start:start + length]
    for index, block in enumerate(model.blocks):
        attention = block.attention
        h = block.norm1(x)
        q, k, v = attention.qkv(h).split(x.size(-1), dim=2)
        shape = (1, length, attention.heads, attention.head_size)
        q, k, v = (t.view(shape).transpose(1, 2) for t in (q, k, v))
        q, k = attention.q_norm(q), attention.k_norm(k)

        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cache[index] is not None:
            k = torch.cat([cache[index][0], k], dim=2)
            v = torch.cat([cache[index][1], v], dim=2)
        cache[index] = (k, v)
        if length == 1:
            out = F.scaled_dot_product_attention(q, k, v)
        else:                                          # new token i may see everything before start + i
            total = k.size(2)
            allowed = torch.arange(total, device=device)[None, :] <= (start + torch.arange(length, device=device))[:, None]
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed)
        out = out.transpose(1, 2).contiguous().view(1, length, -1)
        x = x + attention.out(out)
        x = x + block.mlp(block.norm2(x))
    return model.out(model.norm(x[:, -1]))[0].float()


def pick_token(logits, generated):
    for token_id in set(generated):                    # small penalty for repeating itself
        logits[token_id] = logits[token_id] / REPETITION_PENALTY if logits[token_id] > 0 else logits[token_id] * REPETITION_PENALTY
    probabilities = torch.softmax(logits / TEMPERATURE, dim=-1)
    sorted_probabilities, order = torch.sort(probabilities, descending=True)
    keep = torch.cumsum(sorted_probabilities, dim=0) - sorted_probabilities < TOP_P   # top-p: the smallest set with 90%
    sorted_probabilities[~keep] = 0
    choice = torch.multinomial(sorted_probabilities / sorted_probabilities.sum(), 1)
    return int(order[choice])


def generate(ids, allow_tools=True):
    """yields ("text", piece) while writing, or ("tool", call line) if the model asks for a tool."""
    cache = [None] * len(model.blocks)
    logits = forward_with_cache(ids, cache, 0)
    if not allow_tools:
        logits[SPECIAL["<|search|>"]] = float("-inf")
    position = len(ids)
    generated, sent = [], ""
    for _ in range(MAX_NEW_TOKENS):
        if not generated or generated[0] == SPECIAL["<|search|>"]:
            token_id = int(logits.argmax())           # the tool decision and the tool line: no randomness (exact names, numbers)
        else:
            token_id = pick_token(logits.clone(), generated)
        if token_id == END or token_id in (SPECIAL["<|user|>"], SPECIAL["<|system|>"], SPECIAL["<|assistant|>"]):
            break
        generated.append(token_id)
        if generated[0] != SPECIAL["<|search|>"]:
            text = tokenizer.decode(generated)
            if not text.endswith("�"):            # half of a multi-byte letter: wait for the rest
                yield "text", text[len(sent):]
                sent = text
        if position >= model.context_size - 1:
            break
        logits = forward_with_cache([token_id], cache, position)
        position += 1
    if generated and generated[0] == SPECIAL["<|search|>"]:
        yield "tool", tokenizer.decode(generated[1:]).strip()


def check_memory_call(call, user_text):
    """A small model sometimes 'remembers' things that were never said ("sunt trist" -> name=Tristan) or changes them
    (Alex -> Alexandru). Keep a memory_save only if the value is really in the user's message; returns the call or None."""
    name, _, argument = call.partition(":")
    if name.strip().replace("_", "") != "memorysave":
        return call
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


def reply(messages):
    """the whole answer, with up to 3 tool calls in between.  yields events for the page."""
    messages = list(messages)
    user_text = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    allow_tools = True
    routed = tool_router.route(user_text)
    if routed:                                         # a clear request: plain rules choose the tool, the model writes the answer
        for tool_call in routed:
            result = tools.run_tool(tool_call)
            yield {"type": "tool", "call": tool_call, "result": result}
            messages += [{"role": "tool_call", "content": tool_call}, {"role": "context", "content": result}]
        allow_tools = False
    for _ in range(4):
        tool_call = None
        for kind, value in generate(prompt_ids(messages), allow_tools):
            if kind == "text":
                yield {"type": "token", "text": value}
            else:
                tool_call = value
        if tool_call is None:
            return
        tool_call = check_memory_call(tool_call, user_text)
        if tool_call is None:                          # a bad memory call: answer again, without tools
            allow_tools = False
            continue
        result = tools.run_tool(tool_call)
        yield {"type": "tool", "call": tool_call, "result": result}
        messages += [{"role": "tool_call", "content": tool_call}, {"role": "context", "content": result}]


# ====================================================================== the web server
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = open(os.path.join(HERE, "chat.html"), "rb").read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/memory":
            self.send_json({"memory": tools.memory_get("")})
        else:
            self.send_error(404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        data = json.loads(self.rfile.read(length) or b"{}")
        if self.path == "/memory/clear":
            tools._write_memory({})
            self.send_json({"memory": "nothing remembered"})
            return
        if self.path != "/chat":
            self.send_error(404)
            return
        messages = [m for m in data.get("messages", []) if m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str)]
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        start, count = time.time(), 0
        try:
            for event in reply(messages):
                count += event["type"] == "token"
                self.wfile.write((json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8"))
                self.wfile.flush()
            speed = count / max(time.time() - start, 1e-6)
            self.wfile.write((json.dumps({"type": "done", "speed": round(speed, 1)}) + "\n").encode("utf-8"))
        except (BrokenPipeError, ConnectionResetError):
            pass

    def send_json(self, data):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *arguments):
        pass


if __name__ == "__main__":
    if args.selftest:
        for question in ["salut! cine ești?", "cât face 347 * 29?", "explain recursion in one sentence"]:
            start = time.time()
            print("USER:", question)
            text = ""
            for event in reply([{"role": "user", "content": question}]):
                if event["type"] == "tool":
                    print("  [tool]", event["call"], "->", event["result"][:80])
                else:
                    text += event["text"]
            print("MODEL:", text, f"   ({time.time() - start:.1f} s)\n")
        sys.exit(0)
    host = "0.0.0.0" if args.lan else "127.0.0.1"
    print(f"open http://localhost:{args.port}" + ("   (from other devices: http://<this PC's IP>:" + str(args.port) + ")" if args.lan else ""))
    ThreadingHTTPServer((host, args.port), Handler).serve_forever()
