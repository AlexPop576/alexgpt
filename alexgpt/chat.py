"""
The bridge between the trained models and the web interface.

Every model class here has the same shape (like a C++ interface):
    model.chars           list of characters it knows (its vocabulary)
    model.context_size    how many previous characters it looks at
    model.next_char_probs(context_ids) -> numpy array with one probability per character

generate_stream() only uses those three things, so it works with any model.
To add a new model: write its class, then add one line to MODEL_REGISTRY.
"""

import os
import re
import numpy as np


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class SparkBigram:
    """Stage 1: counts which character follows which. Sees 1 character."""

    def __init__(self, path):
        data = np.load(path)
        self.probs = data["probs"]              # (vocab_size, vocab_size)
        self.chars = list(data["chars"])
        self.context_size = 1

    def next_char_probs(self, context_ids):
        last_id = context_ids[-1]
        return self.probs[last_id]

    def num_params(self):
        return self.probs.size

    def layer_sizes(self):
        return {}


class NebulaMLP:
    """Stage 3: embedding + one hidden layer. Sees the last 8 characters."""

    def __init__(self, path):
        data = np.load(path)
        self.table = data["table"]              # (vocab_size, embed_dim)
        self.W1 = data["W1"]
        self.b1 = data["b1"]
        self.W2 = data["W2"]
        self.b2 = data["b2"]
        self.chars = list(data["chars"])
        self.context_size = int(data["context_size"])
        self.last_input = None                  # embedding vectors glued together, kept for the interface
        self.last_hidden = None                 # neuron activations, kept for the interface

    def next_char_probs(self, context_ids):
        vectors = self.table[context_ids]                       # (context_size, embed_dim)
        x = vectors.reshape(1, -1)                              # (1, context_size * embed_dim)
        pre_activation = x @ self.W1 + self.b1                  # (1, hidden_size)
        hidden = np.tanh(pre_activation)                        # (1, hidden_size)
        logits = hidden @ self.W2 + self.b2                     # (1, vocab_size)
        self.last_input = x[0]
        self.last_pre = pre_activation[0]
        self.last_hidden = hidden[0]
        self.last_logits = logits[0]
        logits = logits[0] - logits[0].max()
        probs = np.exp(logits)
        return probs / probs.sum()

    def strongest_connections(self, output_ids, count_in=120, count_out=12):
        """
        Which weights carry the most signal right now (for the "Weights" view).
        A connection's signal = the value going in * the weight on the wire.
        Returns two lists of [from, to, signal], signals scaled to [-1, 1].
        """
        # embedding -> hidden: signal[i][j] = x[i] * W1[i][j]
        signal_in = self.last_input[:, None] * self.W1          # (128, 200)
        flat = np.abs(signal_in).ravel()
        best = np.argpartition(flat, -count_in)[-count_in:]     # indices of the biggest |signal|
        rows, cols = np.unravel_index(best, signal_in.shape)
        biggest = flat[best].max()
        edges_in = []
        for index in range(len(best)):
            value = signal_in[rows[index], cols[index]] / biggest
            edges_in.append([int(rows[index]), int(cols[index]), round(float(value), 3)])

        # hidden -> each of the top output characters: signal[j] = h[j] * W2[j][char]
        edges_out = []
        for position in range(len(output_ids)):
            char_id = output_ids[position]
            signal_out = self.last_hidden * self.W2[:, char_id]  # (200,)
            best = np.argsort(np.abs(signal_out))[-count_out:]
            for neuron in best:
                edges_out.append([int(neuron), position, float(signal_out[neuron])])
        biggest = 1e-9
        for edge in edges_out:
            biggest = max(biggest, abs(edge[2]))
        for edge in edges_out:
            edge[2] = round(edge[2] / biggest, 3)

        return edges_in, edges_out

    def layer_sizes(self):
        return {"embed_dim": int(self.table.shape[1]), "hidden": int(self.W1.shape[1])}

    def num_params(self):
        total = 0
        for array in (self.table, self.W1, self.b1, self.W2, self.b2):
            total += array.size
        return total

def attention_forward(x, Wq, Wk, Wv):
    # x: (T, C)   T = positions in the context, C = embedding size
    # Wq, Wk, Wv: (C, H)   H = head size
    # 1. q, k, v = x @ each matrix                      -> (T, H) each
    q = x @ Wq
    k = x @ Wk
    v = x @ Wv
    # 2. scores = q @ k.T                               -> (T, T)  scores[t][s] = q[t] · k[s]
    scores = q @ k.T
    # 3. scores = scores / sqrt(H)                      (keeps numbers small so softmax isn't too sharp)
    scores = scores/np.sqrt(Wk.shape[1])
    # 4. for every t, s with s > t: scores[t][s] = -inf (the causal mask, two for loops are fine)
    for t in range(scores.shape[0]):
        for s in range(scores.shape[1]):
            if s > t:
                scores[t][s] = -np.inf
    # 5. weights = softmax on each row                  (subtract the row max first, like in chat.py)
    T = scores.shape[0]
    weights = np.zeros(scores.shape)
    for t in range(T):
        row = scores[t]                      # the scores of position t
        biggest = row.max()                  # A: the biggest score in this row
        exps = np.exp(row - biggest)         # A + B: subtract it from every number, then exp of every number
        weights[t] = exps / exps.sum()       # C: divide every number by the sum of the row
    # 6. out = weights @ v                              -> (T, H)
    out = weights @ v
    return out, weights

def embed_with_positions(ids, token_table, position_table):
    # ids: list of character ids, one per position     (length T)
    # token_table: (vocab_size, C)    one row per character
    # position_table: (max_T, C)      one row per position
    T = len(ids)
    C = token_table.shape[1]
    x = np.zeros((T, C))
    for i in range(T):
        x[i] = token_table[ids[i]] + position_table[i]
    return x

class PulsarTransformer:
    """Stage 4: a small Transformer. Attention + MLP blocks, sees the last 32 characters."""

    def __init__(self, path):
        data = np.load(path)
        self.token_table = data["token_table"]
        self.position_table = data["position_table"]
        self.W_out = data["W_out"]
        self.b_out = data["b_out"]
        self.chars = list(data["chars"])
        self.context_size = self.position_table.shape[0]
        self.blocks = []
        for b in range(int(data["n_blocks"])):
            block = {}
            for name in ("Wq", "Wk", "Wv", "W1", "b1", "W2", "b2"):
                block[name] = data[f"block{b}_{name}"]
            self.blocks.append(block)
        self.last_attention = None              # kept for the interface

    def next_char_probs(self, context_ids):
        x = embed_with_positions(context_ids, self.token_table, self.position_table)
        # for the interface: what happens to the NEWEST letter's vector in every block
        self.trace = {"embedding": x[-1], "blocks": []}
        for block in self.blocks:
            attended, weights = attention_forward(x, block["Wq"], block["Wk"], block["Wv"])
            x = x + attended                                     # residual
            hidden = np.tanh(x @ block["W1"] + block["b1"])
            thought = hidden @ block["W2"] + block["b2"]
            x = x + thought                                      # residual
            self.trace["blocks"].append({
                "attention": weights[-1],       # where the newest letter looks
                "attn_out": attended[-1],       # what attention added to its vector
                "hidden": hidden[-1],           # the MLP neurons
                "mlp_out": thought[-1],         # what the MLP added to its vector
            })
        self.last_attention = weights[-1]       # last block: where the newest letter looks
        logits = x[-1] @ self.W_out + self.b_out
        self.last_logits = logits
        logits = logits - logits.max()
        probs = np.exp(logits)
        return probs / probs.sum()

    def num_params(self):
        total = self.token_table.size + self.position_table.size + self.W_out.size + self.b_out.size
        for block in self.blocks:
            for name in block:
                total += block[name].size
        return total

    def layer_sizes(self):
        return {"transformer": True, "blocks": len(self.blocks),
                "embed_dim": int(self.token_table.shape[1]), "hidden": int(self.blocks[0]["W1"].shape[1])}

# name -> (class, checkpoint file, short description)
def training_running(project_root):
    """True if a training writes its log right now (a runs/*/log.txt changed in the last 3 minutes)."""
    import glob
    import time
    for path in glob.glob(os.path.join(project_root, "runs", "*", "log.txt")):
        if time.time() - os.path.getmtime(path) < 180:
            return True
    return False


def pick_device(torch, needed_gb, project_root=None):
    """The GPU only if no training is running and it has room left, else the CPU.
    (Sharing the GPU with a training makes Windows borrow normal RAM and BOTH crawl.)"""
    if project_root and training_running(project_root):
        return "cpu"
    if torch.cuda.is_available():
        try:
            if torch.cuda.mem_get_info()[0] > needed_gb * 1e9:
                return "cuda"
        except Exception:
            pass
    return "cpu"


class FlowInspector:
    """For the Visualisation: hooks (small functions PyTorch calls after a layer runs) that copy what
    happens to the NEWEST token in every block. The model itself is not changed.
        x_in  -> attention adds something -> MLP adds something -> x_out
    report() turns that into: where every head looks, how much attention / MLP changed the vector,
    which tokens each of them pushes toward, and the "logit lens" guess after every block."""

    SHOWN = 12          # how many of the last tokens the page shows

    def __init__(self, owner, model):
        self.owner = owner              # GalaxyChat / SupernovaBase: has .visualize, .chars, .device
        self.model = model
        self.data = {}
        for index, block in enumerate(model.blocks):
            block.register_forward_hook(self.keep_block(index))
            block.attention.register_forward_hook(self.keep_part(index, "attn"))
            block.mlp.register_forward_hook(self.keep_part(index, "mlp"))

    def keep_block(self, index):
        def hook(module, inputs, output):
            if self.owner.visualize:
                self.data[("in", index)] = inputs[0][0, -1].detach().float()
                self.data[("out", index)] = output[0, -1].detach().float()
        return hook

    def keep_part(self, index, name):
        def hook(module, inputs, output):
            if self.owner.visualize:
                self.data[(name, index)] = output[0, -1].detach().float()
        return hook

    def report(self, context_ids, head_rows):
        """head_rows(index) -> (heads, length) attention of the newest token in that block."""
        import torch
        model = self.model
        pieces = self.owner.chars
        shown = min(self.SHOWN, len(context_ids))
        count = len(model.blocks)
        with torch.no_grad():
            weight = model.out.weight
            norm_scale = model.norm.weight.float()
            # the logit lens: x after each block (and before the first one) -> final norm -> scores -> softmax
            streams = [self.data[("in", 0)]] + [self.data[("out", index)] for index in range(count)]
            stream = torch.stack(streams).to(weight.dtype)
            lens_probs = torch.softmax(model.out(model.norm(stream)).float(), dim=-1)
            lens_values, lens_ids = lens_probs.topk(3, dim=-1)
            # what attention and the MLP ADD, compared with every token vector: which tokens they push toward
            parts = torch.stack([self.data[(name, index)] for index in range(count) for name in ("attn", "mlp")])
            push_ids = ((parts * norm_scale).to(weight.dtype) @ weight.T).float().topk(3, dim=-1).indices

        def guesses(row):
            return [[pieces[int(lens_ids[row, k])], round(float(lens_values[row, k]), 3)] for k in range(3)]

        blocks = []
        for index in range(count):
            size_in = float(self.data[("in", index)].norm()) + 1e-6
            rows = head_rows(index)[:, -shown:]
            blocks.append({
                "heads": [[round(float(value), 2) for value in row] for row in rows],
                "lens": guesses(index + 1),
                "attn": round(float(self.data[("attn", index)].norm()) / size_in, 3),
                "mlp": round(float(self.data[("mlp", index)].norm()) / size_in, 3),
                "attn_push": [pieces[int(token_id)] for token_id in push_ids[2 * index]],
                "mlp_push": [pieces[int(token_id)] for token_id in push_ids[2 * index + 1]],
            })
        return {"tokens": [pieces[token_id] for token_id in list(context_ids)[-shown:]],
                "start": guesses(0), "blocks": blocks,
                "embedding": [round(float(value), 3) for value in self.data[("in", 0)][:8]]}   # 8 of the newest token's numbers


class GalaxyChat:
    """Stage 5: PyTorch Transformer on BPE tokens. Talks in the conversation format:
           **Child**: <your message>
           **Mom**: <its reply>   ... and stops when it starts the next "**Speaker**:" line.
    Uses checkpoints/galaxy.pt if it exists (trained), otherwise random weights (untrained)."""

    def __init__(self, tokenizer_path):
        import torch
        from alexgpt.bpe import BPETokenizer
        from alexgpt.galaxy import Galaxy
        self.torch = torch
        self.tokenizer = BPETokenizer.load(tokenizer_path)
        self.chars = self.tokenizer.pieces           # "chars" = tokens here, each one a piece of text
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(tokenizer_path)))
        self.device = pick_device(torch, 1.0, self.project_root)   # while a model trains: the CPU

        weights_path = os.path.join(os.path.dirname(tokenizer_path), "galaxy.pt")
        if os.path.exists(weights_path):
            saved = torch.load(weights_path, map_location="cpu")
            config = dict(saved["config"])
            config["dropout"] = 0.0
            self.model = Galaxy(**config)
            self.model.load_state_dict(saved["weights"])
            self.trained = True
            print(f"           weights from checkpoints/galaxy.pt (step {saved.get('step')}, val loss {saved.get('val_loss', 0):.3f})")
        else:
            torch.manual_seed(0)                     # the same random Galaxy every time
            self.model = Galaxy(vocab_size=self.tokenizer.vocab_size, dropout=0.0)
            self.trained = False
            self.max_tokens = 120                    # untrained: never stops on its own, keep it short
            print("           no checkpoints/galaxy.pt yet: random (untrained) weights")
        self.model.to(self.device)
        self.model.eval()
        self.visualize = False                       # set per request: only collect the inside when the panel is open
        self.inside = {}
        self.add_inspection_hooks()
        self.context_size = self.model.context_size
        # tokens that would end the reply: a line break (the next speaker starts) or <|endoftext|>
        self.reply_end_ids = [token_id for token_id, piece in enumerate(self.chars) if "\n" in piece]
        self.reply_end_ids.append(self.tokenizer.end_id)
        self.stop_text = "\n**"                      # the next speaker starts -> the reply is over

    def add_inspection_hooks(self):
        self.inspector = FlowInspector(self, self.model)

    def inside_report(self, context_ids):
        def head_rows(index):
            return self.model.blocks[index].attention.last_weights[0, :, -1, :].float().cpu()
        return self.inspector.report(context_ids, head_rows)

    def system_prompt(self):
        """The text in prompts/galaxy.txt, read again for every message (edit it, no restart needed)."""
        path = os.path.join(self.project_root, "prompts", "galaxy.txt")
        if not os.path.exists(path):
            return ""
        with open(path, encoding="utf-8") as file:
            return file.read().strip()

    def encode(self, prompt, history):
        # the whole chat, written in the format Galaxy saw in training
        text = ""
        system = self.system_prompt()
        if system:
            text += system + "\n"
        for turn in history:
            speaker = "**Child**:" if turn["role"] == "user" else "**Mom**:"
            text += speaker + " " + turn["text"].strip() + "\n"
        text += "**Child**: " + prompt.strip() + "\n**Mom**:"
        return self.tokenizer.encode(self.clean(text))

    def clean(self, text):
        """Drops the characters the tokenizer never saw (e.g. "ă", "–", "“"): it can't encode them."""
        known = ""
        for char in text:
            if char in self.tokenizer.char_id:
                known += char
        return known

    def next_char_probs(self, context_ids):
        torch = self.torch
        with torch.no_grad():
            x = torch.tensor([list(context_ids)], dtype=torch.long, device=self.device)
            logits, _ = self.model(x)
            probs = torch.softmax(logits[0, -1].float(), dim=0)
            self.last_logits = logits[0, -1].float().cpu().numpy()
            self.last_inside = self.inside_report(context_ids) if self.visualize else None
        return probs.cpu().numpy().astype(np.float64)

    def num_params(self):
        return self.model.num_params()

    def layer_sizes(self):
        return {"chat": True, "trained": self.trained, "blocks": len(self.model.blocks)}


class SupernovaBase:
    """Supernova while it trains: the BASE model (no chat fine-tuning yet), so it CONTINUES your text.
    Reads runs/supernova/latest.pt and loads it again by itself when training saves a newer one
    (every ~30 minutes), so every message uses the newest weights."""

    def __init__(self, checkpoint_path):
        import torch
        from alexgpt.bpe_fast import FastBPE
        from alexgpt.supernova import Supernova
        self.torch = torch
        self.checkpoint_path = checkpoint_path
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(checkpoint_path))))
        self.tokenizer = FastBPE.load(os.path.join(root, "checkpoints", "supernova_tokenizer.json"))
        self.chars = self.tokenizer.pieces
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.model = Supernova(**state["config"])
        self.context_size = self.model.context_size
        # the GPU is busy training: use it only if there is room left, otherwise the CPU (slower)
        self.device = pick_device(torch, 1.5, root)
        self.loaded_time = 0
        self.step = 0
        self.reload_if_newer(state)
        self.reply_end_ids = [self.tokenizer.end_id]
        self.stop_text = None
        self.max_tokens = 120
        self.last_inside = None
        self.visualize = False
        self.inside = {}
        self.add_inspection_hooks()
        print(f"           step {self.step}, running on {self.device}")

    def reload_if_newer(self, state=None):
        modified = os.path.getmtime(self.checkpoint_path)
        if modified == self.loaded_time:
            return
        if state is None:
            state = self.torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
            print(f"  Supernova: loaded newer weights (step {state['step']})")
        self.model.load_state_dict(state["weights"] if "weights" in state else state["model"])   # final.pt / latest.pt
        self.model.to(self.device)
        self.model.eval()
        self.step = state.get("step", 0)
        self.loaded_time = modified

    def add_inspection_hooks(self):
        """Supernova's attention runs in one fast GPU kernel that doesn't return the weights, so for
        the Visualisation we recompute the newest token's attention row from the attention's input."""
        torch = self.torch
        from alexgpt.supernova import apply_rope

        def keep_attention(index):
            def hook(module, inputs, output):
                if not self.visualize:
                    return
                x, cos, sin = inputs[:3]      # (the 4th input, the SFT mask, is None in chat)
                batch, length, size = x.shape
                q, k, _ = module.qkv(x).split(size, dim=2)
                q = q.view(batch, length, module.heads, module.head_size).transpose(1, 2)
                k = k.view(batch, length, module.heads, module.head_size).transpose(1, 2)
                q, k = module.q_norm(q), module.k_norm(k)
                q, k = apply_rope(q, cos[:length], sin[:length]), apply_rope(k, cos[:length], sin[:length])
                scores = (q[:, :, -1:, :] @ k.transpose(-2, -1)) / module.head_size ** 0.5   # newest row only
                self.inside[f"heads{index}"] = torch.softmax(scores.float(), dim=-1)[0, :, 0, :].cpu()
            return hook

        for index, block in enumerate(self.model.blocks):
            block.attention.register_forward_hook(keep_attention(index))
        self.inspector = FlowInspector(self, self.model)

    def inside_report(self, context_ids):
        return self.inspector.report(context_ids, lambda index: self.inside[f"heads{index}"])

    def encode(self, prompt, history):
        self.reload_if_newer()
        return self.tokenizer.encode(prompt)          # a base model: just continue the text

    def next_char_probs(self, context_ids):
        torch = self.torch
        with torch.no_grad():
            x = torch.tensor([list(context_ids)], dtype=torch.long, device=self.device)
            logits, _ = self.model(x)
            probs = torch.softmax(logits[0, -1].float(), dim=0)
        self.last_logits = logits[0, -1].float().cpu().numpy()
        self.last_inside = self.inside_report(context_ids) if self.visualize else None
        return probs.cpu().numpy().astype(np.float64)

    def num_params(self):
        return self.model.num_params()

    def layer_sizes(self):
        return {"chat": True, "trained": True, "blocks": len(self.model.blocks)}


def split_into_tokens(model, text):
    """How `model` cuts `text` into tokens (for the "Show tokens" view).
    Returns [[piece, id], ...]. Characters the model can't read come back with id -1."""
    if hasattr(model, "tokenizer"):                    # Galaxy, Supernova: BPE
        dropped = []
        if hasattr(model, "clean"):                    # Galaxy's tokenizer only knows some characters
            dropped = [char for char in text if char not in model.tokenizer.char_id]
            text = model.clean(text)
        ids = model.tokenizer.encode(text)
        pieces = [[model.chars[token_id], int(token_id)] for token_id in ids]
        for char in sorted(set(dropped)):
            pieces.append([char, -1])
        return pieces
    stoi = {char: index for index, char in enumerate(model.chars)}   # Spark, Nebula, Pulsar: one character = one token
    return [[char, stoi.get(char, -1)] for char in text]


class SupernovaChat(SupernovaBase):
    """Supernova after SFT: answers in the chat format (alexgpt/sft_format.py), remembers the conversation,
    and stops by itself with <|endoftext|>. Reads runs/sft/final.pt (or the 15-minute pilot)."""

    name = "Supernova"          # goes into the system prompt

    def __init__(self, checkpoint_path):
        super().__init__(checkpoint_path)
        self.stop_text = "<|user|>"
        self.max_tokens = 400
        self.reply_end_ids = [self.tokenizer.end_id, self.tokenizer.special_ids["<|user|>"]]
        self.sampling = ChatSampling()     # how generate_stream picks tokens (see pick_distribution)
        self.history_turns = 6             # the model sees only the last 3 exchanges (6 messages)

    def visible_history(self, history):
        """The part of the chat the model sees: the last 3 exchanges, with the replies CLEANED
        (repeated sentences and a cut last sentence removed). Why: one looping reply left in the chat
        made the next replies copy it, even after you changed the subject."""
        turns = history[-self.history_turns:]
        while turns and turns[0]["role"] != "user":         # SFT conversations always start with the user
            turns = turns[1:]
        visible = []
        for turn in turns:
            role = "user" if turn["role"] == "user" else "assistant"
            text = turn["text"].strip()
            if role == "assistant":
                text = clean_reply(text)
            visible.append({"role": role, "content": no_special_tokens(text)})
        return visible

    def encode(self, prompt, history):
        from alexgpt.sft_format import chat_prompt, system_prompt
        self.reload_if_newer()
        messages = [{"role": "system", "content": system_prompt(self.name)}]   # who it is, every time
        messages += self.visible_history(history)
        messages.append({"role": "user", "content": no_special_tokens(prompt.strip())})
        ids = self.tokenizer.encode(chat_prompt(messages))
        return ids[-(self.context_size - self.max_tokens):]       # very long chats: keep the end


class SingularityChat(SupernovaChat):
    """Singularity after SFT (runs/singularity_sft4/final.pt): like SupernovaChat, plus TOOLS.
    Before the answer, run_tools() decides (on a clean prompt: only the last message) if a tool is needed, runs it
    (alexgpt/tools.py) and puts <|search|>call + <|context|>result in front of the answer."""

    name = "Singularity"
    uses_tools = True

    def __init__(self, checkpoint_path):
        super().__init__(checkpoint_path)
        from alexgpt import tools
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(checkpoint_path))))
        tools.MEMORY_FILE = os.path.join(root, "memory.json")
        self.gate_log = os.path.join(root, "runs", "history_gate.jsonl")   # every history on/off decision, to review
        self.use_history = True            # set by run_tools() each turn (strategy 3: only for follow-ups)
        self.topic = None                  # topic memory: the last article title or name the user talked about
        self.history_turns = 2             # a 126M model gets confused by longer chats: it sees only the last exchange
        self.search_id = self.tokenizer.special_ids["<|search|>"]
        self.block_ids = [self.search_id, self.tokenizer.special_ids["<|context|>"]]    # the answer itself never starts a tool call
        self.tracer = None                 # a list while the Explore page runs (alexgpt/explore.py): every router check goes in it

    def note_gate(self, gate, hit, detail=""):
        """Explore only: notes one router check ("followup", "rules", "identity", "question", "search").
        In the normal chat self.tracer is None and this does nothing."""
        if self.tracer is not None:
            self.tracer.append({"gate": gate, "hit": bool(hit), "detail": detail})

    def messages_ids(self, user_text, history, extra):
        from alexgpt import tools
        from alexgpt.sft_format import chat_prompt, system_prompt
        system = system_prompt(self.name)
        memory = tools.memory_get("")
        if not memory.startswith("error") and memory != "nothing remembered":
            system += "\nMemory: " + memory
        messages = [{"role": "system", "content": system}]
        if not extra and self.use_history:              # the earlier messages only for a follow-up ("why?", "and in Cluj?")
            messages += self.visible_history(history)
        messages.append({"role": "user", "content": no_special_tokens(user_text)})
        messages += [{"role": m["role"], "content": no_special_tokens(m["content"])} for m in extra or []]
        return self.tokenizer.encode(chat_prompt(messages))[-(self.context_size - self.max_tokens):]

    def encode(self, prompt, history, extra=None):
        self.reload_if_newer()
        return self.messages_ids(prompt.strip(), history, extra)

    def greedy(self, ids):
        probs = self.next_char_probs(np.array(ids[-self.context_size:]))
        return int(np.argmax(probs))

    def run_tools(self, prompt, history):
        """A generator: yields events for the page ("tool_start" before a tool runs, "tool" with its result).
        Afterwards self.tool_messages = the tool messages for the prompt (None = answer without tools)
        and self.wiki_details = the longer Wikipedia text + link of a good wiki call (or None)."""
        from alexgpt import tool_router, tools
        self.reload_if_newer()
        self.tool_messages, self.wiki_details, self.rewritten_prompt = None, None, None
        self.abstain = None                 # a fixed "no source" reply: set when a factual question found no evidence
        self.identity_reply = None          # an answer from sft_data/identity_qa.txt ("who made you?")
        self.found_evidence = False         # True once a wiki call returned a usable result
        previous = ""
        for message in reversed(history or []):
            if message.get("role") == "user":
                previous = message.get("text") or message.get("content") or ""
                break
        if not history:                                 # a new chat: nothing from the last chat may leak in
            self.last_call, self.last_wiki_title, self.topic = None, None, None
        follow_up, reason = tool_router.is_follow_up(prompt)
        self.use_history = follow_up and bool(history)
        self.log_history_gate(prompt, reason)
        visualize_after, self.visualize = self.visualize, False       # the tool-call tokens are not drawn; the answer is
        try:
            yield from self.run_tools_inner(prompt, previous, tool_router, tools)
        finally:
            self.visualize = visualize_after
        # the topic memory: the name the USER wrote wins over the article title, because the search can return the
        # wrong article ("what songs does Alesso have" -> "Becky Hill").  The title is kept only when the user
        # named nothing ("tell me about the danube" -> "Danube").
        entity = tool_router.main_entity(self.rewritten_prompt or prompt)
        if entity:
            self.topic = entity

    def log_history_gate(self, prompt, reason):
        """one line per message in runs/history_gate.jsonl: did the model see the history, and why"""
        import json
        import time
        line = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "message": prompt.strip()[:200],
                "history": self.use_history, "reason": reason, "topic_before": self.topic}
        try:
            os.makedirs(os.path.dirname(self.gate_log), exist_ok=True)
            with open(self.gate_log, "a", encoding="utf-8") as file:
                file.write(json.dumps(line, ensure_ascii=False) + "\n")
        except OSError:
            pass                                        # the log must never break the chat

    def run_tools_inner(self, prompt, previous, tool_router, tools):
        text = prompt.strip()
        extra = []
        aliases = {"memorysave": "memory_save", "memoryget": "memory_get", "todayinhistory": "today_in_history"}

        def run(call, must_be_relevant=False, from_model=False):
            """runs one call and yields the page events.  A call the model wrote that fails is dropped quietly
            (it was the model's mistake, not something the user asked for)."""
            name = call.partition(":")[0].strip().lower()
            name = aliases.get(name, name)
            if not from_model:
                yield {"type": "tool_start", "tool": name, "call": call}
            result = tools.run_tool(call)
            ok = not result.startswith("error")
            self.wikipedia_busy = "busy" in result[:60]
            if from_model and not ok:
                return
            if from_model:
                yield {"type": "tool_start", "tool": name, "call": call}
            event = {"type": "tool", "tool": name, "call": call, "result": result}
            if name == "wiki" and ok:
                details = dict(tools.LAST_WIKI)
                if must_be_relevant and not details.get("relevant", True):
                    ok = False
                elif details.get("url"):
                    self.found_evidence = True
                    self.wiki_details = details
                    self.last_wiki_title = details["title"]
                    self.topic = details["title"]
                    event["source"] = {"title": details["title"], "url": details["url"], "site": "Wikipedia"}
            if must_be_relevant and not ok:
                event["unused"] = True                  # nothing useful found: the model answers on its own
            else:
                if ok:
                    self.last_call = call
                shown_call = re.sub(r"^wiki:\s*(?:ro|en):\s*", "wiki: ", call)   # the model was trained on "wiki: X"
                extra.extend([{"role": "tool_call", "content": shown_call}, {"role": "context", "content": result}])
            yield event

        last_call = getattr(self, "last_call", None)
        follow_up = tool_router.route_followup(text, last_call)
        the_one = tool_router.the_one_in(text, previous)
        if the_one:
            self.note_gate("followup", True, "“the one in …”: the earlier question with the new place")
        elif follow_up:
            self.note_gate("followup", True, f"repeats the last tool ({last_call}) → {follow_up}")
        else:
            self.note_gate("followup", False, "not a follow-up of an earlier tool call" if last_call else "there is no earlier tool call to repeat")
        if the_one:                                     # "what is the Wesselenyi castle?" + "the one in Jibou"
            self.rewritten_prompt = the_one             # -> "what is the Wesselenyi castle in Jibou"
            call = tool_router.wiki_call(the_one)
            yield from run(call)
            self.tool_messages = extra or None
            self.abstain_if_no_evidence([call], the_one, tool_router)
            return
        if follow_up:                                   # "but in Cluj?" after a weather question
            old_place = last_call.partition(":")[2].strip()
            new_place = follow_up.partition(":")[2].strip()
            if follow_up.startswith("wiki") and previous:
                # "what countries are in Europe?" + "and in Africa?" -> "what countries are in Africa?"
                old_place = tool_router.main_entity(previous) or ""
                new_place = re.sub(r"^wiki:\s*(?:ro|en):\s*", "", follow_up)
                if old_place:
                    question = re.sub(re.escape(old_place), new_place, previous, count=1)
                    follow_up = tool_router.wiki_call(question)
            if previous and old_place and old_place.lower() in previous.lower():
                # the model reads the whole question: "what is the weather in spain?" -> "what is the weather in cluj?"
                self.rewritten_prompt = re.sub(re.escape(old_place), new_place, previous, count=1, flags=re.I)
            yield from run(follow_up)
            self.tool_messages = extra or None
            self.abstain_if_no_evidence([follow_up], previous or text, tool_router)
            return
        routed = tool_router.route(text)
        self.note_gate("rules", routed, " + ".join(routed) if routed else "no fixed rule matches (math, units, time, weather, dice, memory…)")
        if routed:                                    # a clear request ("2+6", "tell me about X", "roll 3 dice"): a plain rule,
            for call in routed:                         # the model only writes the answer
                yield from run(call)
            self.tool_messages = extra or None
            self.abstain_if_no_evidence(routed, text, tool_router)
            return
        if not self.use_history:                        # identity self-RAG: "who made you?" -> Alex's own answer
            from alexgpt import identity_rag
            answer, group = identity_rag.reply(text, self.name)
            remembered = tools.memory_get("")
            knows_user = not remembered.startswith("error") and remembered != "nothing remembered"
            if answer and not (group.startswith("unknown-you") and knows_user):   # memory may know the answer
                self.note_gate("identity", True, f"close to the questions of “{group}” in identity_qa.txt")
                self.identity_reply = answer
                self.tool_messages = None
                return
            self.note_gate("identity", False, "not a question about the model")
        else:
            self.note_gate("identity", False, "skipped: a follow-up keeps the chat history")
        if not self.topic and tool_router.PRONOUNS.search(text) and not tool_router.named_entities(text):
            if tool_router.route_question(text, previous)[0]:
                # "what songs does he have?" as the first message: nobody to search for -> ask, don't search
                self.note_gate("question", True, "a question about “he / she / it”, but nobody was named → ask who")
                self.abstain = tool_router.who_do_you_mean(text)
                self.tool_messages = None
                return
        question = tool_router.with_topic(text, self.topic)
        call, explicit = tool_router.route_question(question, previous)
        if call and self.use_history and self.topic and not tool_router.named_entities(question):
            call += " " + self.topic                    # "but how old?" after Alesso -> search "but how old Alesso"
        self.note_gate("question", call, call if call else "not a who / what / when question")
        if call:                                      # a question that needs facts, or "search ...": Wikipedia first
            if question != text:
                self.rewritten_prompt = question        # "what songs does he have" -> "what songs does Alesso have"
            yield from run(call, must_be_relevant=not explicit)
            self.tool_messages = extra or None
            self.abstain_if_no_evidence([call], text, tool_router)
            return
        for attempt in range(3):
            ids = self.messages_ids(text, [], extra)
            if self.greedy(ids) != self.search_id:
                if attempt == 0:
                    self.note_gate("search", False, "the model's first token is not <|search|>: it answers on its own")
                break
            ids = ids + [self.search_id]
            call_ids = []
            for _ in range(80):
                next_id = self.greedy(ids)
                if next_id == self.tokenizer.end_id:
                    break
                ids.append(next_id)
                call_ids.append(next_id)
            call = tool_router.check_memory_call(self.tokenizer.decode(call_ids).strip(), text)
            self.note_gate("search", call is not None, f"the model wrote <|search|>{call}" if call else "the model wrote a wrong tool call: dropped")
            if call is None:
                break                                   # a wrong tool call: answer without it
            count_before = len(extra)
            yield from run(call, from_model=True)
            if len(extra) == count_before:
                break                                   # it failed: answer without it
        self.tool_messages = extra or None

    def abstain_if_no_evidence(self, calls, user_text, tool_router):
        """Strategy 1 (docs/fable_plan.md): only for wiki calls made by the code's own rules.
        If none of them found a usable article, the answer is a fixed "no source" reply, not a guess.
        The model's own <|search|> calls never get here, so chit-chat is not affected."""
        if not getattr(self, "abstain_enabled", True):  # eval_alexgpt100.py --no-abstain: the old behaviour
            return
        if len(user_text.split()) > 40:                 # the user pasted a text: the model can answer from it
            return
        is_wiki = all(call.partition(":")[0].strip().lower() == "wiki" for call in calls)
        if is_wiki and not self.found_evidence:
            busy = getattr(self, "wikipedia_busy", False)  # 429: "busy" is the truth, "no source" would be a lie
            self.abstain = tool_router.no_source_reply(user_text, busy)
            self.tool_messages = None


def no_special_tokens(text):
    """"<|user|>" typed in a message stays text: "<|" -> "< |", so it can't become a special token."""
    return text.replace("<|", "< |")


def clean_reply(text):
    """A reply as it goes back into the chat history:
    - a sentence or list line that already appeared is removed (the start of a loop)
    - an unfinished last sentence (the reply was cut by the length limit) is removed
    "Pui de vită. Pui de porc. Pui de vită. Pui de" -> "Pui de vită. Pui de porc." """
    import re
    pieces = re.split(r"(?<=[.!?])\s+|\n", text)
    kept, seen = [], set()
    for piece in pieces:
        key = re.sub(r"[^\w]+", " ", piece.lower()).strip()
        if key and key in seen and len(key) > 3:
            continue
        seen.add(key)
        kept.append(piece)
    last = kept[-1].strip() if kept else ""
    is_list_line = re.match(r"([-*•]|\d+[.)]|#)", last) is not None
    if len(kept) > 1 and last and not is_list_line and not re.search(r"[.!?:)\]\"'`*…]$", last):
        kept.pop()                                          # a sentence the length limit cut in half
    result = ""
    for piece in kept:                                      # put the pieces back (lines stay lines)
        if not result:
            result = piece
        elif re.match(r"\s*([-*•]|\d+[.)]|#)", piece) or result.endswith(":") or not piece.strip():
            result += "\n" + piece
        else:
            result += " " + piece
    return result.strip()


class ChatSampling:
    """The settings for choosing the next token in a chat (Supernova, Singularity)."""
    top_k = 30                  # at most the 30 best tokens ...
    top_p = 0.85                # ... and only those that add up to 85%
    presence_penalty = 0.2      # a word already in this reply: its score - 0.2
    frequency_penalty = 0.2     # ... and - 0.2 more for EVERY time it appeared
    penalty_window = 160        # (only the last 160 tokens of this reply count)
    no_repeat_ngram = 4         # this reply never repeats the same 4 tokens in a row (if they hold a word)
    no_copy_ngram = 6           # and never copies 6 tokens in a row from its earlier replies
    min_p = 0.0                 # 0 = off. e.g. 0.08: drop tokens with less than 8% of the best token's chance
                                # (to be compared blind with top-p on the same prompts before it becomes the default)
    # (tested on Supernova: 3 tokens was too strict, it broke words to get around it: "I don'ts", "masline";
    #  stronger penalties did the same: "beverage" -> "bementare")


def free_token_mask(model):
    """True for tokens that may repeat without a penalty: spaces, punctuation, newlines, digits, markdown,
    and short pieces of 1-2 letters ("ă", "ul", "le": endings of Romanian words).
    (a list needs "\n- " and ", " many times; "Pui de vită, Pui de porc" is about the WORDS.
     Without the 1-2 letter rule, "ă" got penalized and the model started writing "adauga" for "adaugă".)"""
    if getattr(model, "_free_tokens", None) is None:
        import re
        free = []
        for piece in model.chars:
            letters = re.sub(r"[\W\d_]", "", piece)
            free.append(len(letters) <= 2)
        model._free_tokens = np.array(free)
    return model._free_tokens


def blocked_by_ngrams(sequence, size, candidates_from=None, free=None):
    """Tokens that would complete a `size`-token sequence already seen.
    sequence = the tokens so far; the last size-1 of them are the start of the n-gram.
    Example (size 3): reply ... "a" "b" "c" ... "a" "b"  -> "c" is blocked (it would repeat "a b c").
    candidates_from: where to look for earlier n-grams (default: the sequence itself)."""
    if len(sequence) < size - 1:
        return set()
    start = tuple(sequence[-(size - 1):])
    source = sequence if candidates_from is None else candidates_from
    limit = len(source) - size + 1 if candidates_from is not None else len(source) - size + 1
    blocked = set()
    for position in range(max(0, limit)):
        if tuple(source[position:position + size - 1]) == start:
            ngram = source[position:position + size]
            if free is not None and all(free[token_id] for token_id in ngram):
                continue                    # only punctuation / spaces / short pieces: lists may repeat those
            blocked.add(source[position + size - 1])
    return blocked


def pick_distribution(model, probs, reply_ids, earlier_reply_ids, temperature, source_ids=None):
    """probs (from the model) -> the probabilities the dice actually uses. In this order:
      1. penalties on the scores: a word already used in this reply gets
             score - 0.2 (presence) - 0.2 x times_used (frequency)       e.g. used 3x: - 0.8 -> chance / 2.2
         (punctuation, spaces, digits and 1-2 letter pieces are free: lists and word endings need them)
      2. blocked tokens (probability 0):
             - anything that would repeat 4 tokens in a row from THIS reply       ("o cultură vibrantă, și o cultură vibrantă")
             - a word that was one of the last 2 words                            ("turism și turism")
             - anything that would copy 6 tokens in a row from an EARLIER reply   (no copying after a bad reply)
      3. temperature, then top-k 30 and top-p 0.85 (and min-p if set): the long tail of strange tokens is cut off
    source_ids: the tokens of the user's message (and of a <|context|> text). They get no penalty: a name or
    a number from the text may be quoted several times (with a penalty it would be changed into a wrong one)."""
    settings = model.sampling
    scores = np.log(np.maximum(probs, 1e-30))
    recent = reply_ids[-settings.penalty_window:]
    counts = {}
    for token_id in recent:
        counts[token_id] = counts.get(token_id, 0) + 1
    free = free_token_mask(model)
    for token_id, times in counts.items():
        if not free[token_id] and not (source_ids and token_id in source_ids):
            scores[token_id] -= settings.presence_penalty + settings.frequency_penalty * times
    blocked = blocked_by_ngrams(reply_ids, settings.no_repeat_ngram, free=free)
    if earlier_reply_ids:
        blocked |= blocked_by_ngrams(reply_ids, settings.no_copy_ngram, candidates_from=earlier_reply_ids, free=free)
    # "obiceiuri și obiceiuri", "turism și turism": a word can't come back 1-2 words after itself
    recent_words = [token_id for token_id in reply_ids[-6:] if not free[token_id]][-2:]
    blocked |= set(recent_words)
    for token_id in blocked:
        scores[token_id] = -np.inf
    scores = scores / temperature
    order = np.argsort(scores)[::-1][:settings.top_k]
    best = scores[order]
    chances = np.exp(best - best[0])
    chances = chances / chances.sum()
    kept = int(np.searchsorted(np.cumsum(chances), settings.top_p)) + 1
    if settings.min_p > 0:
        kept = min(kept, max(1, int(np.sum(chances >= settings.min_p * chances[0]))))
    result = np.zeros_like(probs)
    result[order[:kept]] = chances[:kept]
    if result.sum() == 0:                                  # everything blocked: fall back to the best token
        result[int(np.argmax(probs))] = 1.0
    return result / result.sum()


MODEL_REGISTRY = {
    "Spark":  (SparkBigram, "checkpoints/spark.npz", "Bigram · sees 1 character"),
    "Nebula": (NebulaMLP,   "checkpoints/nebula.npz", "MLP · sees 8 characters"),
    "Pulsar": (PulsarTransformer, "checkpoints/pulsar.npz", "Transformer · 3 blocks · sees 32 characters"),
    "Galaxy": (GalaxyChat, "checkpoints/galaxy_tokenizer.json", "Transformer · BPE tokens · sees 256 tokens"),
    "Nova": (SupernovaBase, "runs/supernova/final.pt", "Supernova before SFT · continues your text"),
    # Singularity after pre-training, BEFORE SFT: continues your text like Nova (compare the two side by side).
    # final.pt = finished; until then latest.pt (the newest save of the running training).
    "Singularity Base": (SupernovaBase, ["runs/singularity/final.pt", "runs/singularity/latest.pt"],
                         "Singularity before SFT · continues your text · 16 blocks"),
    # the newest SFT that exists: v2 (cleaned data) first, then v1
    # Singularity after SFT, with tools (calc, wiki, weather, ...). v4 = the best one so far
    "Singularity": (SingularityChat, ["runs/singularity_sft4/final.pt", "runs/singularity_sft3/final.pt", "runs/singularity_sft/final.pt"],
                    "After SFT · tools · 126M · EN + RO"),
    "Supernova": (SupernovaChat, ["runs/sft_v2/final.pt", "runs/sft/final.pt", "runs/sft_v2/latest.pt",
                                  "runs/sft_v2_pilot/final.pt", "runs/sft_pilot/final.pt"], "After SFT · answers you, EN + RO"),
}


def load_models(project_root):
    """Loads every model whose checkpoint file exists. Returns {name: (model, description)}."""
    models = {}
    for name in MODEL_REGISTRY:
        model_class, relative_path, description = MODEL_REGISTRY[name]
        if isinstance(relative_path, list):                 # several places: the first one that exists
            existing = [candidate for candidate in relative_path if os.path.exists(os.path.join(project_root, candidate))]
            relative_path = existing[0] if existing else relative_path[0]
        path = os.path.join(project_root, relative_path)
        if os.path.exists(path):
            try:
                models[name] = (model_class(path), description)
                print(f"  loaded {name:8} from {relative_path}")
            except Exception as error:          # e.g. Galaxy needs PyTorch: the other models still load
                print(f"  FAILED  {name:8} {type(error).__name__}: {error}")
        else:
            print(f"  skipped {name:8} ({relative_path} not found)")
    return models


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

END_OF_STORY = "<|endoftext|>"


def broken_marker_length(text):
    """How many characters at the end of text are a damaged <|endoftext|> (0 = none)."""
    if not text.endswith("|>"):
        return 0
    tail = text[-16:]
    for piece in ("<|", "<", "endof", "ndof", "dof", "text|>", "ext|>"):
        position = tail.rfind(piece)
        if position >= 0:
            return len(tail) - position
    return 2          # just "|>" (Spark writes it after any letter): drop those 2 and stop


def rounded(values):
    """numpy array -> plain list with 2 decimals (smaller to send to the page)"""
    result = []
    for value in values:
        result.append(round(float(value), 2))
    return result


def is_looping(reply_ids, window=48, max_distinct=14):
    """True when the reply is stuck in a loop: in its last 48 tokens there are at most 14 different ones.
    Example: "2021 - 2022 - 2023 - ..." uses only "202", the digits and " -" (~13 tokens) -> loop, stop.
    A normal sentence of 48 tokens has 30-40 different ones."""
    if len(reply_ids) < window:
        return False
    return len(set(reply_ids[-window:])) <= max_distinct


def generate_core(model, prompt, max_chars, temperature, rng, history=None, visualize=False, min_tokens=0, tool_messages=None):
    """
    Generates characters one by one and yields a dict for each one,
    with everything the interface wants to draw.   (tool_messages: what the tools found, for Singularity)
    """
    stoi = {}
    for index in range(len(model.chars)):
        stoi[model.chars[index]] = index

    is_chat = hasattr(model, "encode")
    if hasattr(model, "visualize"):
        model.visualize = visualize
    if is_chat and getattr(model, "uses_tools", False):
        if tool_messages:
            history = []                                                          # a tool answer uses a clean prompt
            ids = model.encode(prompt, [], tool_messages)
        else:
            ids = model.encode(prompt, history or [])
        max_chars = min(max_chars, getattr(model, "max_tokens", max_chars))
        if min_tokens > 0:
            max_chars = min(max_chars, min_tokens + 200)
    elif is_chat:
        ids = model.encode(prompt, history or [])   # BPE tokens: the whole chat in the conversation format
        max_chars = min(max_chars, getattr(model, "max_tokens", max_chars))
        if min_tokens > 0:
            max_chars = min(max_chars, min_tokens + 200)
    else:
        # prompt -> ids (characters the model never saw are skipped)
        ids = []
        for char in prompt:
            if char in stoi:
                ids.append(stoi[char])

        # if the prompt is shorter than the context, pad on the left with spaces
        pad_id = stoi.get(" ", 0)
        while len(ids) < model.context_size:
            ids.insert(0, pad_id)

    start_length = len(ids)
    prompt_end = len(ids)
    # the model's own earlier replies in this chat (Mom's lines), as token ids
    model_reply_ids = []
    if is_chat and hasattr(model, "visible_history"):          # Supernova / Singularity: only what the model sees
        for message in model.visible_history(history or []):
            if message["role"] == "assistant":
                model_reply_ids.extend(model.tokenizer.encode(message["content"]))
    elif is_chat:
        for turn in history or []:
            if turn["role"] != "user":
                reply = " " + turn["text"].strip()
                if hasattr(model, "clean"):
                    reply = model.clean(reply)           # e.g. a Supernova reply with "–" in the same chat
                model_reply_ids.extend(model.tokenizer.encode(reply))
    source_ids = None
    if is_chat and getattr(model, "sampling", None) is not None:
        source_ids = set(model.tokenizer.encode(prompt, allow_special=False))   # quoting the user is fine
    stop_text = getattr(model, "stop_text", None)
    wiki_answer = bool(tool_messages) and any(m["content"].startswith("wiki:") for m in tool_messages if m["role"] == "tool_call")
    guard_ids = fact_guard_ids(model, prompt, tool_messages) if wiki_answer else None      # names only after Wikipedia
    guard_words = known_names(prompt, tool_messages) if wiki_answer else None
    waiting = []          # steps held back because they could be the start of stop_text
    generated_text = ""
    for _ in range(max_chars):
        context_ids = ids[-model.context_size:]
        probs = model.next_char_probs(np.array(context_ids))
        if getattr(model, "block_ids", None):            # Singularity: the answer never starts a tool call (<|search|>, <|context|>)
            probs = probs.copy()
            probs[model.block_ids] = 0.0
            probs = probs / probs.sum()
        if guard_ids is not None and len(guard_ids):
            before_guard = probs
            guarded = probs.copy()
            guarded[guard_ids] = 0.0                     # a name or number the tool did not give: not allowed
            if guarded.sum() > 0.05:
                probs = guarded / guarded.sum()
            probs = finish_known_word(model, probs, generated_text, guard_words)
            if getattr(model, "last_inside", None) is not None:      # the Visualisation / Explore: show what was banned
                top_before = np.argsort(before_guard)[::-1][:8]
                model.last_inside["before_guard"] = [[model.chars[i], round(float(before_guard[i]), 4)] for i in top_before]
                model.last_inside["banned"] = [model.chars[i] for i in top_before if probs[i] == 0.0 and before_guard[i] > 0.002]

        # temperature: < 1 makes the model more confident, > 1 more random
        adjusted = probs ** (1.0 / temperature)
        top_p = getattr(model, "sampling", None)
        if top_p is not None:
            # Supernova / Singularity after SFT: penalties, n-gram blocking, top-k, top-p (pick_distribution)
            adjusted = pick_distribution(model, probs, ids[prompt_end:], model_reply_ids, temperature, source_ids)
        # repetition penalty (Galaxy): a word piece the model already wrote gets its chance divided
        # by 1.3 for EVERY time it appeared (up to 1.3^6), so loops like "| | | |" or "etern etern" die out
        elif is_chat:
            seen = {}
            for token_id in ids[prompt_end:] + model_reply_ids:
                seen[token_id] = seen.get(token_id, 0) + 1
            for token_id, times in seen.items():
                adjusted[token_id] /= 1.3 ** min(times, 6)
        # longer replies: until the minimum length, the tokens that would end the reply get probability 0
        produced = len(ids) - start_length
        if is_chat and produced < min_tokens:
            adjusted[model.reply_end_ids] = 0.0
        adjusted = adjusted / adjusted.sum()

        next_id = int(rng.choice(len(adjusted), p=adjusted))
        ids.append(next_id)
        next_char = model.chars[next_id]
        generated_text += next_char

        # top 5 candidates, for the probability bars
        top_ids = np.argsort(adjusted)[::-1][:5]
        top = []
        for candidate in top_ids:
            top.append([model.chars[candidate], float(adjusted[candidate])])

        if is_chat:
            context_chars = [model.chars[context_id] for context_id in context_ids[-32:]]   # one chip per token
            # a chat reply should not start with a space; a base model (Nova) continues your text, so it keeps it
            if len(generated_text) == len(next_char) and type(model) is not SupernovaBase:
                next_char = next_char.lstrip(" ")
        else:
            context_chars = ""
            for context_id in context_ids:
                context_chars += model.chars[context_id]

        step = {
            "type": "token",
            "char": next_char,
            "prob": float(adjusted[next_id]),
            "context": context_chars,
            "top": top,
        }

        attention = getattr(model, "last_attention", None)
        if attention is not None:
            step["attention"] = [round(float(value), 3) for value in attention]

        hidden = getattr(model, "last_hidden", None)
        if hidden is not None:
            step["neurons"] = [round(float(value), 2) for value in hidden]

        embedding = getattr(model, "last_input", None)
        if embedding is not None:
            step["embedding"] = [round(float(value), 2) for value in embedding]
            edges_in, edges_out = model.strongest_connections(top_ids)
            step["edges_in"] = edges_in
            step["edges_out"] = edges_out
            step["pre"] = [round(float(value), 2) for value in model.last_pre]
            scores = []                                     # raw logits of the top 5 characters
            for candidate in top_ids:
                scores.append(round(float(model.last_logits[candidate]), 2))
            step["scores"] = scores

        inside = getattr(model, "last_inside", None)
        if inside is not None:
            # the last stage: scores (logits) -> softmax -> temperature & penalties -> the dice picks one
            output = []
            for candidate in np.argsort(adjusted)[::-1][:8]:
                output.append([model.chars[candidate], round(float(model.last_logits[candidate]), 2),
                               round(float(probs[candidate]), 4), round(float(adjusted[candidate]), 4)])
            inside["output"] = output
            inside["model_top"] = [[model.chars[c], round(float(probs[c]), 4)] for c in np.argsort(probs)[::-1][:8]]   # before temperature / top-p
            inside["picked"] = [model.chars[next_id], round(float(model.last_logits[next_id]), 2),
                                round(float(probs[next_id]), 4), round(float(adjusted[next_id]), 4)]
            inside["temperature"] = temperature
            step["galaxy"] = inside
            step["scores"] = [round(float(model.last_logits[candidate]), 2) for candidate in top_ids]

        trace = getattr(model, "trace", None)
        if trace is not None:
            step["trace"] = {
                "embedding": rounded(trace["embedding"]),
                "blocks": [{name: rounded(block[name]) for name in block} for block in trace["blocks"]],
            }
            scores = []                                     # raw logits of the top 5 characters
            for candidate in top_ids:
                scores.append(round(float(model.last_logits[candidate]), 2))
            step["scores"] = scores

        if stop_text is not None:
            # hold steps back while the text could be turning into "\n**" (the next speaker)
            if stop_text in generated_text:
                yield {"type": "end_of_story"}
                return
            waiting.append(step)
            could_be_stop = False
            for size in range(1, len(stop_text)):
                if generated_text.endswith(stop_text[:size]):
                    could_be_stop = True
            if could_be_stop and not generated_text.endswith(END_OF_STORY):
                continue
            for waiting_step in waiting:
                yield waiting_step
            waiting = []
        else:
            yield step

        if top_p is not None and is_looping(ids[prompt_end:]):
            for waiting_step in waiting:
                yield waiting_step
            yield {"type": "end_of_story"}
            return
        if generated_text.endswith(END_OF_STORY):
            yield {"type": "end_of_story"}
            return
        # small models sometimes write the marker a bit wrong ("endoftext|>", "<|endofext|>"):
        # if the text ends with "|>" and a piece of the marker is near, treat it as the end too
        broken = 0 if is_chat else broken_marker_length(generated_text)
        if broken > 0:
            yield {"type": "end_of_story", "remove": broken}
            return

    for waiting_step in waiting:      # reached the length limit while holding a few steps back
        yield waiting_step


# words a reply may start with even if the tool result does not contain them
ALWAYS_ALLOWED = ("The It In This That These He She They A An On At Its His Her As For According Wikipedia Source Today Yes No I "
                  "You We There Here Pe Este El Ea În In Și Si Da Nu Azi Sursă Sursa Conform Acesta Aceasta Rezultatul Răspunsul "
                  "Raspunsul Sunt Am Are Au Is Was Were Heads Tails Ok OK Eu Rădăcina Radacina Radical Mâine Maine Tomorrow Hello Hi "
                  "Salut Bună Buna Sorry Unfortunately Here Nimic Nothing Pizza")


def fact_guard_ids(model, prompt, tool_messages):
    """The tokens that start with a capital letter or a digit (names, numbers) and do NOT appear in the question,
    the tool results or ALWAYS_ALLOWED.  After a tool, the model may not write them: a 126M model copies the
    sentence but invents the name ("invented by Francis Ronalds") or garbles the number.
    -> numpy array of token ids."""
    if not hasattr(model, "tokenizer"):
        return None
    if getattr(model, "_name_like_ids", None) is None:
        name_like = []
        for token_id, piece in enumerate(model.chars):
            first = piece.strip()[:1]
            if first and (first.isupper() or first.isdigit()) and "<|" not in piece:
                name_like.append(token_id)
        model._name_like_ids = name_like
    source = prompt + "\n" + "\n".join(m["content"] for m in tool_messages) + "\n" + ALWAYS_ALLOWED
    source = source + "\n" + " ".join(source.split())            # every word also with a space before it
    allowed = set(model.tokenizer.encode(source, allow_special=False))
    allowed |= set(model.tokenizer.encode(" " + ALWAYS_ALLOWED.replace(" ", "\n "), allow_special=False))
    return np.array([token_id for token_id in model._name_like_ids if token_id not in allowed], dtype=np.int64)


def known_names(prompt, tool_messages):
    """Every word that may start with a capital letter in the answer: the names in the question and the tool
    results, every word there written with a capital, and ALWAYS_ALLOWED."""
    source = prompt + " " + " ".join(m["content"] for m in tool_messages) + " " + ALWAYS_ALLOWED
    words = set()
    for word in re.findall(r"[^\W\d_][\w'’-]*", source):
        words.add(word)
        words.add(word[:1].upper() + word[1:])
    return words


def finish_known_word(model, probs, generated_text, names):
    """The first token of a name is checked by fact_guard_ids, but "Al" (from "Alesso") could go on as "Al|icante".
    While the reply is in the middle of a capitalized word, only tokens that continue it into a known name
    (or end it, if it already is one) are allowed.   -> the (maybe) changed probabilities."""
    current = re.search(r"(?<![\w'’-])([A-ZĂÂÎȘȚŞŢ][\w'’-]*)$", generated_text)
    if not current or not names:
        return probs
    word = current.group(1)
    rests = [name[len(word):] for name in names if name.startswith(word)]
    if not rests:
        return probs                                     # an unknown word already started: nothing to steer
    if getattr(model, "_pieces", None) is None:
        model._pieces = {}
        word_start = np.zeros(len(model.chars), dtype=bool)
        for token_id, piece in enumerate(model.chars):
            model._pieces.setdefault(piece, []).append(token_id)
            word_start[token_id] = not piece[:1].isalnum() and piece[:1] not in ("'", "’", "-", "_")
        model._ends_word = word_start
    keep = np.zeros(len(probs), dtype=bool)
    for rest in rests:
        if rest == "":
            keep |= model._ends_word                     # the word is complete: it may end here
        for size in range(1, len(rest) + 1):
            for token_id in model._pieces.get(rest[:size], ()):
                keep[token_id] = True
    steered = np.where(keep, probs, 0.0)
    if steered.sum() <= 1e-6:
        return probs
    return steered / steered.sum()


ANSWER_STOPS = ("Source:", "Sursă:", "Sursa:", "Source :", "(http", "http")     # the model's own link line (often misspelled): the page draws a card


def generate_stream(model, prompt, max_chars, temperature, rng, history=None, visualize=False, min_tokens=0):
    """generate_core + the tools of Singularity:
    1. the tools run first ("tool_start" / "tool" events: the page shows "Searching Wikipedia..." while it waits)
    2. the model writes its answer (its "Source: ..." line is cut, it often spells the link wrong)
    3. after a Wikipedia answer: a few more sentences straight from the article ("grounded" tokens, not the model's)
       and a "source" event with the real title and link."""
    tool_messages, wiki = None, None
    if hasattr(model, "encode") and getattr(model, "uses_tools", False):
        if hasattr(model, "visualize"):
            model.visualize = visualize
        for event in model.run_tools(prompt, history or []):
            yield event
        tool_messages, wiki = model.tool_messages, model.wiki_details
        if getattr(model, "abstain", None):             # a factual question with no source: say so, don't guess
            yield {"type": "token", "char": model.abstain, "prob": 1.0, "grounded": True, "abstain": True}
            yield {"type": "end_of_story"}
            return
        if getattr(model, "identity_reply", None):      # a question about the bot: the answer Alex wrote
            yield {"type": "token", "char": model.identity_reply, "prob": 1.0, "grounded": True, "identity": True}
            yield {"type": "end_of_story"}
            return
        if tool_messages and model.rewritten_prompt:
            prompt = model.rewritten_prompt              # the follow-up written out in full, for the answer
    stops = ANSWER_STOPS if wiki else ()
    text, held, ending = "", [], None
    for step in generate_core(model, prompt, max_chars, temperature, rng, history, visualize, min_tokens, tool_messages):
        if step.get("type") != "token" or not stops:
            if step.get("type") == "end_of_story" and stops:
                ending = step
                break
            yield step
            continue
        if "<|" in step["char"]:
            continue                                    # the end marker: the page must not see it before the extra text
        text += step["char"]
        if any(stop in text for stop in stops):
            held = []
            break
        held.append(step)
        tail_could_stop = any(text.endswith(stop[:size]) for stop in stops for size in range(1, len(stop)))
        if tail_could_stop or not "".join(h["char"] for h in held).strip():
            continue                                    # "\n\nSour..." waits until we know
        for waiting in held:
            yield waiting
        held = []
    if wiki:
        said = re.sub(r"(Source|Sursă|Sursa)\s*:.*$", "", text, flags=re.S)
        for waiting in held:                            # the end came while a few tokens waited: show real words
            if waiting["char"].strip():
                yield waiting
        more = grounded_sentences(said, wiki.get("long", ""))
        if more:
            yield {"type": "token", "char": "\n\n" + more, "prob": 1.0, "grounded": True}
        yield {"type": "source", "title": wiki["title"], "url": wiki["url"], "site": "Wikipedia"}
    if ending is not None:
        yield ending


def grounded_sentences(said, article_text, max_chars=700):
    """The article's sentences the model did not already say (at most max_chars), as one paragraph.
    A sentence counts as said if most of its words are already in the answer."""
    said_words = set(re.findall(r"\w+", said.lower()))
    picked = ""
    for sentence in re.split(r"(?<=[.!?])\s+(?=[A-ZĂÂÎȘȚ„\"0-9])", article_text.replace("\n\n", " ")):
        words = [w for w in re.findall(r"\w+", sentence.lower()) if len(w) > 3]
        if not words or sum(w in said_words for w in words) / len(words) > 0.6:
            continue
        if picked and len(picked) + len(sentence) > max_chars:
            break
        picked += (" " if picked else "") + sentence.strip()
    return picked
