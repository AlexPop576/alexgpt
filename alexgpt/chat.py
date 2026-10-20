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
def pick_device(torch, needed_gb):
    """The GPU only if it has room left (Supernova's training may be using almost all of it), else the CPU."""
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
                "start": guesses(0), "blocks": blocks}


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
        self.device = pick_device(torch, 1.0)        # while Supernova trains, the GPU is full: then the CPU

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
        self.device = pick_device(torch, 1.5)
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
        self.model.load_state_dict(state["model"])
        self.model.to(self.device)
        self.model.eval()
        self.step = state["step"]
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
                x, cos, sin = inputs
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


MODEL_REGISTRY = {
    "Spark":  (SparkBigram, "checkpoints/spark.npz", "Bigram · sees 1 character"),
    "Nebula": (NebulaMLP,   "checkpoints/nebula.npz", "MLP · sees 8 characters"),
    "Pulsar": (PulsarTransformer, "checkpoints/pulsar.npz", "Transformer · 3 blocks · sees 32 characters"),
    "Galaxy": (GalaxyChat, "checkpoints/galaxy_tokenizer.json", "Transformer · BPE tokens · sees 256 tokens"),
    "Supernova": (SupernovaBase, "runs/supernova/latest.pt", "In training · base model, continues your text"),
}


def load_models(project_root):
    """Loads every model whose checkpoint file exists. Returns {name: (model, description)}."""
    models = {}
    for name in MODEL_REGISTRY:
        model_class, relative_path, description = MODEL_REGISTRY[name]
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


def generate_stream(model, prompt, max_chars, temperature, rng, history=None, visualize=False, min_tokens=0):
    """
    Generates characters one by one and yields a dict for each one,
    with everything the interface wants to draw.
    """
    stoi = {}
    for index in range(len(model.chars)):
        stoi[model.chars[index]] = index

    is_chat = hasattr(model, "encode")
    if hasattr(model, "visualize"):
        model.visualize = visualize
    if is_chat:
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
    if is_chat:
        for turn in history or []:
            if turn["role"] != "user":
                reply = " " + turn["text"].strip()
                if hasattr(model, "clean"):
                    reply = model.clean(reply)           # e.g. a Supernova reply with "–" in the same chat
                model_reply_ids.extend(model.tokenizer.encode(reply))
    stop_text = getattr(model, "stop_text", None)
    waiting = []          # steps held back because they could be the start of stop_text
    generated_text = ""
    for _ in range(max_chars):
        context_ids = ids[-model.context_size:]
        probs = model.next_char_probs(np.array(context_ids))

        # temperature: < 1 makes the model more confident, > 1 more random
        adjusted = probs ** (1.0 / temperature)
        # repetition penalty (chat models): a word piece the model already wrote gets its chance divided
        # by 1.3 for EVERY time it appeared (up to 1.3^6), so loops like "| | | |" or "etern etern" die out
        if is_chat:
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
            if len(generated_text) == len(next_char):
                next_char = next_char.lstrip(" ")       # the reply should not start with a space
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
