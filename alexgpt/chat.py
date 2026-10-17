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

class PebbleBigram:
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


class SlateMLP:
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

class QuartzTransformer:
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
MODEL_REGISTRY = {
    "Pebble": (PebbleBigram, "checkpoints/pebble.npz", "Bigram · sees 1 character"),
    "Slate":  (SlateMLP,     "checkpoints/slate.npz",  "MLP · sees 8 characters"),
    "Quartz": (QuartzTransformer, "checkpoints/quartz.npz", "Transformer · 3 blocks · sees 32 characters"),
}


def load_models(project_root):
    """Loads every model whose checkpoint file exists. Returns {name: (model, description)}."""
    models = {}
    for name in MODEL_REGISTRY:
        model_class, relative_path, description = MODEL_REGISTRY[name]
        path = os.path.join(project_root, relative_path)
        if os.path.exists(path):
            models[name] = (model_class(path), description)
            print(f"  loaded {name:8} from {relative_path}")
        else:
            print(f"  skipped {name:8} ({relative_path} not found)")
    return models


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

END_OF_STORY = "<|endoftext|>"


def rounded(values):
    """numpy array -> plain list with 2 decimals (smaller to send to the page)"""
    result = []
    for value in values:
        result.append(round(float(value), 2))
    return result


def generate_stream(model, prompt, max_chars, temperature, rng):
    """
    Generates characters one by one and yields a dict for each one,
    with everything the interface wants to draw.
    """
    stoi = {}
    for index in range(len(model.chars)):
        stoi[model.chars[index]] = index

    # prompt -> ids (characters the model never saw are skipped)
    ids = []
    for char in prompt:
        if char in stoi:
            ids.append(stoi[char])

    # if the prompt is shorter than the context, pad on the left with spaces
    pad_id = stoi.get(" ", 0)
    while len(ids) < model.context_size:
        ids.insert(0, pad_id)

    generated_text = ""
    for _ in range(max_chars):
        context_ids = ids[-model.context_size:]
        probs = model.next_char_probs(np.array(context_ids))

        # temperature: < 1 makes the model more confident, > 1 more random
        adjusted = probs ** (1.0 / temperature)
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

        yield step

        if generated_text.endswith(END_OF_STORY):
            yield {"type": "end_of_story"}
            return
