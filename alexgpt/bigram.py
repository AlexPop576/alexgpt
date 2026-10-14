import numpy as np
from alexgpt.tokenizer import CharTokenizer

def count_bigrams(ids, vocab_size):
    N = np.zeros((vocab_size, vocab_size), dtype = np.int64)
    np.add.at(N, (ids[:-1], ids[1:]), 1)
    return N

def to_probs(N, smoothing=1.0):
    N = N.astype(np.float64) + smoothing
    sums = N.sum(axis=1, keepdims=True)
    P = N/sums
    return P

def generate(P, start_id, n, rng):
    gen_text = [start_id]
    current = start_id
    for _ in range (n):
        new_char = rng.choice(len(P), p=P[current])
        current = new_char
        gen_text.append(new_char)
    return gen_text

def nll_loss(P, ids):
    prob_list = P[np.asarray(ids[:-1]), np.asarray(ids[1:])]
    return -np.log(prob_list).mean()