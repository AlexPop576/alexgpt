"""
BPE tokenizer: learns its own pieces of words from the text.

Training (learn):
    1. start with one id per character
    2. count every pair of neighbouring ids
    3. the most frequent pair becomes a new id; replace it everywhere
    4. repeat until the vocabulary is big enough
Encoding: apply the learned merges, in the same order.
"""

import json
import re

END_OF_TEXT = "<|endoftext|>"

# text is first cut into chunks (a word with the space before it, a number, punctuation, spaces);
# merges never cross a chunk, so " the" can become one token but " the dog" never will
CHUNK_PATTERN = re.compile(r" ?[A-Za-z]+| ?[0-9]+| ?[^A-Za-z0-9\s]+|\s+")


def merge_pair(ids, pair, new_id):
    result = []
    i = 0
    while i < len(ids):
        if i + 1 < len(ids) and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            result.append(new_id)
            i+=2
        else:
            result.append(ids[i])
            i+=1
    return result


def count_pairs(ids, counts, weight):
    # STEP 2 (yours): for every pair of neighbours in ids, add `weight` to counts[pair]
    # ids = [5, 7, 5, 7], weight = 10  ->  counts[(5, 7)] += 20,  counts[(7, 5)] += 10
    # counts is a dictionary: pair -> how many times it was seen (a missing pair counts as 0)
    for i in range(len(ids)-1):
        pair = (ids[i], ids[i + 1]) 
        counts[pair] = counts.get(pair, 0) + weight
    pass


class BPETokenizer:

    def __init__(self):
        self.chars = []          # the starting vocabulary: one id per character
        self.merges = []         # learned merges, in order: [((a, b), new_id), ...]
        self.pieces = []         # id -> the text it stands for
        self.end_id = None       # id of <|endoftext|>
        self.cache = {}          # chunk -> ids (encoding the same word twice is free)

    def train(self, text, vocab_size, all_chars=None, report_every=100):
        # 1. one id per character (all_chars: characters that must exist even if this text lacks them)
        char_set = set(text)
        if all_chars is not None:
            char_set = char_set | set(all_chars)
        self.chars = sorted(char_set)
        self.pieces = list(self.chars)
        char_id = {}
        for index in range(len(self.chars)):
            char_id[self.chars[index]] = index

        # every different chunk, with how many times it appears (" the" appears thousands of times)
        chunk_counts = {}
        for chunk in CHUNK_PATTERN.findall(text):
            chunk_counts[chunk] = chunk_counts.get(chunk, 0) + 1
        words = []               # [ids, count] for every different chunk
        for chunk in chunk_counts:
            ids = []
            for char in chunk:
                ids.append(char_id[char])
            words.append([ids, chunk_counts[chunk]])

        # 2-4. STEP 3 (yours): merge the most frequent pair, again and again
        while len(self.pieces) < vocab_size - 1:          # -1: keep one place for <|endoftext|>
            # a. counts = {}; for every [ids, count] in words: count_pairs(ids, counts, count)
            counts = {}
            for ids, count in words:
                count_pairs(ids, counts, count)
            # b. best_pair = the pair with the biggest count:   max(counts, key=counts.get)
            best_pair = max(counts, key = counts.get)
            # c. new_id = len(self.pieces)                      (the next free number)
            new_id = len(self.pieces)
            # d. for every word in words: word[0] = merge_pair(word[0], best_pair, new_id)
            for word in words:
                word[0] = merge_pair(word[0], best_pair, new_id)
            # e. remember it:  self.merges.append((best_pair, new_id))
            #                  self.pieces.append(self.pieces[best_pair[0]] + self.pieces[best_pair[1]])
            self.merges.append((best_pair, new_id))
            self.pieces.append((self.pieces[best_pair[0]] + self.pieces[best_pair[1]]))
            # f. every `report_every` merges, print len(self.pieces) and the newest piece

        self.end_id = len(self.pieces)
        self.pieces.append(END_OF_TEXT)

    def encode_chunk(self, chunk):
        if chunk in self.cache:
            return self.cache[chunk]
        ids = []
        for char in chunk:
            ids.append(self.char_id[char])
        for pair, new_id in self.merges:
            ids = merge_pair(ids, pair, new_id)
        self.cache[chunk] = ids
        return ids

    def encode(self, text):
        # <|endoftext|> is one special token; everything between is cut into chunks
        ids = []
        parts = text.split(END_OF_TEXT)
        for index in range(len(parts)):
            if index > 0:
                ids.append(self.end_id)
            for chunk in CHUNK_PATTERN.findall(parts[index]):
                ids.extend(self.encode_chunk(chunk))
        return ids

    def decode(self, ids):
        text = ""
        for token_id in ids:
            text += self.pieces[token_id]
        return text

    @property
    def vocab_size(self):
        return len(self.pieces)

    @property
    def char_id(self):
        if not hasattr(self, "_char_id"):
            self._char_id = {}
            for index in range(len(self.chars)):
                self._char_id[self.chars[index]] = index
        return self._char_id

    def save(self, path):
        data = {"chars": self.chars, "merges": [[a, b, new_id] for (a, b), new_id in self.merges]}
        with open(path, "w", encoding="utf-8") as file:
            json.dump(data, file)

    @staticmethod
    def load(path):
        with open(path, encoding="utf-8") as file:
            data = json.load(file)
        tokenizer = BPETokenizer()
        tokenizer.chars = data["chars"]
        tokenizer.pieces = list(tokenizer.chars)
        for a, b, new_id in data["merges"]:
            tokenizer.merges.append(((a, b), new_id))
            tokenizer.pieces.append(tokenizer.pieces[a] + tokenizer.pieces[b])
        tokenizer.end_id = len(tokenizer.pieces)
        tokenizer.pieces.append(END_OF_TEXT)
        return tokenizer
