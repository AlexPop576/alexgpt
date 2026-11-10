"""
The same BPE you wrote in bpe.py, made fast enough for billions of tokens, plus:
  - letters of every language (ă â î ș ț ...), not only A-Z
  - special tokens (<|user|>, <|assistant|>, ...), each one a single token

Two speed-ups, same results as your version:
  TRAINING  your loop counted every pair again after every merge.  Here the pair counts are kept
            and only the words that contain the merged pair are updated (a merge touches few words).
            The most frequent pair is taken from a heap instead of scanning all pairs.
  ENCODING  your encode_chunk tried all merges in order (16.000 of them for every new word).
            Here: look at the pairs inside the word, merge the one that was learned FIRST, repeat.
            Same result, but only a few steps per word.
"""

import heapq
import json
import re
import unicodedata

SPECIAL_TOKENS = ["<|endoftext|>", "<|system|>", "<|user|>", "<|assistant|>", "<|context|>", "<|search|>"]

# a chunk = an optional space + a word (letters of any language), up to 3 digits, punctuation, line breaks or spaces
CHUNK_PATTERN = re.compile(r" ?[^\W\d_]+| ?\d{1,3}| ?[^\w\s]+|\n+|\s+")
SPECIAL_PATTERN = re.compile("(" + "|".join(re.escape(token) for token in SPECIAL_TOKENS) + ")")


def normalize(text):
    """Same letter, one form: NFC unicode, and the old Romanian ş ţ (cedilla) -> ș ț (comma)."""
    text = unicodedata.normalize("NFC", text)
    return text.replace("ş", "ș").replace("Ş", "Ș").replace("ţ", "ț").replace("Ţ", "Ț")


class FastBPE:

    def __init__(self):
        self.chars = []          # id -> character, for the first ids
        self.merges = []         # [((a, b), new_id), ...] in the order they were learned
        self.pieces = []         # id -> text
        self.special_ids = {}    # "<|user|>" -> id
        self.cache = {}

    # ------------------------------------------------------------------ training
    def train(self, texts, vocab_size, min_char_count=30, report_every=1000):
        # 1. count the chunks (the same word appears thousands of times: count it once)
        chunk_counts = {}
        char_counts = {}
        for text in texts:
            for chunk in CHUNK_PATTERN.findall(normalize(text)):
                chunk_counts[chunk] = chunk_counts.get(chunk, 0) + 1
        for chunk, count in chunk_counts.items():
            for char in chunk:
                char_counts[char] = char_counts.get(char, 0) + count

        # 2. the alphabet: characters seen often enough (rare symbols are dropped)
        always = set(chr(code) for code in range(32, 127)) | {"\n"} | set("ăâîșțĂÂÎȘȚ")
        self.chars = sorted(always | set(char for char, count in char_counts.items() if count >= min_char_count))
        char_id = {char: index for index, char in enumerate(self.chars)}
        self.pieces = list(self.chars)

        # 3. every different chunk as a list of ids, with how many times it appears
        words, counts = [], []
        for chunk, count in chunk_counts.items():
            ids = [char_id[char] for char in chunk if char in char_id]
            if len(ids) >= 2:
                words.append(ids)
                counts.append(count)

        # 4. pair -> total count, pair -> which words contain it
        pair_counts, pair_words = {}, {}
        for index, ids in enumerate(words):
            for pair in zip(ids, ids[1:]):
                pair_counts[pair] = pair_counts.get(pair, 0) + counts[index]
                pair_words.setdefault(pair, set()).add(index)
        heap = [(-count, pair) for pair, count in pair_counts.items()]
        heapq.heapify(heap)

        target = vocab_size - len(SPECIAL_TOKENS)
        while len(self.pieces) < target and heap:
            negative, pair = heapq.heappop(heap)
            if pair_counts.get(pair, 0) != -negative or -negative < 2:
                continue                                    # an old entry: the count changed since
            new_id = len(self.pieces)
            self.merges.append((pair, new_id))
            self.pieces.append(self.pieces[pair[0]] + self.pieces[pair[1]])

            for index in list(pair_words.get(pair, ())):
                ids, count = words[index], counts[index]
                for old in zip(ids, ids[1:]):               # take this word's pairs out
                    pair_counts[old] -= count
                    if old != pair and pair_counts[old] > 0:
                        heapq.heappush(heap, (-pair_counts[old], old))
                merged, i = [], 0
                while i < len(ids):
                    if i + 1 < len(ids) and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                        merged.append(new_id)
                        i += 2
                    else:
                        merged.append(ids[i])
                        i += 1
                words[index] = merged
                for new in zip(merged, merged[1:]):         # and put the new pairs back
                    pair_counts[new] = pair_counts.get(new, 0) + count
                    pair_words.setdefault(new, set()).add(index)
                    heapq.heappush(heap, (-pair_counts[new], new))
            pair_counts.pop(pair, None)
            pair_words.pop(pair, None)
            if report_every and len(self.merges) % report_every == 0:
                print(f"  {len(self.pieces):6d} tokens   newest: {self.pieces[-1]!r}", flush=True)

        self.add_specials()

    def add_specials(self):
        self.special_ids = {}
        for token in SPECIAL_TOKENS:
            self.special_ids[token] = len(self.pieces)
            self.pieces.append(token)
        self.char_id = {char: index for index, char in enumerate(self.chars)}
        self.rank = {pair: rank for rank, (pair, _) in enumerate(self.merges)}
        self.merged_id = {pair: new_id for pair, new_id in self.merges}

    # ------------------------------------------------------------------ encoding
    def encode_chunk(self, chunk):
        cached = self.cache.get(chunk)
        if cached is not None:
            return cached
        ids = [self.char_id[char] for char in chunk if char in self.char_id]
        while len(ids) >= 2:
            # the pair in this word that was learned first
            best_rank, best_index = None, -1
            for i in range(len(ids) - 1):
                rank = self.rank.get((ids[i], ids[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank, best_index = rank, i
            if best_rank is None:
                break
            pair = (ids[best_index], ids[best_index + 1])
            ids[best_index:best_index + 2] = [self.merged_id[pair]]
        if len(self.cache) < 2_000_000:
            self.cache[chunk] = ids
        return ids

    def encode(self, text, allow_special=True):
        """allow_special=False: "<|user|>" written inside a message stays plain text (several normal tokens).
        Use it for anything a person wrote (chat messages, data files); the markers are then added as ids."""
        ids = []
        for part in SPECIAL_PATTERN.split(normalize(text)):
            if allow_special and part in self.special_ids:
                ids.append(self.special_ids[part])
            elif part:
                for chunk in CHUNK_PATTERN.findall(part):
                    ids.extend(self.encode_chunk(chunk))
        return ids

    def decode(self, ids):
        return "".join(self.pieces[token_id] for token_id in ids)

    @property
    def vocab_size(self):
        return len(self.pieces)

    @property
    def end_id(self):
        return self.special_ids["<|endoftext|>"]

    # ------------------------------------------------------------------ files
    def save(self, path):
        data = {"chars": self.chars, "merges": [[a, b, new_id] for (a, b), new_id in self.merges],
                "specials": SPECIAL_TOKENS}
        with open(path, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False)

    @staticmethod
    def load(path):
        with open(path, encoding="utf-8") as file:
            data = json.load(file)
        tokenizer = FastBPE()
        tokenizer.chars = data["chars"]
        tokenizer.pieces = list(tokenizer.chars)
        for a, b, new_id in data["merges"]:
            tokenizer.merges.append(((a, b), new_id))
            tokenizer.pieces.append(tokenizer.pieces[a] + tokenizer.pieces[b])
        tokenizer.add_specials()
        return tokenizer
