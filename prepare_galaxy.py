"""
Prepares the data for Galaxy:  python prepare_galaxy.py
  1. reads the conversations (TinyDialogues) and the stories (TinyStories)
  2. cleans them: real line breaks, only simple English characters
  3. learns the BPE tokenizer on a part of the text
  4. turns all the text into token ids and saves them
"""
import random
import time
import numpy as np
from alexgpt.bpe import BPETokenizer, END_OF_TEXT

VOCAB_SIZE = 2048
ALLOWED = set(chr(code) for code in range(32, 127)) | {"\n"}     # printable English + line break


def clean(text):
    text = text.replace(" \\n\\n ", "\n").replace("\\n\\n", "\n").replace("\\n", "\n")
    result = []
    for char in text:
        if char in ALLOWED:
            result.append(char)
    return "".join(result).strip()


def read_documents(path):
    # one document = one conversation or one story, separated by <|endoftext|>
    text = open(path, encoding="utf-8").read()
    documents = []
    for part in text.split(END_OF_TEXT):
        part = clean(part)
        if len(part) > 20:
            documents.append(part)
    return documents


start = time.time()
dialogues_train = read_documents("data/tinydialogue_train_ordered.txt")
dialogues_val = read_documents("data/tinydialogue_val_ordered.txt")
stories = read_documents("data/TinyStories-valid.txt")
split = int(0.9 * len(stories))
print(f"conversations: {len(dialogues_train)} train, {len(dialogues_val)} val   stories: {len(stories)}   ({time.time() - start:.0f} s)")

random.seed(0)
train_documents = dialogues_train + stories[:split]
val_documents = dialogues_val + stories[split:]
random.shuffle(train_documents)          # mix conversations and stories (the file is ordered by age)
random.shuffle(val_documents)
train_text = END_OF_TEXT.join(train_documents[:3000])      # only used to learn BPE
train_chars = 0
for document in train_documents:
    train_chars += len(document)
print(f"characters: {train_chars:,} train")

print("learning BPE on 3000 documents...")
tokenizer = BPETokenizer()
tokenizer.train(train_text, VOCAB_SIZE, all_chars=ALLOWED, report_every=200)
tokenizer.save("checkpoints/galaxy_tokenizer.json")
print(f"vocabulary: {tokenizer.vocab_size}   ({time.time() - start:.0f} s)")

def encode_documents(documents):
    # one document at a time, so memory stays small; <|endoftext|> after each one
    import array
    ids = array.array("H")                 # "H" = 16-bit numbers (our ids are < 65536)
    for document in documents:
        ids.extend(tokenizer.encode(document))
        ids.append(tokenizer.end_id)
    return np.array(ids, dtype=np.uint16)

print("encoding all the text...")
train_ids = encode_documents(train_documents)
val_ids = encode_documents(val_documents)
np.save("data/galaxy_train.npy", train_ids)
np.save("data/galaxy_val.npy", val_ids)
print(f"tokens: {len(train_ids):,} train, {len(val_ids):,} val   "
      f"({train_chars / len(train_ids):.1f} characters per token)   ({time.time() - start:.0f} s)")

example = val_documents[0][:200]
pieces = []
for token_id in tokenizer.encode(example):
    pieces.append(tokenizer.pieces[token_id])
print("example:", pieces[:40])
