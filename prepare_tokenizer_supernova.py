"""
Step 1 for Supernova:  python prepare_tokenizer_supernova.py
  1. streams a small sample (~70 MB of text) from every source, ~38% of it Romanian
  2. learns a BPE tokenizer with 16384 tokens on it (your algorithm, the fast version)
  3. prints how many tokens a word costs in English and in Romanian (fertility)
Saves checkpoints/supernova_tokenizer.json. Takes about 10-20 minutes.
"""
import builtins
import os
import time
from alexgpt.bpe_fast import FastBPE, CHUNK_PATTERN, normalize
from alexgpt.data_sources import SOURCES, stream_documents

os.makedirs("runs", exist_ok=True)
report_file = open("runs/tokenizer_report.txt", "w", encoding="utf-8")


def print(*parts, **options):                  # everything printed also goes to runs/tokenizer_report.txt
    builtins.print(*parts, **options)
    if options.get("end", "\n") == "\n":
        report_file.write(" ".join(str(part) for part in parts) + "\n")
        report_file.flush()


VOCAB_SIZE = 16384
SAMPLE_MB = {"fineweb_edu": 25, "cosmopedia": 10, "wiki_en": 6, "soda": 2, "wiki_ro": 12, "fineweb2_ro": 15}
TEST_DOCS = 200          # documents per language kept aside to measure fertility

start = time.time()
train_texts = []
test_texts = {"en": [], "ro": []}
for source, megabytes in SAMPLE_MB.items():
    language = SOURCES[source][2]
    collected = 0
    print(f"reading {megabytes} MB from {source} ...", flush=True)
    for count, text in stream_documents(source):
        if len(test_texts[language]) < TEST_DOCS and count % 50 == 0:
            test_texts[language].append(text)
            continue
        train_texts.append(text)
        collected += len(text.encode("utf-8"))
        if collected >= megabytes * 1_000_000:
            break
print(f"sample: {sum(len(t) for t in train_texts) / 1e6:.0f} M characters   ({time.time() - start:.0f} s)\n")

print("learning BPE ...", flush=True)
tokenizer = FastBPE()
tokenizer.train(train_texts, VOCAB_SIZE, report_every=2000)
tokenizer.save("checkpoints/supernova_tokenizer.json")
print(f"saved checkpoints/supernova_tokenizer.json   vocabulary {tokenizer.vocab_size}   ({time.time() - start:.0f} s)\n")

# fertility: how many tokens one word costs (1.0 = every word is a single token)
for language in ("en", "ro"):
    words = 0
    tokens = 0
    characters = 0
    for text in test_texts[language]:
        text = normalize(text)
        words += len(text.split())
        tokens += len(tokenizer.encode(text))
        characters += len(text)
    print(f"{language}: {tokens / max(words, 1):.2f} tokens per word   {characters / max(tokens, 1):.1f} characters per token")

for example in ["The children played outside because the weather was beautiful.",
                "Copiii s-au jucat afară pentru că vremea era frumoasă.",
                "Universitățile din România își publică rezultatele în această săptămână.",
                "<|user|>Salut! Ce mai faci?<|assistant|>Bine, mersi! Tu?"]:
    pieces = [tokenizer.pieces[token_id] for token_id in tokenizer.encode(example)]
    print(f"\n{len(pieces):3d} tokens  |" + "|".join(pieces) + "|")
