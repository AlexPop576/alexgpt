"""
Where Supernova's text comes from. Everything is read with STREAMING: the file on Hugging Face is
opened over the internet and read document by document, like a video on YouTube, nothing big is
downloaded first. If the connection drops, we open it again and skip the documents already done.
"""
import time

# name: (dataset on Hugging Face, config, language, tokens we want for training)
SOURCES = {
    "fineweb_edu": ("HuggingFaceFW/fineweb-edu", "sample-10BT", "en", 900_000_000),   # educational web pages
    "cosmopedia":  ("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", "en", 330_000_000),  # textbooks written by a big model
    "wiki_en":     ("wikimedia/wikipedia", "20231101.en", "en", 170_000_000),          # facts, names, dates
    "soda":        ("allenai/soda", None, "en", 60_000_000),                           # everyday conversations
    "wiki_ro":     ("wikimedia/wikipedia", "20231101.ro", "ro", 160_000_000),
    "fineweb2_ro": ("HuggingFaceFW/fineweb-2", "ron_Latn", "ro", 450_000_000),         # Romanian web, filtered
}

# how often each source is picked while training (they add up to 1)
TRAIN_MIX = {"fineweb_edu": 0.45, "cosmopedia": 0.15, "wiki_en": 0.075, "soda": 0.025,
             "wiki_ro": 0.10, "fineweb2_ro": 0.20}
# the last ~15% of training (the learning rate goes down): only the cleanest text
ANNEAL_MIX = {"fineweb_edu": 0.25, "cosmopedia": 0.30, "wiki_en": 0.15, "soda": 0.05,
              "wiki_ro": 0.20, "fineweb2_ro": 0.05}


def document_text(source, row):
    """One row of the dataset -> plain text (or None to skip it)."""
    if source == "soda":
        lines = []
        for speaker, line in zip(row.get("speakers") or [], row.get("dialogue") or []):
            lines.append(f"{speaker}: {line}")
        text = "\n".join(lines)
        return text if len(text) >= 50 else None
    text = row.get("text") or ""
    return text if len(text) >= 200 else None


def stream_documents(source, skip=0):
    """Yields the texts of `source`, starting after `skip` documents. Reconnects by itself."""
    from datasets import load_dataset
    path, config, _, _ = SOURCES[source]
    done = skip
    failures = 0
    while True:
        try:
            dataset = load_dataset(path, config, split="train", streaming=True)
            if done:
                dataset = dataset.skip(done)
            for row in dataset:
                done += 1
                failures = 0
                text = document_text(source, row)
                if text is not None:
                    yield done, text
            return                                        # the whole dataset was read
        except Exception as error:                        # network hiccup: wait and continue where we were
            failures += 1
            if failures > 20:
                raise
            wait = min(300, 15 * failures)
            print(f"\n  [{source}] connection problem ({type(error).__name__}: {str(error)[:120]}), "
                  f"retrying in {wait} s ...", flush=True)
            time.sleep(wait)
