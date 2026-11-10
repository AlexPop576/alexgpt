"""
Where Singularity's text comes from.

Two folders on the SSD:
  C:/alexgpt_data/supernova/     the ~2B tokens Supernova already read (reused, a bit less often: "old_...")
  C:/alexgpt_data/singularity/   ~2.7B NEW tokens, read from the same datasets starting where Supernova stopped
                                 (prepare_singularity.py), so no document is read twice by accident

New: fineweb_edu is split in two by its quality score (0..5, given by a classifier):
  score 4-5 -> fineweb_edu_hq   (the best ~10-20%: used a lot at the end of training)
  score 3   -> fineweb_edu
"""

SUPERNOVA_DIR = "C:/alexgpt_data/supernova"
SINGULARITY_DIR = "C:/alexgpt_data/singularity"

# NEW data. name: (dataset on Hugging Face, config, language, tokens we want)
NEW_SOURCES = {
    "fineweb_edu":    ("HuggingFaceFW/fineweb-edu", "sample-10BT", "en", 1_000_000_000),
    "cosmopedia":     ("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", "en", 400_000_000),
    "wiki_en":        ("wikimedia/wikipedia", "20231101.en", "en", 350_000_000),
    "wiki_ro":        ("wikimedia/wikipedia", "20231101.ro", "ro", 300_000_000),     # probably ends sooner: all of it
    "fineweb2_ro":    ("HuggingFaceFW/fineweb-2", "ron_Latn", "ro", 550_000_000),
}
HQ_SOURCE = "fineweb_edu_hq"      # filled while reading fineweb_edu (no target of its own)
HQ_MIN_SCORE = 4

# language of every source the training can see (for the en / ro validation loss)
LANGUAGE = {"fineweb_edu": "en", "fineweb_edu_hq": "en", "cosmopedia": "en", "wiki_en": "en", "soda": "en",
            "wiki_ro": "ro", "fineweb2_ro": "ro"}

# how often each source is picked (adds up to 1). "old_x" = Supernova's data, a second time.
TRAIN_MIX = {
    "fineweb_edu": 0.28, "fineweb_edu_hq": 0.06, "cosmopedia": 0.10, "wiki_en": 0.08,
    "wiki_ro": 0.015, "fineweb2_ro": 0.17,                                               # new: 0.705
    "old_fineweb_edu": 0.09, "old_cosmopedia": 0.04, "old_wiki_en": 0.03, "old_soda": 0.02,
    "old_wiki_ro": 0.05, "old_fineweb2_ro": 0.065,                                       # old: 0.295
}
# Changed at step ~1500 (22 oct): the new wiki_ro has only 59M tokens (mostly the end of the dump: lists, stubs).
# At 0.08 it would have been read 5.7 times. Now every source is read about once (see the table in the plan doc).
# the last 15% (learning rate going down to 0): the cleanest text. Romanian stays ~30%.
# fineweb_edu_hq only 10%: the FineWeb-Edu card says scores 4-5 help knowledge tests but "significantly degrade
# performance on HellaSwag and PIQA" (the benchmarks where we must beat GPT-2). Score 3+ (fineweb_edu) was best overall.
ANNEAL_MIX = {
    "fineweb_edu_hq": 0.10, "fineweb_edu": 0.20, "cosmopedia": 0.15, "wiki_en": 0.15,
    "wiki_ro": 0.03, "fineweb2_ro": 0.15,
    "old_soda": 0.05, "old_wiki_ro": 0.12, "old_cosmopedia": 0.05,
}

# Decay B (an experiment, run after decay A from the same checkpoint): the same text + 3.5% conversations written
# by people (OpenAssistant EN, NoRobots/Dolly/OASST RO) in the chat format, and GSM8K math explained step by step.
# prepare_anneal_b.py writes chat_human_train.bin and gsm8k_train.bin.
ANNEAL_EXTRA_B = {"chat_human": 0.035, "gsm8k": 0.005}   # chat: 9.3M tokens -> read ~2x; gsm8k ~2x
ANNEAL_MIX_B = {name: weight * (1 - sum(ANNEAL_EXTRA_B.values())) for name, weight in ANNEAL_MIX.items()}
ANNEAL_MIX_B.update(ANNEAL_EXTRA_B)
LANGUAGE.update({"chat_human": "mixed", "gsm8k": "en"})
