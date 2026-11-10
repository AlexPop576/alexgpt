# Data used for Singularity (SFT stage)

Pre-training data: **TODO (Alex): list the pre-training corpora and their licenses here.**

## Supervised fine-tuning mix (built by `build_sft_mix.py`)

### Public datasets (downloaded with `prepare_sft.py`)
| Source (id in the mix) | Dataset | Language | Notes / license (as far as I know, check before publishing) |
|---|---|---|---|
| smol_general, smol_everyday, smol_summarize, smol_rewrite, smol_constraints | HuggingFaceTB/smoltalk | EN | Apache-2.0; synthetic, generated with other LLMs (Llama / Qwen family), so those models' terms apply to the outputs |
| ro_ultrachat | OpenLLM-Ro/ro_sft_ultrachat | RO | translation of UltraChat (ChatGPT-generated dialogues) |
| ro_alpaca | OpenLLM-Ro/ro_sft_alpaca | RO | translation of Alpaca (text-davinci-003 outputs): **CC BY-NC 4.0 = non-commercial** |
| ro_dolly | OpenLLM-Ro/ro_sft_dolly | RO | translation of Databricks Dolly 15k (CC BY-SA 3.0) |
| ro_norobots | OpenLLM-Ro/ro_sft_norobots | RO | translation of No Robots, human written: **CC BY-NC 4.0 = non-commercial** |
| en_oasst, ro_oasst | OpenAssistant/oasst_top1_2023-08-25 and its RO translation | EN / RO | Apache-2.0 |

Because of ro_alpaca and ro_norobots the trained model is for **research / portfolio use, not commercial use**, unless these two sources are removed and the model is retrained.
Some of these datasets were produced with OpenAI / Meta models; their terms of use may restrict using the outputs to build competing commercial models.

### Data made by code in this repository (no external license, facts computed or verified)
| Source | Script | What it is |
|---|---|---|
| gen_tools | gen_data_tools.py | tool-call conversations (calc, convert, time, weather, dice, coin, pick, memory); the results are computed by the script |
| gen_constraints | gen_data_constraints.py | format instructions (one word, yes/no, bold, lists of N, headings) with answers made by code |
| gen_markdown | gen_data_markdown.py | lists, tables, code blocks made from fixed content |
| gen_idk | gen_data_idk.py | "I don't know" answers (made-up names, live data, future events, unclear requests) |
| gen_facts | gen_data_facts.py | short facts answered after a `wiki:` call; each answer is checked to appear in the fetched Wikipedia intro; also not-found cases for made-up names |

### Data from Wikipedia (CC BY-SA 4.0 — attribution and share-alike apply; the source URL is kept in the answers)
| Source | Script | What it is |
|---|---|---|
| gen_wiki | gen_data_wiki.py | `wiki:` call, Wikipedia intro as tool result, short answer with source link |
| gen_grounded | gen_grounded.py | summaries / Q&A / "not in the text" answers from real Wikipedia passages, checked automatically (numbers and words must come from the text; sentence counts verified; an LLM verifier for support) |

### Data written with a local LLM (EuroLLM-9B-Instruct via Ollama, `gen_local.py`)
Only non-factual content: small talk, markdown formatting of user-provided content, summaries of provided text, "I don't know" answers. Facts are never taken from this LLM. License of the generator: EuroLLM-9B-Instruct is Apache-2.0.

### Hand written
- `sft_data/identity*.txt`, `banter.txt`, `persona_conversations.txt`, `my_conversations*.txt`: written by Alex (and edited with ChatGPT help for the persona).

## Filtering
`build_sft_mix.py` removes: "as an AI language model" clichés, mentions of other assistants (ChatGPT, OpenAI ...), repeated sentences, non-English "English" data, direct arithmetic answers (the calc tool must answer those), answers over ~700 tokens.
Benchmark questions (`benchmarks/alexgpt100.jsonl`) are not used for training: identity questions equal to the benchmark are dropped, and the generated data uses other facts, words and numbers.

## Wikipedia attribution
Text from Wikipedia is licensed under CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/). Authors and history are available at the source page linked in each answer.
