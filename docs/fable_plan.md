**Main idea:** treat the model as a writer, not a knower. At 126M parameters it can phrase a fact you hand it, but it cannot recall facts or pick between similar ones. Most of the ideas below move the choosing into code and leave the wording to the model.

## 1. Ranked strategies

**1. Abstain instead of answering alone (2–3 h)**
- **Idea:** if a question is factual and no tool returned evidence, the model writes "I couldn't find a source for that" instead of guessing.
- **Why at 126M:** its stored facts are mostly noise ("China", "30 minutes"), so a guess is rarely better than nothing.
- **Risk:** over-refusing chit-chat. Only apply it to questions your factual router already catches.
- **Test:** 30 factual questions with the tool disabled. Wrong answers should drop to near 0, and chit-chat should be unchanged.

**2. Shrink the evidence, then rank candidates (5–6 h)**
- **Idea:** give the model the 1–2 best sentences, not 900 characters. Then extract candidate answers of the expected type from those sentences, score each by the model's log-probability as the continuation of the answer, and boost the winner in decoding.
- **Why at 126M:** "Captain John Taylor" happened because the passage held several names. Ranking 5 candidates is much easier for a small model than generating the right one.
- **Risk:** the best sentence is wrong, so the answer is confidently wrong. Fall back to abstaining when the top sentence scores low.
- **Test:** 40 "who/when/where" questions whose passages contain 3+ names or dates. Measure exact-entity accuracy before and after.

**3. Gate the history (3–4 h)**
- **Idea:** send zero history by default. Include it only when a follow-up detector fires (pronoun, no named entity, starts with "but/and/dar/și"). Keep a small topic memory (last entity, last topic) and append it to the search query in code.
- **Why at 126M:** the model imitates whatever style is in its context, so the reliable fix is to keep that style out of the context.
- **Risk:** a missed follow-up loses context. Log the detector's decisions and review them.
- **Test:** 20 scripted dialogues: 3 long wiki answers, then "What is your name?", plus the "countries → Africa" and "the one in Jibou" cases.

**4. Fix search before the model sees anything (4–5 h)**
- **Idea:** take the top 5 search results and score sentences across all of them. Require every key entity in the question to appear in the chosen passage. If "Napoleon" and "Cluj" never appear together, answer "I found no source saying that". Detect Romanian with a function-word vote (este, care, unde, cine, și/si), not diacritics.
- **Why at 126M:** it cannot notice that an article is off-topic, and it will summarize whatever it gets.
- **Risk:** strict matching rejects good passages that use a synonym or alias. Add Wikipedia redirects as aliases.
- **Test:** 30 questions, half with false premises. Count wrong-article answers.

**5. Wikidata with fixed templates, plus computed tools (8–10 h)**
- **Idea:** no free SPARQL. Map about 25 question patterns to fixed properties (capital, population, borders, head of state, birth date, inventor, coordinates). Pass the model a tiny context like `Romania | borders | Hungary, Serbia…` to verbalize. Add a flight-time tool: coordinates → great-circle distance → divide by ~800 km/h, plus 30 minutes.
- **Why at 126M:** a short fact line leaves no distractors to pick from.
- **Risk:** entity linking picks the wrong "Paris". Prefer the candidate with the most sitelinks.
- **Test:** 50 questions (10 per relation type), including "where is Romania" and "Cluj–Paris".

**6. Read tables and lists from article HTML (6–8 h)**
- **Idea:** for "most/biggest/list of" questions, search for list and ranking pages, parse the first wikitable, and pass the top rows as context. Superlatives are poorly covered in Wikidata, so tables are the better source.
- **Why at 126M:** it only has to turn rows it is handed into a sentence.
- **Risk:** messy tables, and outdated tables. Take the first sortable table only and show the source card.
- **Test:** 20 ranking and discography questions that currently fail.

**7. Context-contrastive decoding (2–3 h, non-obvious)**
- **Idea:** for the first sentence, run two forward passes, one with the evidence and one without. Use `logits = (1+a)·with − a·without`, with `a` around 0.5–1, restricted to the top-k tokens of the "with" pass.
- **Why at 126M:** it cancels the model's own prior (the "China" reflex) and amplifies what the evidence adds. Small models have weak but stubborn priors.
- **Risk:** at high `a`, the text gets strange. Tune on 20 examples.
- **Test:** the same set as strategy 2, run with and without the hard capitalization bans. If accuracy holds, you can relax the bans.

**8. Sample-consistency check (2–3 h, non-obvious)**
- **Idea:** when no tool was used, generate 5 short samples at temperature 1 and compare the main entity or number. If they disagree, abstain or show "unverified".
- **Why at 126M:** token probability is badly calibrated in small models, so a wrong fact can score 90%. Disagreement between samples is a much better signal of a made-up answer.
- **Risk:** consistent wrong answers pass. That is why this comes after strategy 1.
- **Test:** on 50 no-tool answers, compare how well each signal (token probability vs agreement) separates right from wrong.

## 2. Your own ideas, judged

- **Wikidata:** yes, but as templates (strategy 5).
- **Best-of-N with n-gram support:** this has a flaw. "John Taylor" is in the passage, so it passes. Measure support against the one sentence that also matches the question's key words, not the whole passage.
- **Identity self-RAG:** good and cheap. Combine it with zero history.
- **Storing only the model's sentence:** good, but strategy 3 goes further.
- **Confidence badge:** replace token probability with sample agreement (strategy 8).
- **Romanian via language links:** try Romanian Wikipedia first for Romanian questions. Falling back to an English passage with a Romanian answer is cross-lingual work the model probably can't do well.
- **Planner:** skip a general planner. Hard-code 2–3 chains in rules ("how old is X" = birth date → calc).

## 3. The SFT round (about 30–40 min of GPU)

Roughly 20k new examples plus about 6k replayed from your old SFT set to prevent forgetting. Use about one third of the original learning rate, 2 passes.

| Data | Count | Fixes |
|---|---|---|
| Grounded answers with distractors: 1–3 sentences containing several same-type names or dates, where the answer copies the right one. Build from Wikidata triples matched to Wikipedia sentences, with question templates. | ~8k | Wrong name from the right passage |
| Abstention: context that does not contain the answer → "The source doesn't say." Plus factual questions with no context → "I'm not sure without a source." | ~4k | Guessing |
| Fact-line verbalization: `entity \| property \| value` → one sentence, EN and RO | ~3k | Strategies 5 and 6 |
| Style reset: long wiki-style history, then a short identity or chit-chat answer | ~2k | History poisoning |
| Follow-up query writing: history + "but in Cluj?" → the full search query after `<|search|>` | ~3k | Lost topic |

Keep abstention under about 25% of the grounded data, or the model will refuse everything.

Optional DPO: log your best-of-N runs. Use the supported candidate as "chosen" and the unsupported one as "rejected". About 2k pairs is enough, and the data costs you nothing extra.

## 4. Stop doing

- **Stop the "model answers alone" fallback for factual questions.** It produces most of your worst failures, and no amount of decoding tricks will put facts into 126M parameters.
- **Stop asking it for formats it can't do.** Build tables in code from structured data and let the model write the intro sentence. For translation, have it decline honestly.
- **Grow the benchmark.** With 100 questions, a score of 74.7 has roughly ±4 points of noise, so you can't see small gains. Make about 300 questions, split by failure type, and keep a held-out part you never tune on.