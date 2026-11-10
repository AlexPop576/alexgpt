// Explore mode: every text shown in the info panel and in the "How it was built" timeline.
// Edit only the strings. Each stage has:
//   title : the name shown in the panel
//   short : one sentence, shown first (what it does)
//   long  : 2-5 sentences, shown under "More" (how it works). DRAFT = written by Claude, Alex rewrites.
//   code  : where it lives in the project (shown small, so people can find it on GitHub)

const EXPLORE_TEXT = {
  router: {
    title: "Tool router",
    short: "DRAFT: Before the model reads anything, plain code checks whether the question needs a tool.",
    long: "DRAFT: The checks run in order: a follow-up like \"and in Cluj?\", fixed rules (calculator, units, time, weather, memory), then questions like who / what / when, which go to Wikipedia. If no rule fires, the model itself may decide to write <|search|>. A small model is bad at knowing facts, so code decides when to look them up.",
    code: "alexgpt/tool_router.py · route_followup → route → route_question",
  },
  tools: {
    title: "Tools",
    short: "DRAFT: Small programs for things a language model is bad at: math, units, time, weather, dice, memory.",
    long: "DRAFT: Plain code computes the result (17*23 = 391 is always right), then it is given to the model as a <|context|> message. The model only puts the result into a sentence. A follow-up like \"and in Cluj?\" repeats the last tool with the new place.",
    code: "alexgpt/tools.py · calc, convert, now, weather, dice, memory",
  },
  identity: {
    title: "Identity answers",
    short: "DRAFT: Questions about the model itself (\"who made you?\") are answered from a small file of true facts.",
    long: "DRAFT: The question is compared with the questions in sft_data/identity_qa.txt. If one is close enough, its answer is used, so the model does not invent things about itself.",
    code: "alexgpt/identity_rag.py",
  },
  search: {
    title: "The model asks for a search",
    short: "DRAFT: If no rule fired, the model can write the special token <|search|> followed by a query.",
    long: "DRAFT: It learned this during chat training (SFT): examples where the answer needed a fact started with <|search|>wiki: ... . The code sees that token, runs the search and puts the result back as <|context|>.",
    code: "alexgpt/chat.py · run_tools_inner",
  },
  wikipedia: {
    title: "Wikipedia retrieval",
    short: "DRAFT: One request finds the best article and its text; the sentences that match the question are kept.",
    long: "DRAFT: The top 5 search results are checked: the names in the question must appear in the article. Sentences are scored by the question's words and by what kind of answer is expected (\"when\" wants a year, \"how long\" a duration). If nothing relevant is found, the result is marked unused.",
    code: "alexgpt/tools.py · wiki()",
  },
  abstain: {
    title: "No source, no guess",
    short: "DRAFT: A factual question with no evidence gets \"I couldn't find a source\" instead of a made-up answer.",
    long: "DRAFT: Small models invent facts with full confidence. When the search finds nothing relevant, a fixed reply is shown and the model is not asked at all.",
    code: "alexgpt/chat.py · abstain_if_no_evidence",
  },
  tokens: {
    title: "Tokens",
    short: "DRAFT: The text is cut into pieces from a vocabulary of 16,384, each with a number (its id).",
    long: "DRAFT: The tokenizer is byte-pair encoding (BPE), trained on the same text as the model: frequent pieces like \" the\" are one token, rare words are split. Special tokens like <|user|> and <|assistant|> mark who is talking.",
    code: "alexgpt/bpe_fast.py",
  },
  embeddings: {
    title: "Embeddings",
    short: "DRAFT: Every token id becomes a list of 768 numbers, a vector. Similar tokens get similar vectors.",
    long: "DRAFT: It is just a table with one row per token (16,384 × 768), learned during training. Here each vector is drawn as a column of colored cells: only 24 of the 768 numbers are shown.",
    code: "alexgpt/supernova.py · embedding",
  },
  block: {
    title: "Transformer block",
    short: "DRAFT: 16 identical blocks; each one reads the vectors, adds something to them and passes them on.",
    long: "DRAFT: The line running through all blocks is the residual stream: each block only adds a correction. Inside a block, attention lets every token look at earlier tokens, then the MLP thinks about each token alone.",
    code: "alexgpt/supernova.py · Block",
  },
  attention: {
    title: "Attention",
    short: "DRAFT: The newest token decides which earlier tokens matter right now, and copies information from them.",
    long: "DRAFT: Each head scores every earlier token (query · key), turns the scores into percentages with softmax, and mixes their values. The arcs show those percentages: thicker = it looks there more. A token can never look at the future.",
    code: "alexgpt/supernova.py · Attention",
  },
  mlp: {
    title: "MLP",
    short: "DRAFT: A small network applied to each token alone: it is where much of the stored knowledge lives.",
    long: "DRAFT: The vector is made 4× wider, passed through a non-linear function, and made narrow again. Many facts and word patterns the model learned are stored in these weights.",
    code: "alexgpt/supernova.py · MLP",
  },
  logits: {
    title: "Logits",
    short: "DRAFT: The last vector is compared with all 16,384 tokens: one score each, turned into probabilities.",
    long: "DRAFT: The bars are the top guesses. With a Wikipedia result, fact_guard bans names and numbers that are neither in the question nor in the article (red), so the model cannot invent them.",
    code: "alexgpt/chat.py · fact_guard_ids, finish_known_word",
  },
  sampling: {
    title: "Sampling",
    short: "DRAFT: One token is drawn at random, using the probabilities. Then everything repeats for the next token.",
    long: "DRAFT: Temperature changes how bold the draw is: low = almost always the top guess, high = more surprises. The chosen token is added to the end of the text and the model runs again: that loop is why it is called autoregressive.",
    code: "alexgpt/chat.py · generate_core",
  },
  grounded: {
    title: "Grounded sentences",
    short: "DRAFT: After the model's first sentence, sentences are copied word for word from the article.",
    long: "DRAFT: The idea of Round 1: the model writes, the code chooses the facts. Copied text is marked in green with the source link, so you can always check it.",
    code: "alexgpt/chat.py · grounded_sentences",
  },
  answer: {
    title: "The answer",
    short: "DRAFT: The model writes one token at a time; every new token is added to its own input before the next one is chosen.",
    long: "DRAFT: That loop (tokens → blocks → scores → pick → back to the input) runs once per token until the model writes <|endoftext|>. White text is written by the model, green text is copied from Wikipedia.",
    code: "alexgpt/chat.py · generate_stream",
  },
};

// "How it was built": the timeline drawer. DRAFT texts, Alex fixes dates and numbers.
const EXPLORE_HISTORY = [
  { name: "BPE tokenizer", when: "DRAFT", text: "Byte-pair encoding written from scratch: first 2,048 tokens for Galaxy, then 16,384 for Supernova and Singularity." },
  { name: "Spark", when: "DRAFT", text: "A bigram: a table of which letter follows which. 9.6K numbers." },
  { name: "Nebula", when: "DRAFT", text: "An MLP over the last 8 characters, with embeddings." },
  { name: "Pulsar", when: "DRAFT", text: "The first transformer: 3 blocks of attention over 32 characters." },
  { name: "Galaxy", when: "DRAFT", text: "26M parameters on the GPU, BPE tokens, trained on simple dialogues: the first model that chats." },
  { name: "Supernova", when: "DRAFT", text: "98M, LLaMA-style, pre-trained on 2B tokens of English and Romanian, then chat-trained (SFT)." },
  { name: "Singularity: pre-training", when: "DRAFT", text: "126M parameters, 16 blocks, context 1024, started from Supernova's checkpoint." },
  { name: "SFT v1 → v4", when: "DRAFT", text: "Four rounds of chat training; v4 scores 74.7 on AlexGPT-100 with tools." },
  { name: "Tools", when: "DRAFT", text: "Wikipedia, calculator, units, time, weather and memory, called with <|search|>." },
  { name: "Round 1 grounding", when: "DRAFT", text: "No guessing without a source, stricter search, identity answers, a 300-question failure benchmark." },
];
