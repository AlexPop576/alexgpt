// First-use popups: one card per model, shown once above the message box (app.js -> showIntro).
// Edit the texts here; no other file needs to change.
//   kind     : one short line under the name
//   what     : one or two sentences, what this model is
//   points   : 2-4 short bullets (what it can do / what to expect)
//   tryText  : a prompt that shows the model at its best (the "Try it" button writes it in the box)

const MODEL_INTROS = {
  Spark: {
    kind: "Bigram · sees 1 character · 9.6K params",
    what: "The simplest possible language model. It only knows which letter tends to follow which letter, so it writes word-shaped noise.",
    points: [
      "It sees only the last character, nothing before it",
      "Learned from TinyStories, letter pairs only",
      "Not a chatbot: it continues your text",
    ],
    tryText: "Once upon a time",
  },
  Nebula: {
    kind: "MLP · sees 8 characters · 47K params",
    what: "A small neural network that looks at the last 8 characters. Short real words start to appear.",
    points: [
      "Each character becomes a vector (embedding)",
      "Turn on Weights in the panel to see the strongest connections",
      "Not a chatbot: it continues your text",
    ],
    tryText: "The little dog ",
  },
  Pulsar: {
    kind: "Transformer · 3 blocks · sees 32 characters",
    what: "The first model here with attention: every character can look back at the ones before it and decide which matter.",
    points: [
      "Same idea as GPT, just tiny and character-level",
      "Open the panel to see where each block looks",
      "Not a chatbot: it continues your text",
    ],
    tryText: "One day, Lily went to the ",
  },
  Galaxy: {
    kind: "Transformer · 8 blocks · 26M params · GPU",
    what: "The first model that chats. It reads word pieces (BPE tokens) and was trained on simple children's dialogues.",
    points: [
      "Talks like a friendly kid, with a small vocabulary",
      "Knows almost no facts about the world",
      "The panel shows all 8 blocks for every token",
    ],
    tryText: "Hi! What do you like to play?",
  },
  Nova: {
    kind: "Supernova before chat training · 98M params",
    what: "A raw base model: it read about 2 billion tokens of English and Romanian web text, but was never taught to answer.",
    points: [
      "Give it the start of a text and it continues it",
      "Questions get continued, not answered",
      "Compare it with Supernova to see what chat training adds",
    ],
    tryText: "The main reason the sky looks blue is",
  },
  Supernova: {
    kind: "Chat model · 12 blocks · 98M params · EN + RO",
    what: "Nova after supervised fine-tuning (SFT): it learned the question / answer format in English and Romanian.",
    points: [
      "Answers you, in English or Romanian",
      "Small model: it often invents facts",
      "No tools: it answers only from memory",
    ],
    tryText: "Why is the sky blue?",
  },
  "Singularity Base": {
    kind: "Singularity before chat training · 126M params",
    what: "Singularity's base model, 16 blocks: it continues text and does not answer questions.",
    points: [
      "Continues your text, like Nova",
      "Compare the two to see what 4 extra blocks change",
    ],
    tryText: "Capitala României este",
  },
  Singularity: {
    kind: "Chat model · 16 blocks · 126M params · tools",
    what: "The newest model. It can search Wikipedia and use small tools, and it copies facts from the source instead of guessing.",
    points: [
      "Tools: Wikipedia, calculator, units, time, weather, memory",
      "Text copied from Wikipedia is marked, with the source",
      "Says it found no source rather than invent a fact",
    ],
    tryText: "Who wrote Romeo and Juliet?",
  },
};
