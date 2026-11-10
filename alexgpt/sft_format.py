"""
The chat format: how a conversation becomes token ids for SFT, and for chatting later.

    <|system|>You are Supernova.<|user|>Salut!<|assistant|>Bună! Ce faci?<|endoftext|><|user|>Bine.<|assistant|>Mă bucur!<|endoftext|>

  system     <|system|> + text          (optional, only at the start)
  user       <|user|> + text
  assistant  <|assistant|> + text + <|endoftext|>     <- <|endoftext|> = "my reply is over": that's how it learns to STOP

Used by: sft_train.py (training), alexgpt/chat.py (the chat), eval_alexgpt100.py (the benchmark).
"""


# ============================================================ YOUR FUNCTION
def build_example(messages, tokenizer):
    ids = []
    mask = []
    for message in messages:
        role = message["role"]
        text_ids = tokenizer.encode(message["content"], allow_special=False)   # "Salut!" -> [3011, 337, 3]
        # (allow_special=False: a "<|user|>" typed inside the text can't become a real marker)

        if role == "user":
            # the user's marker, then the user's text: nothing here is the assistant's -> all 0
            ids.append(tokenizer.special_ids["<|user|>"])
            mask.append(0)
            for token_id in text_ids:
                ids.append(token_id)
                mask.append(0)

        elif role == "system":
            # exactly like user, only with the "<|system|>" marker
            ids.append(tokenizer.special_ids["<|system|>"])
            mask.append(0)
            for token_id in text_ids:
                ids.append(token_id)
                mask.append(0)

        elif role == "tool_call":
            # the model asks for a tool: <|assistant|><|search|>calc: 245 * 18<|endoftext|>
            # the app runs the tool; the model writes the call (mask 1) and STOPS
            ids.append(tokenizer.special_ids["<|assistant|>"])
            mask.append(0)
            ids.append(tokenizer.special_ids["<|search|>"])
            mask.append(1)
            for token_id in text_ids:
                ids.append(token_id)
                mask.append(1)
            ids.append(tokenizer.end_id)
            mask.append(1)

        elif role == "context":
            # the tool's result, put there by the app (never written by the model): <|context|>4410
            ids.append(tokenizer.special_ids["<|context|>"])
            mask.append(0)
            for token_id in text_ids:
                ids.append(token_id)
                mask.append(0)

        else:   # assistant
            # 1. the "<|assistant|>" marker -> 0 (it is given to the model, not guessed)
            # 2. every token of the text   -> 1 (this is what it must learn to write)
            # 3. tokenizer.end_id at the end -> 1 (it must learn to STOP)
            ids.append(tokenizer.special_ids["<|assistant|>"])
            mask.append(0)
            for token_id in text_ids:
                ids.append(token_id)
                mask.append(1)
            ids.append(tokenizer.end_id)
            mask.append(1)
            
    return ids, mask


# ============================================================ ready-made helpers (Claude)
# The system prompt: the first thing the model reads in every chat. It puts the model's name and creator
# right in front of it, so it can COPY them (attention) instead of having to remember them.
# Used by: alexgpt/chat.py (every chat), build_sft_mix.py (part of the training data), eval_alexgpt100.py.
SYSTEM_PROMPT = ("You are {NAME}, a small AI language model from the CosmosLM family, built from scratch by Alex, "
                 "a computer science student in Cluj-Napoca. You are an AI assistant, not a person. "
                 "Be friendly, helpful and brief. Answer in the language of the user (English or Romanian).")


def system_prompt(name):
    """-> the system prompt for the model called `name` ("Supernova", "Singularity")"""
    return SYSTEM_PROMPT.replace("{NAME}", name)


def chat_prompt(messages):
    """The text the model gets when it must REPLY: the conversation so far + <|assistant|>.
    messages = everything up to and including the user's last message."""
    text = ""
    for message in messages:
        if message["role"] == "system":
            text += "<|system|>" + message["content"]
        elif message["role"] == "user":
            text += "<|user|>" + message["content"]
        elif message["role"] == "tool_call":
            text += "<|assistant|><|search|>" + message["content"] + "<|endoftext|>"
        elif message["role"] == "context":
            text += "<|context|>" + message["content"]
        else:
            text += "<|assistant|>" + message["content"] + "<|endoftext|>"
    return text + "<|assistant|>"
