"""Strategy 3 test (docs/fable_plan.md): scripted dialogues, live Wikipedia, on the GPU.
For every message it prints: HISTORY on/off (+ why), the tool calls, the topic memory and the answer.
    .venv\\Scripts\\python.exe test_history_gate.py"""
import time

import numpy as np

from alexgpt.chat import SingularityChat, generate_stream

DIALOGUES = {
    "style reset after 3 wiki answers": ["tell me about NATO", "tell me about the Danube", "tell me about Mount Everest",
                                         "What is your name?", "how are you?"],
    "countries -> Africa": ["what countries are in Europe?", "and in Africa?"],
    "pronoun -> topic": ["who is Alesso?", "what songs does he have?", "why?"],
    "castle -> the one in Jibou": ["what is the Wesselenyi castle?", "the one in Jibou"],
    "Romanian follow-up": ["cine a fost Stefan cel Mare?", "ce a facut el?"],
    "new chat: no leak": ["what songs does he have?"],
}

model = SingularityChat("runs/singularity_sft4/final.pt")
rng = np.random.default_rng(0)
for title, messages in DIALOGUES.items():
    print(f"\n==================== {title}")
    history = []                                    # every dialogue is a new chat
    for message in messages:
        time.sleep(2)                               # Wikipedia limits fast requests
        text, calls = "", []
        for step in generate_stream(model, message, 1200, 0.7, rng, history):
            if step["type"] == "tool":
                calls.append(step["call"] + (" [unused]" if step.get("unused") else ""))
            elif step["type"] == "token":
                text += step["char"]
        text = text.replace("<|endoftext|>", "").strip()
        print(f"USER: {message}")
        print(f"   history={'ON ' if model.use_history else 'off'}  topic now={model.topic!r}  tools={calls}")
        print(f"   MODEL ({len(text)} chars): {text[:220]!r}")
        history += [{"role": "user", "text": message}, {"role": "assistant", "text": text}]
