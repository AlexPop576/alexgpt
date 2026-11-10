"""Replays questions from a real chat with Singularity:  python test_routing_chat.py"""
import time
import numpy as np
from alexgpt.chat import SingularityChat, generate_stream

model = SingularityChat("runs/singularity_sft4/final.pt")
rng = np.random.default_rng(0)
history = []
for question in ["what s the time in cluj", "and the hour?", "ce ora e in Paris", "who are some famous romanians?", "create a table with romanian foods"]:
    time.sleep(3)                                   # Wikipedia limits fast requests
    text, used = "", []
    for step in generate_stream(model, question, 2000, 0.7, rng, history):
        if step["type"] == "tool_start":
            used.append("START " + step["call"])
        elif step["type"] == "tool":
            used.append(step["call"] + " -> " + step["result"][:150] + (" [UNUSED]" if step.get("unused") else ""))
        elif step["type"] == "source":
            used.append("SOURCE " + step["title"] + " " + step["url"])
        elif step["type"] == "token":
            text += ("[[" + step["char"] + "]]") if step.get("grounded") else step["char"]
    text = text.replace("<|endoftext|>", "").strip()
    print("USER:", question)
    for line in used:
        print("   [tool]", line)
    print("MODEL:", text[:1400], "\n")
    history += [{"role": "user", "text": question}, {"role": "assistant", "text": text}]
