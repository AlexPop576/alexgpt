"""Checks the Singularity model of the web interface without a browser:  python test_singularity_ui.py"""
import numpy as np
from alexgpt.chat import SingularityChat, generate_stream

model = SingularityChat("runs/singularity_sft4/final.pt")
rng = np.random.default_rng(0)
history = []
for question in ["salut! cine ești?", "cât face 2+6?", "who invented the telephone?", "spune-mi despre Cluj-Napoca", "ține minte că mă cheamă Alex", "sunt trist azi", "care este capitala Poloniei?", "scrie o funcție în python care adună două numere"]:
    text, tools_used = "", []
    for step in generate_stream(model, question, 2000, 0.7, rng, history):
        if step["type"] == "tool":
            tools_used.append(step["call"] + " -> " + step["result"][:70])
        elif step["type"] == "token":
            text += step["char"]
    text = text.replace("<|endoftext|>", "").strip()
    print("USER:", question)
    for line in tools_used:
        print("   [tool]", line)
    print("MODEL:", text[:300], "\n")
    history += [{"role": "user", "text": question}, {"role": "assistant", "text": text}]
