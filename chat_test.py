"""Ask a checkpoint the questions in prompts.txt (one per line) and write the answers to runs/chat_test_<label>.txt.
Tool calls are run (calc, wiki); other tools answer 'error: tool not available'.
    python chat_test.py runs/singularity_sft4/final.pt v4
"""
import sys

import torch

checkpoint, label = sys.argv[1], sys.argv[2]
sys.argv = ["eval_alexgpt100.py", "--format", "chat"]
import eval_alexgpt100 as ev                      # reuses generate(), the tools and the prompt format
from alexgpt.bpe_fast import FastBPE
from alexgpt.sft_format import chat_prompt, system_prompt
from alexgpt.supernova import Supernova

device = "cuda" if torch.cuda.is_available() else "cpu"
saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
model = Supernova(**saved["config"])
model.load_state_dict(saved["weights"] if "weights" in saved else saved["model"])
model = model.to(device).eval()
tokenizer = FastBPE.load("checkpoints/supernova_tokenizer.json")
ev.MAX_NEW_TOKENS = 220
lines = [line.strip() for line in open("prompts.txt", encoding="utf-8") if line.strip() and not line.startswith("#")]
out = open(f"runs/chat_test_{label}.txt", "w", encoding="utf-8")
for line in lines:
    messages = [{"role": "system", "content": system_prompt("Singularity")}, {"role": "user", "content": line}]
    calls = []
    for _ in range(3):
        text = ev.generate(model, tokenizer, chat_prompt(messages), "chat", device)
        if text.startswith("<|search|>"):
            call = text[len("<|search|>"):].strip()
            result, who = ev.run_tool_for_eval(call)
            calls.append(f"{call}  ->  {result[:80]}")
            messages += [{"role": "tool_call", "content": call}, {"role": "context", "content": result}]
            continue
        break
    block = f"USER: {line}\n" + "".join(f"  [tool] {c}\n" for c in calls) + f"MODEL: {text}\n"
    print(block, flush=True)
    out.write(block + "\n")
out.close()
