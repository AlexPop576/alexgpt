"""Strategy 1 test (docs/fable_plan.md): factual questions with NO source must get the fixed "no source" reply,
chit-chat must still be answered by the model.
Wikipedia is switched off here (every wiki call returns an error), so no network is needed.
    .venv\\Scripts\\python.exe test_abstain.py"""
import numpy as np

from alexgpt import tools
from alexgpt.chat import SingularityChat, generate_stream

real_run_tool = tools.run_tool


def run_tool_without_wiki(call):
    if call.partition(":")[0].strip().lower() == "wiki":
        return "error: wikipedia switched off for this test"
    return real_run_tool(call)


tools.run_tool = run_tool_without_wiki

FACTUAL = [
    "what is the most visited country in the world?", "where is Romania?", "who invented the telephone?",
    "when did World War 2 end?", "how long does the Moon take to orbit the Earth?", "what is the capital of Australia?",
    "who wrote Hamlet?", "how tall is Mount Everest?", "which planet is the largest?", "when was Napoleon born?",
    "who painted the Mona Lisa?", "where is the Eiffel Tower?", "how many people live in Japan?",
    "what is the population of Cluj-Napoca?", "who was Albert Einstein?", "tell me about NATO",
    "cine a inventat telefonul?", "unde este Romania?", "care este capitala Frantei?", "cand a murit Stefan cel Mare?",
    "cine a scris Luceafarul?", "ce este fotosinteza?", "cat de inalt este Varful Moldoveanu?", "search for the Danube",
    "which country has the most islands?", "how fast does light travel?", "who discovered penicillin?",
    "when did Romania join the EU?", "what is the longest river in Europe?", "who is the president of France?",
]
CHIT_CHAT = ["hello!", "how are you?", "what is your name?", "salut, ce faci?", "tell me a joke",
             "multumesc!", "what can you do?", "good night"]

model = SingularityChat("runs/singularity_sft4/final.pt")
rng = np.random.default_rng(0)


def ask(question):
    text, abstained = "", False
    for step in generate_stream(model, question, 600, 0.7, rng, []):
        if step["type"] == "token":
            text += step["char"]
            abstained = abstained or bool(step.get("abstain"))
    return text.replace("<|endoftext|>", "").strip(), abstained


factual_ok = 0
print("=== FACTUAL (no source available) ===")
for question in FACTUAL:
    answer, abstained = ask(question)
    factual_ok += abstained
    print(("ABSTAIN " if abstained else "GUESSED ") + question, "->", answer[:120].replace("\n", " "))
chat_ok = 0
print("\n=== CHIT-CHAT (must NOT abstain) ===")
for question in CHIT_CHAT:
    answer, abstained = ask(question)
    chat_ok += not abstained
    print(("ANSWERED " if not abstained else "REFUSED  ") + question, "->", answer[:120].replace("\n", " "))
print(f"\nfactual abstained: {factual_ok}/{len(FACTUAL)}   chit-chat answered: {chat_ok}/{len(CHIT_CHAT)}")
