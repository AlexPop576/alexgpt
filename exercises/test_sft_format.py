"""
Tests your build_example() in alexgpt/sft_format.py:   python exercises/test_sft_format.py
sft_train.py runs the same tests before it starts, so the training uses YOUR function.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from alexgpt.bpe_fast import FastBPE
from alexgpt.sft_format import build_example, chat_prompt


def show(ids, mask, tokenizer):
    for token_id, value in zip(ids, mask):
        print(f"     {value}  {tokenizer.pieces[token_id]!r}")


def run_tests(tokenizer, verbose=True):
    """returns True if build_example is right"""
    cases = [
        [{"role": "user", "content": "Salut!"}, {"role": "assistant", "content": "Bună!"}],
        [{"role": "system", "content": "You are Supernova."},
         {"role": "user", "content": "Hi! How are you?"}, {"role": "assistant", "content": "Great, thanks! And you?"},
         {"role": "user", "content": "Fine."}, {"role": "assistant", "content": "Glad to hear it."}],
    ]
    for number, messages in enumerate(cases, 1):
        try:
            result = build_example(messages, tokenizer)
        except NotImplementedError:
            if verbose:
                print("build_example: not written yet (alexgpt/sft_format.py)")
            return False
        ids, mask = result
        # what it must be: the same text the chat uses, and 1 exactly on the assistant's tokens
        expected_ids = tokenizer.encode(chat_prompt(messages[:-1]) + messages[-1]["content"] + "<|endoftext|>")
        expected_mask = []
        inside = False
        for token_id in expected_ids:
            if token_id == tokenizer.special_ids["<|assistant|>"]:
                expected_mask.append(0)
                inside = True
            elif token_id in (tokenizer.special_ids["<|user|>"], tokenizer.special_ids["<|system|>"]):
                expected_mask.append(0)
                inside = False
            else:
                expected_mask.append(1 if inside else 0)
        if list(ids) == expected_ids and list(mask) == expected_mask:
            if verbose:
                print(f"case {number}: OK ({len(ids)} tokens, {sum(mask)} of them are the assistant's)")
                if number == 1:
                    show(ids, mask, tokenizer)
            continue
        if verbose:
            print(f"case {number}: WRONG")
            if len(ids) != len(mask):
                print(f"   ids has {len(ids)} numbers but mask has {len(mask)}: they must be the same length")
            print("   should be (mask, token):")
            show(expected_ids, expected_mask, tokenizer)
            print("   yours:")
            show(ids, mask, tokenizer)
        return False
    return True


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tokenizer = FastBPE.load(os.path.join(root, "checkpoints", "supernova_tokenizer.json"))
    if run_tests(tokenizer):
        print("\nAll good. Training can start:  python sft_train.py --pilot")
