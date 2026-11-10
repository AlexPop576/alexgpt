"""Small patch data for a short extra SFT round:
  1. feelings ("sunt trist", "I'm tired") -> a kind reply, NO tool call (memory is only for explicit facts/requests)
  2. simple code (Python and C) that is checked here by running it (Python) before it is written
    python gen_data_patch.py  ->  data/sft/gen_patch.jsonl
"""
import random

from gen_common import assistant, user, write_jsonl

FEELINGS = [  # (en, ro, reply en, reply ro)
    ("I'm sad today", "sunt trist azi", "I'm sorry you're feeling sad. Do you want to tell me what happened?", "Îmi pare rău că ești trist. Vrei să-mi spui ce s-a întâmplat?"),
    ("I'm so tired", "sunt foarte obosit", "That sounds rough. Did you sleep badly, or was it a long day?", "Sună greu. Ai dormit prost sau a fost o zi lungă?"),
    ("I'm happy today!", "sunt fericit azi!", "That's great to hear! What made your day good?", "Ce bine! Ce ți-a făcut ziua frumoasă?"),
    ("I'm bored", "mă plictisesc", "Want a riddle, a quick quiz or a random fact?", "Vrei o ghicitoare, un mic quiz sau o curiozitate?"),
    ("I'm stressed about my exam", "sunt stresat pentru examen", "That's normal before an exam. When is it, and what subject is left to review?", "E normal înainte de examen. Când e și ce materie mai ai de recapitulat?"),
    ("I feel lonely", "mă simt singur", "I'm sorry you feel that way. Talking helps a bit, I'm here. Is there someone you could call today too?", "Îmi pare rău că te simți așa. Vorbitul ajută puțin, sunt aici. Ai pe cineva pe care l-ai putea suna azi?"),
    ("I'm angry", "sunt nervos", "That's okay, it happens. What made you angry?", "E în regulă, se întâmplă. Ce te-a enervat?"),
    ("I'm hungry", "mi-e foame", "Then it's food time! Something quick, or are you in the mood to cook?", "Atunci e timpul pentru mâncare! Ceva rapid sau ai chef să gătești?"),
    ("I'm excited, I got the job!", "sunt entuziasmat, am primit jobul!", "Congratulations! That's a big deal. When do you start?", "Felicitări! E un lucru mare. Când începi?"),
    ("I'm a bit nervous", "sunt cam emoționat", "That's normal. What's coming up?", "E normal. Ce urmează?"),
    ("I'm cold", "mi-e frig", "A warm drink and a blanket usually help. Is it cold outside too?", "O băutură caldă și o pătură ajută de obicei. E frig și afară?"),
    ("I'm sick", "sunt răcit", "I'm sorry. Rest, drink plenty of water, and see a doctor if it gets worse.", "Îmi pare rău. Odihnește-te, bea multă apă și mergi la doctor dacă se agravează."),
]
PREFIXES = {"en": ["", "hey, ", "hi. ", "honestly, "], "ro": ["", "hei, ", "salut. ", "sincer, "]}
# (request en, request ro, language, code, test)  -- the Python code is executed with its test before being used
CODE = [
    ("write a python function that adds two numbers", "scrie o funcție în Python care adună două numere", "python",
     "def add(first, second):\n    return first + second\n\nprint(add(2, 3))  # 5", "assert add(2, 3) == 5 and add(-1, 1) == 0"),
    ("write a python function that returns the largest number in a list", "scrie o funcție în Python care întoarce cel mai mare număr dintr-o listă", "python",
     "def largest(numbers):\n    best = numbers[0]\n    for number in numbers:\n        if number > best:\n            best = number\n    return best\n\nprint(largest([3, 9, 4]))  # 9",
     "assert largest([3, 9, 4]) == 9 and largest([-5, -2]) == -2"),
    ("write a python function that reverses a string", "scrie o funcție în Python care inversează un șir de caractere", "python",
     "def reverse(text):\n    return text[::-1]\n\nprint(reverse(\"salut\"))  # tulas", "assert reverse('abc') == 'cba'"),
    ("write a python function that checks if a number is even", "scrie o funcție în Python care verifică dacă un număr este par", "python",
     "def is_even(number):\n    return number % 2 == 0\n\nprint(is_even(4))  # True", "assert is_even(4) and not is_even(7)"),
    ("write a python function that computes the factorial of n", "scrie o funcție în Python care calculează factorialul lui n", "python",
     "def factorial(n):\n    result = 1\n    for value in range(2, n + 1):\n        result *= value\n    return result\n\nprint(factorial(5))  # 120",
     "assert factorial(5) == 120 and factorial(0) == 1"),
    ("write a python function that sums a list of numbers", "scrie o funcție în Python care face suma unei liste de numere", "python",
     "def total(numbers):\n    result = 0\n    for number in numbers:\n        result += number\n    return result\n\nprint(total([1, 2, 3]))  # 6",
     "assert total([1, 2, 3]) == 6 and total([]) == 0"),
    ("write a python function that counts the vowels in a word", "scrie o funcție în Python care numără vocalele dintr-un cuvânt", "python",
     "def count_vowels(word):\n    count = 0\n    for letter in word.lower():\n        if letter in \"aeiou\":\n            count += 1\n    return count\n\nprint(count_vowels(\"banana\"))  # 3",
     "assert count_vowels('banana') == 3"),
    ("write a python function that multiplies two numbers", "scrie o funcție în Python care înmulțește două numere", "python",
     "def multiply(first, second):\n    return first * second\n\nprint(multiply(4, 5))  # 20", "assert multiply(4, 5) == 20"),
    ("write a C function that adds two numbers", "scrie o funcție în C care adună două numere", "c",
     "int add(int first, int second) {\n    return first + second;\n}", None),
    ("write a C function that returns the maximum of two numbers", "scrie o funcție în C care întoarce maximul a două numere", "c",
     "int maximum(int first, int second) {\n    if (first > second)\n        return first;\n    return second;\n}", None),
]
CODE_INTRO = {"en": ["Here it is:", "Sure:", "Here's a simple version:"], "ro": ["Iat-o:", "Sigur:", "Uite o variantă simplă:"]}


def main():
    for _, _, language, code, test in CODE:          # check the Python code really works
        if language == "python":
            scope = {}
            exec(code.split("\n\nprint(")[0], scope)
            exec(test, scope)
    rng = random.Random(31)
    conversations = []
    for _ in range(420):
        en, ro, reply_en, reply_ro = rng.choice(FEELINGS)
        language = "ro" if rng.random() < 0.55 else "en"
        text = rng.choice(PREFIXES[language]) + (en if language == "en" else ro)
        conversations.append([user(text), assistant(reply_en if language == "en" else reply_ro)])
    for _ in range(360):
        en, ro, code_language, code, _ = rng.choice(CODE)
        language = "ro" if rng.random() < 0.45 else "en"
        request = en if language == "en" else ro
        if rng.random() < 0.3:
            request = request[0].upper() + request[1:] + "."
        conversations.append([user(request), assistant(rng.choice(CODE_INTRO[language]) + f"\n\n```{code_language}\n{code}\n```")])
    write_jsonl("gen_patch", conversations, rng)


if __name__ == "__main__":
    main()
