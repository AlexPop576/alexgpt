"""Markdown conversations made by CODE (no AI): lists, numbered lists, tables, code blocks, bold terms.
    python gen_data_markdown.py         -> data/sft/gen_markdown.jsonl   (about 2,500 conversations, EN + RO, offline)
Everything is either computed here (sorting, averages, multiplication tables, primes) or copied from the small
fact banks below (capitals, planets, C vs Python ...), so nothing is invented."""
import argparse
import random
import unicodedata

from gen_common import assistant, format_number, user, write_jsonl


def table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def bullets(items):
    return "\n".join(f"- {item}" for item in items)


def numbered(items):
    return "\n".join(f"{number}. {item}" for number, item in enumerate(items, 1))


def plain_key(text):
    return unicodedata.normalize("NFD", text).encode("ascii", "ignore").decode().casefold()


def join_items(items, language):
    word = " and " if language == "en" else " și "
    return ", ".join(items[:-1]) + word + items[-1]


INTRO = {"en": ["Here you go:", "Here it is:", "Here's the result:", ""], "ro": ["Poftim:", "Uite:", "Iată rezultatul:", ""]}


def reply(rng, language, body, after=""):
    intro = rng.choice(INTRO[language])
    return (intro + "\n\n" if intro else "") + body + (("\n\n" + after) if after else "")


# ---------------------------------------------------------------- banks (English, Romanian pairs)
BANKS = {
    "fruits": [("apples", "mere"), ("bananas", "banane"), ("oranges", "portocale"), ("pears", "pere"), ("grapes", "struguri"),
               ("cherries", "cireșe"), ("plums", "prune"), ("strawberries", "căpșuni"), ("lemons", "lămâi"), ("peaches", "piersici")],
    "animals": [("dog", "câine"), ("cat", "pisică"), ("horse", "cal"), ("cow", "vacă"), ("sheep", "oaie"), ("rabbit", "iepure"),
                ("bird", "pasăre"), ("fish", "pește"), ("goat", "capră"), ("duck", "rață")],
    "subjects": [("math", "matematică"), ("physics", "fizică"), ("chemistry", "chimie"), ("biology", "biologie"), ("history", "istorie"),
                 ("geography", "geografie"), ("music", "muzică"), ("art", "desen"), ("informatics", "informatică")],
    "groceries": [("bread", "pâine"), ("milk", "lapte"), ("eggs", "ouă"), ("cheese", "brânză"), ("rice", "orez"), ("butter", "unt"),
                  ("tomatoes", "roșii"), ("potatoes", "cartofi"), ("onions", "ceapă"), ("pasta", "paste")],
}
CAPITALS = [("France", "Franța", "Paris", "Paris"), ("Germany", "Germania", "Berlin", "Berlin"), ("Italy", "Italia", "Rome", "Roma"),
            ("Spain", "Spania", "Madrid", "Madrid"), ("Portugal", "Portugalia", "Lisbon", "Lisabona"), ("Romania", "România", "Bucharest", "București"),
            ("Hungary", "Ungaria", "Budapest", "Budapesta"), ("Poland", "Polonia", "Warsaw", "Varșovia"), ("Austria", "Austria", "Vienna", "Viena"),
            ("Greece", "Grecia", "Athens", "Atena"), ("Ireland", "Irlanda", "Dublin", "Dublin"), ("Norway", "Norvegia", "Oslo", "Oslo"),
            ("Sweden", "Suedia", "Stockholm", "Stockholm"), ("Finland", "Finlanda", "Helsinki", "Helsinki"), ("Denmark", "Danemarca", "Copenhagen", "Copenhaga"),
            ("the Netherlands", "Olanda", "Amsterdam", "Amsterdam"), ("Belgium", "Belgia", "Brussels", "Bruxelles"), ("Switzerland", "Elveția", "Bern", "Berna"),
            ("Czechia", "Cehia", "Prague", "Praga"), ("Bulgaria", "Bulgaria", "Sofia", "Sofia"), ("Japan", "Japonia", "Tokyo", "Tokyo"),
            ("China", "China", "Beijing", "Beijing"), ("India", "India", "New Delhi", "New Delhi"), ("Canada", "Canada", "Ottawa", "Ottawa"),
            ("Brazil", "Brazilia", "Brasília", "Brasília"), ("Argentina", "Argentina", "Buenos Aires", "Buenos Aires"), ("Egypt", "Egipt", "Cairo", "Cairo"),
            ("Turkey", "Turcia", "Ankara", "Ankara"), ("Australia", "Australia", "Canberra", "Canberra"), ("Mexico", "Mexic", "Mexico City", "Ciudad de México")]
PLANETS = {"en": ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"],
           "ro": ["Mercur", "Venus", "Pământ", "Marte", "Jupiter", "Saturn", "Uranus", "Neptun"]}
NAMES = ["Ana", "Mihai", "Ioana", "Andrei", "Elena", "David", "Sofia", "Radu", "Maria", "Tudor", "Bianca", "Vlad", "Irina", "Paul", "Diana", "Luca"]
CITIES = ["Cluj", "Iași", "Brașov", "Sibiu", "Oradea", "Timișoara", "Constanța", "București"]
ORDINAL_EN = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth"]
ORDINAL_RO = ["prima", "a doua", "a treia", "a patra", "a cincea", "a șasea", "a șaptea", "a opta"]


# ---------------------------------------------------------------- generators
def list_from_items(rng, language):
    bank = rng.choice(list(BANKS))
    pairs = rng.sample(BANKS[bank], rng.randint(3, 7))
    items = [pair[0 if language == "en" else 1] for pair in pairs]
    numbered_list = rng.random() < 0.4
    body = numbered(items) if numbered_list else bullets(items)
    if language == "en":
        text = rng.choice([f"make a {'numbered ' if numbered_list else ''}list: {', '.join(items)}", f"put these in a {'numbered ' if numbered_list else 'bullet '}list: {join_items(items, language)}"])
    else:
        text = rng.choice([f"fă o listă{' numerotată' if numbered_list else ''}: {', '.join(items)}", f"pune astea într-o listă {'numerotată' if numbered_list else 'cu buline'}: {join_items(items, language)}"])
    return [user(text), assistant(reply(rng, language, body))]


def sorted_list(rng, language):
    if rng.random() < 0.5:
        numbers = rng.sample(range(1, 100), rng.randint(5, 9))
        descending = rng.random() < 0.4
        result = sorted(numbers, reverse=descending)
        shown = ", ".join(map(str, numbers))
        if language == "en":
            text = f"sort these numbers {'from largest to smallest' if descending else 'in ascending order'}: {shown}"
        else:
            text = f"sortează numerele {'de la cel mai mare la cel mai mic' if descending else 'crescător'}: {shown}"
        return [user(text), assistant(reply(rng, language, numbered(result)))]
    bank = rng.choice(list(BANKS))
    pairs = rng.sample(BANKS[bank], rng.randint(4, 8))
    items = [pair[0 if language == "en" else 1] for pair in pairs]
    result = sorted(items, key=plain_key)
    text = (f"sort alphabetically: {', '.join(items)}" if language == "en" else f"pune în ordine alfabetică: {', '.join(items)}")
    return [user(text), assistant(reply(rng, language, bullets(result)))]


def table_from_records(rng, language):
    count = rng.randint(3, 6)
    people = [(name, rng.randint(18, 45), rng.choice(CITIES)) for name in rng.sample(NAMES, count)]
    given = ", ".join(f"{name} {age} {city}" for name, age, city in people)
    headers = ["Name", "Age", "City"] if language == "en" else ["Nume", "Vârstă", "Oraș"]
    wish = rng.choice(["plain", "plain", "sorted", "average"])
    rows = people
    after = ""
    if wish == "sorted":
        rows = sorted(people, key=lambda person: person[1])
    if wish == "average":
        average = format_number(sum(person[1] for person in people) / count, 1)
        after = f"The average age is {average}." if language == "en" else f"Vârsta medie este {average}."
    if language == "en":
        ask = {"plain": "make a table", "sorted": "make a table sorted by age (youngest first)", "average": "make a table and tell me the average age"}[wish]
        text = f"{ask}: {given}"
    else:
        ask = {"plain": "fă un tabel", "sorted": "fă un tabel sortat după vârstă (cei mai tineri primii)", "average": "fă un tabel și spune-mi vârsta medie"}[wish]
        text = f"{ask}: {given}"
    return [user(text), assistant(reply(rng, language, table(headers, rows), after))]


def math_tables(rng, language):
    kind = rng.choice(["times", "times", "primes", "squares", "powers", "fibonacci"])
    en = language == "en"
    if kind == "times":
        n = rng.randint(2, 12)
        rows = [(f"{n} x {i}", n * i) for i in range(1, 11)]
        text = rng.choice([f"multiplication table of {n}", f"show me the {n} times table"]) if en else rng.choice([f"tabla înmulțirii cu {n}", f"arată-mi tabla lui {n}"])
        return [user(text), assistant(reply(rng, language, table(["Expression", "Result"] if en else ["Operație", "Rezultat"], rows)))]
    if kind == "primes":
        count = rng.randint(5, 15)
        primes = []
        candidate = 2
        while len(primes) < count:
            if all(candidate % p for p in primes if p * p <= candidate):
                primes.append(candidate)
            candidate += 1
        text = f"list the first {count} prime numbers" if en else f"scrie primele {count} numere prime"
        return [user(text), assistant(reply(rng, language, bullets(primes)))]
    if kind == "squares":
        top = rng.randint(5, 12)
        rows = [(i, i * i) for i in range(1, top + 1)]
        text = f"table of squares from 1 to {top}" if en else f"tabel cu pătratele numerelor de la 1 la {top}"
        return [user(text), assistant(reply(rng, language, table(["n", "n²"], rows)))]
    if kind == "powers":
        top = rng.randint(6, 12)
        rows = [(i, 2 ** i) for i in range(0, top + 1)]
        text = f"table of powers of 2 up to 2^{top}" if en else f"tabel cu puterile lui 2 până la 2^{top}"
        return [user(text), assistant(reply(rng, language, table(["n", "2^n"], rows)))]
    count = rng.randint(6, 14)
    numbers = [0, 1]
    while len(numbers) < count:
        numbers.append(numbers[-1] + numbers[-2])
    text = f"first {count} Fibonacci numbers" if en else f"primele {count} numere Fibonacci"
    return [user(text), assistant(reply(rng, language, ", ".join(map(str, numbers[:count])), "Each number is the sum of the two before it." if en else "Fiecare număr este suma celor două dinaintea lui."))]


def capitals(rng, language):
    count = rng.randint(3, 6)
    chosen = rng.sample(CAPITALS, count)
    en = language == "en"
    countries = [c[0] if en else c[1] for c in chosen]
    if rng.random() < 0.5:
        rows = [(c[0] if en else c[1], c[2] if en else c[3]) for c in chosen]
        body = table(["Country", "Capital"] if en else ["Țară", "Capitală"], rows)
        text = (f"capitals of {join_items(countries, language)} in a table" if en else f"capitalele pentru {join_items(countries, language)}, într-un tabel")
    else:
        body = bullets([f"**{c[0] if en else c[1]}**: {c[2] if en else c[3]}" for c in chosen])
        text = (f"what are the capitals of {join_items(countries, language)}?" if en else f"care sunt capitalele pentru {join_items(countries, language)}?")
    return [user(text), assistant(reply(rng, language, body))]


def planets(rng, language):
    en = language == "en"
    names = PLANETS[language]
    if rng.random() < 0.5:
        text = rng.choice(["list the planets in order from the Sun", "planets of the solar system, in order"]) if en else rng.choice(["listează planetele în ordine față de Soare", "planetele sistemului solar, în ordine"])
        return [user(text), assistant(reply(rng, language, numbered(names)))]
    number = rng.randint(1, 8)
    if en:
        return [user(f"which planet is the {ORDINAL_EN[number - 1]} from the Sun?"), assistant(f"{names[number - 1]} is number {number} from the Sun.")]
    return [user(f"care este {ORDINAL_RO[number - 1]} planetă față de Soare?"), assistant(f"{names[number - 1]} este planeta numărul {number} față de Soare.")]


# ---------------------------------------------------------------- fixed comparison tables
COMPARISONS = [
    {"keys": {"en": ["C", "Python"], "ro": ["C", "Python"]},
     "en": (["Feature", "C", "Python"], [("Typing", "Static", "Dynamic"), ("Execution", "Compiled to machine code", "Interpreted (bytecode)"),
                                          ("Memory", "Manual (malloc/free)", "Automatic (garbage collector)"), ("Speed", "Usually faster", "Usually slower"),
                                          ("Typical use", "Systems, embedded, performance", "Scripting, data, AI")]),
     "ro": (["Caracteristică", "C", "Python"], [("Tipare", "Statică", "Dinamică"), ("Execuție", "Compilat în cod mașină", "Interpretat (bytecode)"),
                                                 ("Memorie", "Manuală (malloc/free)", "Automată (garbage collector)"), ("Viteză", "De obicei mai rapid", "De obicei mai lent"),
                                                 ("Utilizare", "Sisteme, embedded, performanță", "Scripturi, date, AI")])},
    {"keys": {"en": ["stack", "queue"], "ro": ["stivă", "coadă"]},
     "en": (["", "Stack", "Queue"], [("Order", "LIFO (last in, first out)", "FIFO (first in, first out)"), ("Add", "push", "enqueue"), ("Remove", "pop", "dequeue"),
                                      ("Example", "Undo history, call stack", "Print queue, waiting line")]),
     "ro": (["", "Stivă", "Coadă"], [("Ordine", "LIFO (ultimul intrat, primul ieșit)", "FIFO (primul intrat, primul ieșit)"), ("Adăugare", "push", "enqueue"),
                                      ("Eliminare", "pop", "dequeue"), ("Exemplu", "Istoric undo, stiva de apeluri", "Coadă de imprimare, coadă la casă")])},
    {"keys": {"en": ["TCP", "UDP"], "ro": ["TCP", "UDP"]},
     "en": (["", "TCP", "UDP"], [("Connection", "Yes (handshake)", "No"), ("Reliability", "Delivery and order guaranteed", "No guarantee"),
                                  ("Speed", "Slower", "Faster"), ("Typical use", "Web, email, file transfer", "Video calls, games, DNS")]),
     "ro": (["", "TCP", "UDP"], [("Conexiune", "Da (handshake)", "Nu"), ("Fiabilitate", "Livrare și ordine garantate", "Fără garanții"),
                                  ("Viteză", "Mai lent", "Mai rapid"), ("Utilizare", "Web, email, transfer de fișiere", "Apeluri video, jocuri, DNS")])},
    {"keys": {"en": ["array", "linked list"], "ro": ["array", "listă înlănțuită"]},
     "en": (["", "Array", "Linked list"], [("Access by index", "O(1)", "O(n)"), ("Insert at the start", "O(n)", "O(1)"),
                                            ("Memory", "Contiguous", "Nodes with pointers, scattered"), ("Size", "Fixed (usually)", "Grows easily")]),
     "ro": (["", "Array", "Listă înlănțuită"], [("Acces după index", "O(1)", "O(n)"), ("Inserare la început", "O(n)", "O(1)"),
                                                  ("Memorie", "Contiguă", "Noduri cu pointeri, împrăștiate"), ("Dimensiune", "Fixă (de obicei)", "Crește ușor")])},
    {"keys": {"en": ["SSD", "HDD"], "ro": ["SSD", "HDD"]},
     "en": (["", "SSD", "HDD"], [("Speed", "Much faster", "Slower"), ("Moving parts", "None", "Spinning disk"), ("Price per GB", "Higher", "Lower"),
                                  ("Noise", "Silent", "Audible"), ("Shock resistance", "Better", "Worse")]),
     "ro": (["", "SSD", "HDD"], [("Viteză", "Mult mai rapid", "Mai lent"), ("Piese în mișcare", "Niciuna", "Disc rotativ"), ("Preț per GB", "Mai mare", "Mai mic"),
                                  ("Zgomot", "Silențios", "Se aude"), ("Rezistență la șocuri", "Mai bună", "Mai slabă")])},
]


def comparison(rng, language):
    item = rng.choice(COMPARISONS)
    first, second = item["keys"][language]
    headers, rows = item[language]
    if language == "en":
        text = rng.choice([f"compare {first} and {second} in a table", f"{first} vs {second}", f"what's the difference between {first} and {second}? use a table"])
    else:
        text = rng.choice([f"compară {first} și {second} într-un tabel", f"{first} vs {second}", f"care e diferența dintre {first} și {second}? folosește un tabel"])
    return [user(text), assistant(reply(rng, language, table(headers, rows)))]


# ---------------------------------------------------------------- fixed how-to steps
STEPS = [
    {"en_q": ["how do I boil an egg?", "steps to boil an egg"], "ro_q": ["cum fierb un ou?", "pașii pentru un ou fiert"],
     "en": ["Put the egg in a pot and cover it with cold water.", "Bring the water to a boil.", "Turn the heat down and cook: about 6-7 minutes for a soft yolk, 9-10 for a hard one.",
            "Move the egg to cold water for a minute, then peel it."],
     "ro": ["Pune oul într-o oală și acoperă-l cu apă rece.", "Adu apa la fierbere.", "Dă focul mai mic și lasă-l: circa 6-7 minute pentru gălbenuș moale, 9-10 pentru ou tare.",
            "Mută oul în apă rece pentru un minut, apoi curăță-l."]},
    {"en_q": ["how do I make tea?", "steps to make tea"], "ro_q": ["cum fac un ceai?", "pașii pentru un ceai"],
     "en": ["Heat the water until it is hot or just boiling.", "Put the tea bag or leaves in a cup.", "Pour the water over it.", "Wait 3-5 minutes, then remove the tea bag."],
     "ro": ["Încălzește apa până fierbe sau aproape.", "Pune plicul sau frunzele de ceai într-o cană.", "Toarnă apa peste ele.", "Așteaptă 3-5 minute, apoi scoate plicul."]},
    {"en_q": ["how does binary search work?", "explain binary search in steps"], "ro_q": ["cum funcționează căutarea binară?", "explică căutarea binară pe pași"],
     "en": ["Start with a **sorted** list and look at the middle element.", "If it equals the target, you are done.", "If the target is smaller, repeat on the left half; if larger, on the right half.",
            "Stop when you find it or the half is empty (not found). Each step halves the list, so it takes O(log n)."],
     "ro": ["Începe cu o listă **sortată** și uită-te la elementul din mijloc.", "Dacă e egal cu ținta, ai terminat.", "Dacă ținta e mai mică, repetă pe jumătatea din stânga; dacă e mai mare, pe cea din dreapta.",
            "Oprește-te când o găsești sau când jumătatea e goală (nu există). Fiecare pas înjumătățește lista, deci durează O(log n)."]},
    {"en_q": ["how should I prepare for an exam?", "give me steps to study for an exam"], "ro_q": ["cum mă pregătesc pentru un examen?", "dă-mi pași pentru a învăța pentru un examen"],
     "en": ["List the topics and see how much time you have.", "Start with the hardest topics while you are fresh.", "Solve past exam problems, not only read the notes.",
            "Sleep well the night before: it helps more than one extra hour of studying."],
     "ro": ["Fă o listă cu subiectele și vezi cât timp ai.", "Începe cu subiectele grele cât ești odihnit.", "Rezolvă subiecte din anii trecuți, nu doar citi cursul.",
            "Dormi bine în noaptea dinainte: ajută mai mult decât încă o oră de învățat."]},
]


def steps(rng, language):
    item = rng.choice(STEPS)
    text = rng.choice(item["en_q" if language == "en" else "ro_q"])
    return [user(text), assistant(reply(rng, language, numbered(item[language])))]


# ---------------------------------------------------------------- bold terms + code blocks
CONCEPTS = [
    {"q": {"en": ["what is a pointer in C?", "explain pointers briefly"], "ro": ["ce este un pointer în C?", "explică pe scurt pointerii"]},
     "a": {"en": "A **pointer** is a variable that stores the memory address of another variable.",
           "ro": "Un **pointer** este o variabilă care ține adresa de memorie a altei variabile."},
     "code": "```c\nint x = 5;\nint *p = &x;        // p holds the address of x\nprintf(\"%d\", *p);   // prints 5\n```"},
    {"q": {"en": ["what is recursion?", "explain recursion with an example"], "ro": ["ce este recursivitatea?", "explică recursivitatea cu un exemplu"]},
     "a": {"en": "**Recursion** is when a function calls itself on a smaller problem until it reaches a **base case**.",
           "ro": "**Recursivitatea** înseamnă că o funcție se apelează pe ea însăși pentru o problemă mai mică, până la un **caz de bază**."},
     "code": "```c\nint factorial(int n) {\n    if (n <= 1) return 1;       // base case\n    return n * factorial(n - 1);\n}\n```"},
    {"q": {"en": ["what is an array?", "explain arrays in C"], "ro": ["ce este un array?", "explică array-urile în C"]},
     "a": {"en": "An **array** stores several values of the same type one after another in memory. You read them by index, starting at 0.",
           "ro": "Un **array** ține mai multe valori de același tip una după alta în memorie. Le citești cu un index, care începe de la 0."},
     "code": "```c\nint a[3] = {10, 20, 30};\nprintf(\"%d\", a[1]);   // prints 20\n```"},
    {"q": {"en": ["what is a for loop?", "show me a for loop"], "ro": ["ce este un for?", "arată-mi un ciclu for"]},
     "a": {"en": "A **for loop** repeats a block of code a set number of times.",
           "ro": "Un **ciclu for** repetă un bloc de cod de un număr stabilit de ori."},
     "code": "```c\nfor (int i = 0; i < 3; i++) {\n    printf(\"%d\\n\", i);   // prints 0, 1, 2\n}\n```"},
    {"q": {"en": ["what is a function?", "explain functions in programming"], "ro": ["ce este o funcție?", "explică funcțiile în programare"]},
     "a": {"en": "A **function** is a named block of code that takes inputs, does something and can return a result.",
           "ro": "O **funcție** este un bloc de cod cu nume care primește date, face ceva și poate returna un rezultat."},
     "code": "```c\nint add(int a, int b) {\n    return a + b;\n}\n```"},
    {"q": {"en": ["what is a git commit?", "how do I make a commit?"], "ro": ["ce este un commit în git?", "cum fac un commit?"]},
     "a": {"en": "A **commit** is a saved snapshot of your project in Git.",
           "ro": "Un **commit** este o fotografie salvată a proiectului tău în Git."},
     "code": "```bash\ngit add .\ngit commit -m \"Fix login bug\"\n```"},
]


def concept(rng, language):
    item = rng.choice(CONCEPTS)
    return [user(rng.choice(item["q"][language])), assistant(item["a"][language] + "\n\n" + item["code"])]


GENERATORS = [(list_from_items, 450), (sorted_list, 400), (table_from_records, 500), (math_tables, 400), (capitals, 300), (planets, 100),
              (comparison, 120), (steps, 70), (concept, 90)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--scale", type=float, default=1.0)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    conversations = []
    for function, count in GENERATORS:
        for _ in range(int(count * args.scale)):
            conversations.append(function(rng, "ro" if rng.random() < 0.4 else "en"))
        print(f"  {function.__name__:18s} {int(count * args.scale)}")
    write_jsonl("gen_markdown", conversations, rng)


if __name__ == "__main__":
    main()
