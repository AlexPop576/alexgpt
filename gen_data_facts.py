"""Short factual questions -> the model calls wiki, reads the intro, answers in one sentence.
Needs internet (Wikipedia). Every fact is verified: the answer must literally appear in the fetched intro,
otherwise the fact is dropped. The questions here are NOT the ones from benchmarks/alexgpt100.jsonl.
    python gen_data_facts.py      ->  data/wiki_cache_facts.jsonl (cache)  +  data/sft/gen_facts.jsonl
"""
import json
import os
import random
import re
import time
import urllib.error
import urllib.parse

from gen_common import assistant, context, tool_call, user, write_jsonl
from gen_data_wiki import clean_and_cut, fetch_intro

CACHE = "data/wiki_cache_facts.jsonl"
# (EN country, EN capital, RO country article, RO genitive, RO capital as written in the intro)
CAPITALS = [("Italy", "Rome", "Italia", "Italiei", "Roma"), ("Spain", "Madrid", "Spania", "Spaniei", "Madrid"),
            ("Germany", "Berlin", "Germania", "Germaniei", "Berlin"), ("Japan", "Tokyo", "Japonia", "Japoniei", "Tokyo"),
            ("Poland", "Warsaw", "Polonia", "Poloniei", "Varșovia"), ("Portugal", "Lisbon", "Portugalia", "Portugaliei", "Lisabona"),
            ("Greece", "Athens", "Grecia", "Greciei", "Atena"), ("Hungary", "Budapest", "Ungaria", "Ungariei", "Budapesta"),
            ("Egypt", "Cairo", "Egipt", "Egiptului", "Cairo"), ("Canada", "Ottawa", "Canada", "Canadei", "Ottawa"),
            ("Australia", "Canberra", "Australia", "Australiei", "Canberra"), ("Brazil", "Brasília", "Brazilia", "Braziliei", "Brasília"),
            ("Austria", "Vienna", "Austria", "Austriei", "Viena"), ("Norway", "Oslo", "Norvegia", "Norvegiei", "Oslo"),
            ("Turkey", "Ankara", "Turcia", "Turciei", "Ankara"), ("Bulgaria", "Sofia", "Bulgaria", "Bulgariei", "Sofia"),
            ("Ukraine", "Kyiv", "Ucraina", "Ucrainei", "Kiev"), ("Switzerland", "Bern", "Elveția", "Elveției", "Berna"),
            ("Sweden", "Stockholm", "Suedia", "Suediei", "Stockholm"), ("Ireland", "Dublin", "Irlanda", "Irlandei", "Dublin"),
            ("Czech Republic", "Prague", "Cehia", "Cehiei", "Praga"), ("Serbia", "Belgrade", "Serbia", "Serbiei", "Belgrad"),
            ("Argentina", "Buenos Aires", "Argentina", "Argentinei", "Buenos Aires"), ("Mexico", "Mexico City", "Mexic", "Mexicului", "Ciudad de México"),
            ("India", "New Delhi", "India", "Indiei", "New Delhi"), ("China", "Beijing", "China", "Chinei", "Beijing")]
# (EN question, EN title, EN answer, EN sentence,  RO question, RO title, RO answer, RO sentence)
FACTS = [
    ("Who wrote Hamlet?", "Hamlet", "Shakespeare", "Hamlet was written by William Shakespeare.",
     "Cine a scris Hamlet?", "Hamlet", "Shakespeare", "Hamlet a fost scrisă de William Shakespeare."),
    ("Who wrote Don Quixote?", "Don Quixote", "Cervantes", "Don Quixote was written by Miguel de Cervantes.",
     "Cine a scris Don Quijote?", "Don Quijote", "Cervantes", "Don Quijote a fost scris de Miguel de Cervantes."),
    ("Who wrote War and Peace?", "War and Peace", "Tolstoy", "War and Peace was written by Leo Tolstoy.",
     "Cine a scris Război și pace?", "Război și pace", "Tolstoi", "Război și pace a fost scris de Lev Tolstoi."),
    ("Who wrote Pride and Prejudice?", "Pride and Prejudice", "Jane Austen", "Pride and Prejudice was written by Jane Austen.",
     "Cine a scris Mândrie și prejudecată?", "Mândrie și prejudecată", "Jane Austen", "Mândrie și prejudecată a fost scris de Jane Austen."),
    ("Who wrote The Divine Comedy?", "Divine Comedy", "Dante", "The Divine Comedy was written by Dante Alighieri.",
     "Cine a scris Divina Comedie?", "Divina Comedie", "Dante", "Divina Comedie a fost scrisă de Dante Alighieri."),
    ("Who wrote Nineteen Eighty-Four?", "Nineteen Eighty-Four", "Orwell", "Nineteen Eighty-Four was written by George Orwell.",
     "Cine a scris romanul 1984?", "O mie nouă sute optzeci și patru", "Orwell", "Romanul 1984 a fost scris de George Orwell."),
    ("Who painted the Mona Lisa?", "Mona Lisa", "Leonardo da Vinci", "The Mona Lisa was painted by Leonardo da Vinci.",
     "Cine a pictat Mona Lisa?", "Mona Lisa", "Leonardo da Vinci", "Mona Lisa a fost pictată de Leonardo da Vinci."),
    ("Who developed the theory of relativity?", "Theory of relativity", "Einstein", "The theory of relativity was developed by Albert Einstein.",
     "Cine a formulat teoria relativității?", "Teoria relativității", "Einstein", "Teoria relativității a fost formulată de Albert Einstein."),
    ("Who was the first person to walk on the Moon?", "Neil Armstrong", "first person", "Neil Armstrong was the first person to walk on the Moon.",
     "Cine a fost primul om pe Lună?", "Neil Armstrong", "Armstrong", "Primul om pe Lună a fost Neil Armstrong."),
    ("Who invented the telephone?", "Alexander Graham Bell", "telephone", "The telephone is credited to Alexander Graham Bell.",
     "Cine a inventat telefonul?", "Alexander Graham Bell", "telefon", "Inventarea telefonului i se atribuie lui Alexander Graham Bell."),
    ("What is the tallest mountain above sea level?", "Mount Everest", "Everest", "Mount Everest is the highest mountain above sea level.",
     "Care este cel mai înalt munte de pe Pământ?", "Everest", "Everest", "Muntele Everest este cel mai înalt munte de pe Pământ."),
    ("What is the largest ocean?", "Pacific Ocean", "largest", "The Pacific Ocean is the largest ocean on Earth.",
     "Care este cel mai mare ocean?", "Oceanul Pacific", "mare", "Oceanul Pacific este cel mai mare ocean de pe Pământ."),
    ("Who wrote the novel Ion?", "Ion (novel)", "Rebreanu", "Ion is a novel by Liviu Rebreanu.",
     "Cine a scris romanul Ion?", "Ion (roman)", "Rebreanu", "Romanul Ion a fost scris de Liviu Rebreanu."),
    ("Who wrote Childhood Memories (Amintiri din copilărie)?", "Amintiri din copilărie", "Creangă", "Amintiri din copilărie was written by Ion Creangă.",
     "Cine a scris Amintiri din copilărie?", "Amintiri din copilărie", "Creangă", "Amintiri din copilărie a fost scrisă de Ion Creangă."),
    ("Who was Nikola Tesla?", "Nikola Tesla", "inventor", "Nikola Tesla was an inventor and electrical engineer.",
     "Cine a fost Nikola Tesla?", "Nikola Tesla", "inventator", "Nikola Tesla a fost inventator și inginer electrician."),
    ("What is the largest hot desert?", "Sahara", "Sahara", "The largest hot desert is the Sahara.",
     "Care este cel mai mare deșert fierbinte?", "Sahara", "Sahara", "Cel mai mare deșert fierbinte este Sahara."),
]

# more facts: (question, Wikipedia title, text that must appear in the intro, one-sentence answer)
FACTS_EN = [
    ("What is the chemical symbol for silver?", "Silver", "Ag", "The chemical symbol for silver is Ag."),
    ("What is the chemical symbol for iron?", "Iron", "Fe", "The chemical symbol for iron is Fe."),
    ("What is the chemical symbol for sodium?", "Sodium", "Na", "The chemical symbol for sodium is Na."),
    ("What is the chemical symbol for copper?", "Copper", "Cu", "The chemical symbol for copper is Cu."),
    ("What is the chemical symbol for lead?", "Lead", "Pb", "The chemical symbol for lead is Pb."),
    ("What is the chemical symbol for potassium?", "Potassium", "K", "The chemical symbol for potassium is K."),
    ("Which planet is known as the Red Planet?", "Mars", "Red Planet", "Mars is known as the Red Planet."),
    ("Which planet has the most famous rings?", "Saturn", "rings", "Saturn, known for its rings."),
    ("What is the smallest planet in the Solar System?", "Mercury (planet)", "smallest", "Mercury is the smallest planet in the Solar System."),
    ("What is the largest animal on Earth?", "Blue whale", "largest", "The blue whale is the largest animal on Earth."),
    ("What is the fastest land animal?", "Cheetah", "fastest", "The cheetah is the fastest land animal."),
    ("What is the tallest animal?", "Giraffe", "tallest", "The giraffe is the tallest animal."),
    ("What is the largest island in the world?", "Greenland", "largest", "Greenland is the largest island in the world."),
    ("What is the smallest country in the world?", "Vatican City", "smallest", "Vatican City is the smallest country in the world."),
    ("What is the largest country in the world by area?", "Russia", "largest", "Russia is the largest country in the world by area."),
    ("Who was the first president of the United States?", "George Washington", "first president", "George Washington was the first president of the United States."),
    ("Who painted The Starry Night?", "The Starry Night", "Van Gogh", "The Starry Night was painted by Vincent van Gogh."),
    ("Who wrote the Odyssey?", "Odyssey", "Homer", "The Odyssey is attributed to Homer."),
    ("Who wrote the Harry Potter books?", "Harry Potter", "Rowling", "The Harry Potter books were written by J. K. Rowling."),
    ("Who directed Titanic?", "Titanic (1997 film)", "James Cameron", "Titanic was directed by James Cameron."),
    ("Who discovered penicillin?", "Alexander Fleming", "penicillin", "Penicillin was discovered by Alexander Fleming."),
    ("Who founded Microsoft?", "Microsoft", "Bill Gates", "Microsoft was founded by Bill Gates and Paul Allen."),
    ("Who created Linux?", "Linux", "Linus Torvalds", "Linux was created by Linus Torvalds."),
    ("Who invented the World Wide Web?", "World Wide Web", "Berners-Lee", "The World Wide Web was invented by Tim Berners-Lee."),
    ("Who created the Python programming language?", "Python (programming language)", "Guido van Rossum", "Python was created by Guido van Rossum."),
    ("Who created the C programming language?", "C (programming language)", "Dennis Ritchie", "C was created by Dennis Ritchie."),
    ("What is the speed of light?", "Speed of light", "299,792,458", "The speed of light is 299,792,458 metres per second."),
    ("What is the largest organ of the human body?", "Skin", "largest", "The skin is the largest organ of the human body."),
    ("How many bones does an adult human have?", "Human skeleton", "206", "An adult human has 206 bones."),
    ("What does CPU stand for?", "Central processing unit", "central processing unit", "CPU stands for central processing unit."),
    ("What does HTML stand for?", "HTML", "HyperText Markup Language", "HTML stands for HyperText Markup Language."),
    ("What does SQL stand for?", "SQL", "Structured Query Language", "SQL stands for Structured Query Language."),
    ("What does USB stand for?", "USB", "Universal Serial Bus", "USB stands for Universal Serial Bus."),
    ("What does RAM stand for?", "Random-access memory", "random-access memory", "RAM stands for random-access memory."),
    ("What is the capital of Moldova?", "Moldova", "Chișinău", "The capital of Moldova is Chișinău."),
    ("Who was the first king of Romania?", "Carol I of Romania", "first", "The first king of Romania was Carol I."),
    ("Who was Stephen the Great?", "Stephen the Great", "Moldavia", "Stephen the Great was a ruler of Moldavia."),
    ("Who was Albert Einstein?", "Albert Einstein", "physicist", "Albert Einstein was a theoretical physicist."),
    ("Who was Isaac Newton?", "Isaac Newton", "mathematician", "Isaac Newton was an English mathematician, physicist and astronomer."),
    ("Who was Marie Curie?", "Marie Curie", "physicist", "Marie Curie was a physicist and chemist who researched radioactivity."),
]
FACTS_RO = [
    ("Care este simbolul chimic al argintului?", "Argint", "Ag", "Simbolul chimic al argintului este Ag."),
    ("Care este simbolul chimic al fierului?", "Fier", "Fe", "Simbolul chimic al fierului este Fe."),
    ("Care este simbolul chimic al sodiului?", "Sodiu", "Na", "Simbolul chimic al sodiului este Na."),
    ("Care este simbolul chimic al cuprului?", "Cupru", "Cu", "Simbolul chimic al cuprului este Cu."),
    ("Care este simbolul chimic al plumbului?", "Plumb", "Pb", "Simbolul chimic al plumbului este Pb."),
    ("Care planetă este numită Planeta Roșie?", "Marte (planetă)", "roșie", "Marte este numită Planeta Roșie."),
    ("Care este cea mai mare insulă din lume?", "Groenlanda", "cea mai mare", "Groenlanda este cea mai mare insulă din lume."),
    ("Care este cea mai mică țară din lume?", "Vatican", "cel mai mic", "Vaticanul este cel mai mic stat din lume."),
    ("Care este cea mai mare țară din lume ca suprafață?", "Rusia", "cea mai mare", "Rusia este cea mai mare țară din lume ca suprafață."),
    ("Cine a fost primul președinte al Statelor Unite?", "George Washington", "primul", "George Washington a fost primul președinte al Statelor Unite."),
    ("Cine a scris Odiseea?", "Odiseea", "Homer", "Odiseea îi este atribuită lui Homer."),
    ("Cine a creat limbajul de programare Python?", "Python (limbaj de programare)", "Guido van Rossum", "Python a fost creat de Guido van Rossum."),
    ("Cine a creat Linux?", "Linux", "Linus Torvalds", "Linux a fost creat de Linus Torvalds."),
    ("Cine a fondat Microsoft?", "Microsoft", "Bill Gates", "Microsoft a fost fondat de Bill Gates și Paul Allen."),
    ("Care este viteza luminii?", "Viteza luminii", "299", "Viteza luminii în vid este de aproximativ 299.792.458 de metri pe secundă."),
    ("Care este capitala Republicii Moldova?", "Republica Moldova", "Chișinău", "Capitala Republicii Moldova este Chișinău."),
    ("Cine a fost primul rege al României?", "Carol I al României", "primul", "Primul rege al României a fost Carol I."),
    ("Cine a fost Ștefan cel Mare?", "Ștefan cel Mare", "Moldovei", "Ștefan cel Mare a fost domnitor al Moldovei."),
    ("Cine a fost Albert Einstein?", "Albert Einstein", "fizician", "Albert Einstein a fost un fizician teoretician."),
    ("Cine a fost Nicolae Iorga?", "Nicolae Iorga", "istoric", "Nicolae Iorga a fost un istoric și om politic român."),
    ("Ce înseamnă CPU?", "Unitate centrală de procesare", "procesare", "CPU înseamnă unitate centrală de procesare."),
    ("Ce înseamnă HTML?", "HTML", "HyperText Markup Language", "HTML înseamnă HyperText Markup Language."),
]
# made-up names: the wiki tool finds nothing -> the model says so, it does not invent
SYLLABLES = ["bra", "vel", "kor", "zin", "mar", "tul", "pex", "dor", "lin", "qua", "sol", "nev", "rik", "tal", "vun", "hel", "fos", "gri", "mon", "zar"]
NOT_FOUND_QUESTIONS = {"en": ["Who is {n}?", "What is {n}?", "Tell me about {n}", "Who was {n}?"],
                       "ro": ["Cine este {n}?", "Ce este {n}?", "Spune-mi despre {n}", "Cine a fost {n}?"]}
NOT_FOUND_ANSWERS = {"en": ["I couldn't find anything about {n}, so I won't make something up. Could you check the spelling or tell me more?",
                            "Wikipedia has nothing on {n}. I don't know what it is, and I'd rather not guess.",
                            "I found nothing about {n}. I don't know, and I don't want to invent an answer."],
                     "ro": ["Nu am găsit nimic despre {n}, așa că nu inventez. Poți verifica scrierea sau să-mi spui mai multe?",
                            "Pe Wikipedia nu există nimic despre {n}. Nu știu ce este și prefer să nu ghicesc.",
                            "Nu am găsit nimic despre {n}. Nu știu și nu vreau să inventez un răspuns."]}


def made_up_name(rng):
    name = "".join(rng.choice(SYLLABLES) for _ in range(rng.choice([2, 3])))
    return name.capitalize() + (" " + rng.choice(["Delmar", "Voss", "Ortega", "Petrescu", "Hartmann"]) if rng.random() < 0.4 else "")
QUESTIONS_CAPITAL = {"en": ["What is the capital of {c}?", "capital of {c}?", "Which city is the capital of {c}?", "what's the capital of {c}"],
                     "ro": ["Care este capitala {c}?", "capitala {c}?", "Ce oraș este capitala {c}?", "care e capitala {c}"]}


def norm(text):
    return re.sub(r"[^a-z0-9ăâîșțéíóúáç ]", "", text.lower().replace("ş", "ș").replace("ţ", "ț"))


def get(cache, language, title):
    key = f"{language}|{title}"
    if key in cache:
        return cache[key]
    found = None
    for attempt in range(6):
        try:
            found = fetch_intro(language, title)
            break
        except urllib.error.HTTPError as error:
            time.sleep(10 * (attempt + 1))
        except Exception as error:
            print(f"  {key}: {type(error).__name__}")
            break
    item = None
    if found:
        real_title, text = found
        passage = clean_and_cut(text, limit=900)
        item = {"title": real_title, "passage": passage,
                "url": f"https://{language}.wikipedia.org/wiki/{urllib.parse.quote(real_title.replace(' ', '_'))}"}
    cache[key] = item
    with open(CACHE, "a", encoding="utf-8") as file:
        file.write(json.dumps({"key": key, "item": item}, ensure_ascii=False) + "\n")
    time.sleep(1.5)
    return item


def main():
    rng = random.Random(11)
    cache = {}
    if os.path.exists(CACHE):
        for line in open(CACHE, encoding="utf-8"):
            row = json.loads(line)
            cache[row["key"]] = row["item"]
    jobs = []     # (language, question list, title, answer, sentence)
    for en_country, en_capital, ro_country, ro_genitive, ro_capital in CAPITALS:
        jobs.append(("en", [q.format(c=en_country) for q in QUESTIONS_CAPITAL["en"]], en_country, en_capital,
                     f"The capital of {en_country} is {en_capital}."))
        jobs.append(("ro", [q.format(c=ro_genitive) for q in QUESTIONS_CAPITAL["ro"]], ro_country, ro_capital,
                     f"Capitala {ro_genitive} este {ro_capital}."))
    for en_q, en_t, en_a, en_s, ro_q, ro_t, ro_a, ro_s in FACTS:
        jobs.append(("en", [en_q, en_q.lower().rstrip("?")], en_t, en_a, en_s))
        jobs.append(("ro", [ro_q, ro_q.lower().rstrip("?")], ro_t, ro_a, ro_s))
    for question, title, key, sentence in FACTS_EN:
        jobs.append(("en", [question, question.lower().rstrip("?")], title, key, sentence))
    for question, title, key, sentence in FACTS_RO:
        jobs.append(("ro", [question, question.lower().rstrip("?")], title, key, sentence))
    conversations, dropped = [], []
    for language, questions, title, answer, sentence in jobs:
        item = get(cache, language, title)
        if not item or norm(answer) not in norm(item["passage"]):
            dropped.append(f"{language}:{title}")
            continue
        for question in questions[:3]:
            result = f"{item['title']} — {item['passage']} ({item['url']})"
            reply = sentence + ("" if rng.random() < 0.5 else ("\n\nSource: " if language == "en" else "\n\nSursă: ") + item["url"])
            conversations.append([user(question), tool_call(f"wiki: {item['title']}"), context(result), assistant(reply)])
    for _ in range(260):          # unknown names: wiki finds nothing
        language = "ro" if rng.random() < 0.45 else "en"
        name = made_up_name(rng)
        question = rng.choice(NOT_FOUND_QUESTIONS[language]).format(n=name)
        conversations.append([user(question), tool_call(f"wiki: {name}"), context(f"error: nothing found for '{name}'"),
                              assistant(rng.choice(NOT_FOUND_ANSWERS[language]).format(n=name))])
    print(f"{len(conversations)} conversations; dropped (answer not in the intro): {dropped}")
    write_jsonl("gen_facts", conversations, rng)


if __name__ == "__main__":
    main()
