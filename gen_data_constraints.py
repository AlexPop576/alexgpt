"""Format instructions with answers made by code (always correct): one word, yes/no, bold, lists of exactly N,
heading + bullets, one sentence, and 'answer in English / reply in Romanian'.
The benchmark questions (benchmarks/alexgpt100.jsonl) are not used: other words, other topics.
    python gen_data_constraints.py   ->  data/sft/gen_constraints.jsonl
"""
import random

from gen_common import assistant, user, write_jsonl

CATEGORIES = {   # name en, name ro, items (en, ro)
    "vegetables": ("vegetables", "legume", [("carrot", "morcov"), ("potato", "cartof"), ("onion", "ceapă"), ("tomato", "roșie"), ("cucumber", "castravete"), ("pepper", "ardei"), ("cabbage", "varză")]),
    "fruits2": ("fruits", "fructe", [("apple", "măr"), ("pear", "pară"), ("plum", "prună"), ("grapes", "struguri"), ("lemon", "lămâie"), ("peach", "piersică"), ("strawberry", "căpșună"), ("watermelon", "pepene")]),
    "colors": ("colours", "culori", [("red", "roșu"), ("blue", "albastru"), ("green", "verde"), ("yellow", "galben"), ("purple", "mov"), ("orange", "portocaliu"), ("black", "negru")]),
    "pets": ("pets", "animale de companie", [("dog", "câine"), ("rabbit", "iepure"), ("hamster", "hamster"), ("parrot", "papagal"), ("goldfish", "pește auriu"), ("turtle", "broască țestoasă")]),
    "planets": ("planets", "planete", [("Mercury", "Mercur"), ("Venus", "Venus"), ("Earth", "Pământ"), ("Mars", "Marte"), ("Jupiter", "Jupiter"), ("Saturn", "Saturn"), ("Neptune", "Neptun")]),
    "months": ("months of the year", "luni ale anului", [("January", "ianuarie"), ("March", "martie"), ("May", "mai"), ("July", "iulie"), ("September", "septembrie"), ("November", "noiembrie")]),
    "instruments": ("musical instruments", "instrumente muzicale", [("guitar", "chitară"), ("piano", "pian"), ("violin", "vioară"), ("drums", "tobe"), ("flute", "flaut"), ("trumpet", "trompetă")]),
    "sports": ("sports", "sporturi", [("football", "fotbal"), ("tennis", "tenis"), ("swimming", "înot"), ("basketball", "baschet"), ("running", "alergare"), ("volleyball", "volei")]),
    "languages": ("programming languages", "limbaje de programare", [("C", "C"), ("Python", "Python"), ("Java", "Java"), ("Rust", "Rust"), ("Go", "Go"), ("JavaScript", "JavaScript")]),
}
ONE_WORD = [  # (question en, answer en, question ro, answer ro)
    ("what colour is a banana?", "Yellow", "ce culoare are o banană?", "Galbenă"),
    ("what colour is snow?", "White", "ce culoare are zăpada?", "Albă"),
    ("what colour is a ripe tomato?", "Red", "ce culoare are o roșie coaptă?", "Roșie"),
    ("how many legs does a dog have?", "Four", "câte picioare are un câine?", "Patru"),
    ("how many days are there in a week?", "Seven", "câte zile are o săptămână?", "Șapte"),
    ("what is the opposite of hot?", "Cold", "care este opusul cuvântului cald?", "Rece"),
    ("what is the opposite of big?", "Small", "care este opusul cuvântului mare?", "Mic"),
    ("what do bees make?", "Honey", "ce produc albinele?", "Miere"),
    ("what is frozen water called?", "Ice", "cum se numește apa înghețată?", "Gheață"),
    ("which season comes after winter?", "Spring", "care anotimp vine după iarnă?", "Primăvara"),
    ("what do cows drink?", "Water", "ce beau vacile?", "Apă"),
    ("what is the first month of the year?", "January", "care este prima lună a anului?", "Ianuarie"),
    ("which animal says meow?", "Cat", "care animal face miau?", "Pisica"),
    ("what is the largest planet?", "Jupiter", "care este cea mai mare planetă?", "Jupiter"),
    ("how many wheels does a bicycle have?", "Two", "câte roți are o bicicletă?", "Două"),
    ("what colour is a lemon?", "Yellow", "ce culoare are o lămâie?", "Galbenă"),
    ("what is two plus two?", "Four", "cât face doi plus doi?", "Patru"),
    ("what do we call a baby dog?", "Puppy", "cum se numește puiul de câine?", "Cățel"),
    ("which fruit is famous for keeping doctors away?", "Apple", "care fruct este asociat cu ținerea doctorului departe?", "Mărul"),
    ("what is the capital of Spain?", "Madrid", "care este capitala Spaniei?", "Madrid"),
]
YES_NO = [  # (question en, question ro, answer is yes)
    ("is Paris in France?", "este Parisul în Franța?", True), ("is a whale a fish?", "este balena un pește?", False),
    ("is 7 a prime number?", "este 7 număr prim?", True), ("is ice colder than steam?", "este gheața mai rece decât aburul?", True),
    ("do birds have feathers?", "au păsările pene?", True), ("is Australia in Europe?", "este Australia în Europa?", False),
    ("is gold a metal?", "este aurul un metal?", True), ("is 10 greater than 20?", "este 10 mai mare decât 20?", False),
    ("is a spider an insect?", "este păianjenul o insectă?", False), ("does the Sun rise in the west?", "răsare Soarele la vest?", False),
    ("is the Pacific the largest ocean?", "este Pacificul cel mai mare ocean?", True), ("is a tomato a vegetable in the kitchen?", "este roșia considerată legumă în bucătărie?", True),
    ("is 15 an even number?", "este 15 număr par?", False), ("can fish live in water?", "pot trăi peștii în apă?", True),
    ("is Berlin the capital of Germany?", "este Berlinul capitala Germaniei?", True), ("is the Moon a star?", "este Luna o stea?", False),
    ("is Madrid in Portugal?", "este Madridul în Portugalia?", False), ("is copper a metal?", "este cuprul un metal?", True),
]
BOLD_WORDS = [("welcome", "bun venit"), ("danger", "pericol"), ("important", "esențial"), ("note", "notă"), ("summary", "rezumat"), ("warning", "atenție"),
              ("free", "gratuit"), ("deadline", "termen limit"), ("new", "nou"), ("tip", "sfat"), ("goodbye", "la revedere"), ("thanks", "mulțumesc")]
TITLES = [("Packing list", "Listă de bagaje"), ("Study plan", "Plan de studiu"), ("Weekend ideas", "Idei pentru weekend"), ("Groceries", "Cumpărături"),
          ("Team members", "Membrii echipei"), ("Favourites", "Preferate"), ("To do today", "De făcut azi"), ("Gift ideas", "Idei de cadouri")]
SENTENCES = [("dogs", "câini", "Dogs are loyal animals that love to play and to stay close to people.", "Câinii sunt animale loiale care iubesc să se joace și să stea aproape de oameni."),
             ("the sea", "mare", "The sea is a huge body of salt water that is home to countless animals.", "Marea este o întindere uriașă de apă sărată în care trăiesc nenumărate animale."),
             ("winter", "iarnă", "Winter is the coldest season of the year, with short days and often with snow.", "Iarna este cel mai rece anotimp al anului, cu zile scurte și adesea cu zăpadă."),
             ("books", "cărți", "Books let us learn new things and travel to other worlds without leaving home.", "Cărțile ne lasă să învățăm lucruri noi și să călătorim în alte lumi fără să ieșim din casă."),
             ("the Moon", "Lună", "The Moon is the natural satellite that orbits the Earth.", "Luna este satelitul natural care se învârte în jurul Pământului."),
             ("bread", "pâine", "Bread is a food made mostly from flour, water and yeast.", "Pâinea este un aliment făcut în principal din făină, apă și drojdie."),
             ("rain", "ploaie", "Rain falls when water in the clouds gets too heavy to stay in the air.", "Ploaia cade atunci când apa din nori devine prea grea ca să rămână în aer."),
             ("rivers", "râuri", "Rivers carry fresh water from the mountains to the sea.", "Râurile duc apă dulce de la munte până la mare."),
             ("trees", "copaci", "Trees give us shade, wood and the oxygen we breathe.", "Copacii ne oferă umbră, lemn și oxigenul pe care îl respirăm."),
             ("coffee", "cafea", "Coffee is a hot drink made from roasted beans that many people drink in the morning.", "Cafeaua este o băutură caldă din boabe prăjite pe care mulți oameni o beau dimineața."),
             ("computers", "calculatoare", "Computers follow instructions to process information very quickly.", "Calculatoarele urmează instrucțiuni ca să proceseze informații foarte repede."),
             ("music", "muzică", "Music is the art of arranging sounds in time, and almost everyone enjoys it.", "Muzica este arta de a aranja sunete în timp și aproape oricine o apreciază."),
             ("the Sun", "Soare", "The Sun is the star at the center of our Solar System.", "Soarele este steaua din centrul sistemului nostru solar."),
             ("bees", "albine", "Bees collect nectar from flowers and make honey.", "Albinele strâng nectar din flori și produc miere."),
             ("football", "fotbal", "Football is a team sport in which two sides try to score goals.", "Fotbalul este un sport de echipă în care două echipe încearcă să marcheze goluri."),
             ("the ocean", "ocean", "Oceans cover most of the surface of our planet.", "Oceanele acoperă cea mai mare parte din suprafața planetei noastre.")]
CAPITALS = [("Spain", "Madrid", "Spaniei", "Spania"), ("Germany", "Berlin", "Germaniei", "Germania"), ("Poland", "Warsaw", "Poloniei", "Polonia"),
            ("Portugal", "Lisbon", "Portugaliei", "Portugalia"), ("Greece", "Athens", "Greciei", "Grecia"), ("Austria", "Vienna", "Austriei", "Austria"),
            ("Hungary", "Budapest", "Ungariei", "Ungaria"), ("Norway", "Oslo", "Norvegiei", "Norvegia"), ("Japan", "Tokyo", "Japoniei", "Japonia"),
            ("Canada", "Ottawa", "Canadei", "Canada")]
ROMANIAN_CAPITAL = {"Madrid": "Madrid", "Berlin": "Berlin", "Warsaw": "Varșovia", "Lisbon": "Lisabona", "Athens": "Atena", "Vienna": "Viena",
                    "Budapest": "Budapesta", "Oslo": "Oslo", "Tokyo": "Tokyo", "Ottawa": "Ottawa"}
SMALL_TALK = [("how are you?", "I'm doing well, thanks! How about you?", "cum ești?", "Sunt bine, mulțumesc! Tu cum ești?"),
              ("what is your name?", "My name is Singularity.", "cum te cheamă?", "Mă numesc Singularity."),
              ("good morning!", "Good morning! How can I help?", "bună dimineața!", "Bună dimineața! Cu ce te pot ajuta?")]


def pick_items(rng, items, n):
    chosen = rng.sample(items, n)
    return sorted(chosen, key=items.index)      # keep the natural order (months, planets)


def list_conversation(rng, language):
    key = rng.choice(list(CATEGORIES))
    name_en, name_ro, items = CATEGORIES[key]
    n = rng.randint(2, 5)
    chosen = pick_items(rng, items, n)
    index = 0 if language == "en" else 1
    values = [item[index] for item in chosen]
    numbered = rng.random() < 0.5
    if language == "en":
        text = rng.choice([f"List {n} {name_en} as a numbered list.", f"Give me exactly {n} {name_en}, as a numbered list.", f"Name {n} {name_en}."]) if numbered else \
               rng.choice([f"List {n} {name_en} as a bulleted list.", f"Give me {n} {name_en} in a bulleted list.", f"Write {n} {name_en} as bullet points."])
    else:
        text = rng.choice([f"Dă-mi {n} {name_ro} ca listă numerotată.", f"Enumeră exact {n} {name_ro}, într-o listă numerotată."]) if numbered else \
               rng.choice([f"Enumeră {n} {name_ro} sub formă de listă cu buline.", f"Dă-mi {n} {name_ro} într-o listă cu buline."])
    body = "\n".join(f"{i}. {v}" for i, v in enumerate(values, 1)) if numbered else "\n".join(f"- {v}" for v in values)
    return [user(text), assistant(body)]


def one_word_conversation(rng, language):
    question_en, answer_en, question_ro, answer_ro = rng.choice(ONE_WORD)
    if language == "en":
        return [user(rng.choice([f"Answer with only one word: {question_en}", f"One word only: {question_en}", f"In one word, {question_en}"])), assistant(answer_en)]
    return [user(rng.choice([f"Răspunde cu un singur cuvânt: {question_ro}", f"Un singur cuvânt: {question_ro}", f"Într-un cuvânt, {question_ro}"])), assistant(answer_ro)]


def yes_no_conversation(rng, language):
    question_en, question_ro, yes = rng.choice(YES_NO)
    if language == "en":
        return [user(rng.choice([f"Answer only 'yes' or 'no': {question_en}", f"Yes or no: {question_en}", f"Reply with just yes or no: {question_en}"])), assistant("Yes." if yes else "No.")]
    return [user(rng.choice([f"Răspunde doar cu „da” sau „nu”: {question_ro}", f"Da sau nu: {question_ro}", f"Răspunde cu da sau nu: {question_ro}"])), assistant("Da." if yes else "Nu.")]


def bold_conversation(rng, language):
    word_en, word_ro = rng.choice(BOLD_WORDS)
    italic = rng.random() < 0.25
    word = word_en if language == "en" else word_ro
    if language == "en":
        text = f"Write the word '{word}' in italics using markdown." if italic else rng.choice([f"Write the word '{word}' in bold using markdown.", f"Make '{word}' bold in markdown.", f"Format the word '{word}' as bold."])
    else:
        text = f"Scrie cuvântul „{word}” cursiv (italic), în markdown." if italic else rng.choice([f"Scrie cuvântul „{word}” îngroșat (bold), în markdown.", f"Formatează cuvântul „{word}” ca bold."])
    return [user(text), assistant(f"*{word}*" if italic else f"**{word}**")]


def heading_conversation(rng, language):
    title_en, title_ro = rng.choice(TITLES)
    key = rng.choice(list(CATEGORIES))
    _, _, items = CATEGORIES[key]
    n = rng.randint(2, 4)
    chosen = pick_items(rng, items, n)
    index = 0 if language == "en" else 1
    title = title_en if language == "en" else title_ro
    if language == "en":
        text = f"Write a markdown heading '{title}', followed by {n} items as a bulleted list."
    else:
        text = f"Scrie un titlu markdown „{title}”, urmat de {n} elemente ca listă cu buline."
    return [user(text), assistant(f"## {title}\n" + "\n".join(f"- {item[index]}" for item in chosen))]


def sentence_conversation(rng, language):
    topic_en, topic_ro, sentence_en, sentence_ro = rng.choice(SENTENCES)
    if language == "en":
        return [user(rng.choice([f"Write one sentence about {topic_en}.", f"Say something about {topic_en} in one sentence.", f"One sentence about {topic_en}, please."])), assistant(sentence_en)]
    return [user(rng.choice([f"Scrie o singură propoziție despre {topic_ro}.", f"Spune ceva despre {topic_ro} într-o singură propoziție."])), assistant(sentence_ro)]


def language_conversation(rng, language):
    if rng.random() < 0.7:
        country_en, capital_en, genitive_ro, country_ro = rng.choice(CAPITALS)
        if language == "en":     # asked in Romanian, answer in English
            return [user(rng.choice([f"Answer in English: care este capitala {genitive_ro}?", f"Răspunde în engleză: care este capitala {genitive_ro}?"])),
                    assistant(f"The capital of {country_en} is {capital_en}.")]
        return [user(rng.choice([f"Reply in Romanian: what is the capital of {country_en}?", f"Answer in Romanian: what is the capital of {country_en}?"])),
                assistant(f"Capitala {genitive_ro} este {ROMANIAN_CAPITAL[capital_en]}.")]
    question_en, answer_en, question_ro, answer_ro = rng.choice(SMALL_TALK)
    if language == "en":
        return [user(f"Answer in English: {question_ro}"), assistant(answer_en)]
    return [user(f"Reply in Romanian: {question_en}"), assistant(answer_ro)]


KINDS = [("lists", list_conversation, 450), ("one_word", one_word_conversation, 350), ("yes_no", yes_no_conversation, 350),
         ("bold", bold_conversation, 220), ("heading", heading_conversation, 220), ("sentence", sentence_conversation, 220),
         ("language", language_conversation, 260)]


def main():
    rng = random.Random(21)
    conversations = []
    for name, function, count in KINDS:
        for _ in range(count):
            conversations.append(function(rng, "ro" if rng.random() < 0.45 else "en"))
        print(f"  {name:10s} {count}")
    write_jsonl("gen_constraints", conversations, rng)


if __name__ == "__main__":
    main()
