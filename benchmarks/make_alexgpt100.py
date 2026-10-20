"""
Writes benchmarks/alexgpt100.jsonl: the 100 fixed questions of the AlexGPT-100 benchmark.
Run once: python benchmarks/make_alexgpt100.py      (the .jsonl is what eval_alexgpt100.py reads)

Every question has: id, category, lang, prompt, optional context, and a list of checks.
Checks the computer can do by itself:
    contains     the answer has one of these words (no diacritics/case needed)
    number       the answer has this number
    bullets      the answer has exactly N list lines (- item, * item, 1. item)
    max_words    the answer has at most N words
    max_sentences  at most N sentences
    language     the answer is mostly "en" or "ro"
    bold         the answer has **bold** text
    heading      the answer has a markdown heading (# Title)
    unsure       the answer says it doesn't know / would need to search
    manual       a person gives 0, 1 or 2 points (eval_alexgpt100.py --grade)
The RAG questions are about INVENTED places and people on purpose: the model can only
answer them by reading the context, not from memory.
"""
import json
import os

CONVERSATION_RUBRIC = "2 = natural, friendly, on topic, right language; 1 = ok but generic or a bit odd; 0 = off topic, wrong language or nonsense"
SUMMARY_RUBRIC = "2 = short, correct, keeps the main idea; 1 = partly correct or too long; 0 = wrong, copied or nonsense"

questions = []


def add(category, lang, prompt, checks, context=None):
    number = sum(1 for q in questions if q["category"] == category) + 1
    item = {"id": f"{category}_{number:02d}", "category": category, "lang": lang, "prompt": prompt, "checks": checks}
    if context:
        item["context"] = context
    questions.append(item)


# ------------------------------------------------------------------ conversation (manual)
for lang, prompt in [
    ("en", "Hi! How are you today?"),
    ("en", "I just got back from university and I'm really tired."),
    ("en", "Can you recommend something fun to do on a rainy weekend?"),
    ("en", "I failed my exam today. Any advice?"),
    ("en", "What's your favourite season and why?"),
    ("en", "Thanks for your help earlier!"),
    ("en", "I'm learning to play the guitar. Where should I start?"),
    ("en", "Tell me a short joke."),
    ("ro", "Salut! Ce mai faci?"),
    ("ro", "Azi am fost la facultate și am avut un laborator foarte greu."),
    ("ro", "Ce pot să gătesc rapid diseară?"),
    ("ro", "Mi-e cam dor de casă."),
    ("ro", "Îmi recomanzi o carte bună?"),
    ("ro", "Mulțumesc mult pentru ajutor!"),
    ("ro", "Cum pot să învăț mai eficient pentru examene?"),
]:
    add("conversation", lang, prompt, [{"type": "language", "value": lang}, {"type": "manual", "rubric": CONVERSATION_RUBRIC}])

# ------------------------------------------------------------------ facts it should know (no search)
for lang, prompt, answers in [
    ("en", "What is the capital of France?", ["Paris"]),
    ("en", "How many continents are there on Earth?", ["seven", "7"]),
    ("en", "Which gas do plants take from the air for photosynthesis?", ["carbon dioxide", "CO2"]),
    ("en", "Who wrote Romeo and Juliet?", ["Shakespeare"]),
    ("en", "What is the largest planet in the Solar System?", ["Jupiter"]),
    ("en", "At what temperature does water boil at sea level, in degrees Celsius?", ["100"]),
    ("en", "What is the chemical symbol for gold?", ["Au"]),
    ("en", "How many days are there in a leap year?", ["366"]),
    ("ro", "Care este capitala României?", ["București"]),
    ("ro", "Ce fluviu mare se varsă în Marea Neagră printr-o deltă, în România?", ["Dunăr"]),
    ("ro", "Cine a scris poezia „Luceafărul”?", ["Eminescu"]),
    ("ro", "Care este cel mai înalt vârf muntos din România?", ["Moldoveanu"]),
    ("ro", "Câte zile are luna februarie într-un an obișnuit?", ["28"]),
    ("ro", "În ce an a avut loc Marea Unire a României?", ["1918"]),
    ("ro", "Ce limbă oficială se vorbește în Republica Moldova?", ["român"]),
]:
    add("facts", lang, prompt, [{"type": "contains", "values": answers}])

# ------------------------------------------------------------------ answer from a given text (RAG)
for lang, context, prompt, answers in [
    ("en", "Velmora is a small town in the northern hills. It was founded in 1742 by salt miners, and today about 3,400 people live there. Its main festival, Lantern Night, takes place every October.",
     "When was Velmora founded?", ["1742"]),
    ("en", "The Orrin river is 212 kilometres long. It starts in the Kesh mountains and flows into Lake Tamsin.",
     "Into which lake does the Orrin river flow?", ["Tamsin"]),
    ("en", "Dr. Hana Ilves invented the Corvex lamp in 1931. The lamp used a coiled glass tube filled with neon and was first sold in Tallinn.",
     "Who invented the Corvex lamp?", ["Ilves"]),
    ("en", "The Bristle Cup is a chess tournament held every two years. The 2024 edition was won by Marta Okafor, who beat Leon Varga in the final.",
     "Who won the 2024 Bristle Cup?", ["Okafor"]),
    ("en", "Zephyrine tea is made from the dried leaves of the blue sage plant. It is usually served cold with lemon and honey.",
     "Which plant is Zephyrine tea made from?", ["blue sage"]),
    ("en", "The Kalden Bridge has 14 arches and is 380 metres long. It was finished in 1889 after nine years of work.",
     "How many arches does the Kalden Bridge have?", ["14", "fourteen"]),
    ("en", "In the game Starfold, players collect crystals called shards. A player needs 50 shards to unlock the final level, which is called the Hollow Gate.",
     "What is the final level of Starfold called?", ["Hollow Gate"]),
    ("en", "The Pelican Robotics Club meets every Thursday at 6 pm in room B204. New members must bring a laptop and a signed form.",
     "On which day does the Pelican Robotics Club meet?", ["Thursday"]),
    ("ro", "Satul Stejăreni se află la poalele munților Vâlcan. A fost atestat documentar în anul 1563 și are astăzi aproximativ 900 de locuitori.",
     "În ce an a fost atestat documentar satul Stejăreni?", ["1563"]),
    ("ro", "Lacul Ardeal-Mic are o adâncime maximă de 27 de metri. În el trăiesc păstrăvi și știuci.",
     "Ce adâncime maximă are lacul Ardeal-Mic?", ["27"]),
    ("ro", "Inginera Ioana Vlaicu a proiectat în 1978 primul tramvai electric din orașul Novaria. Tramvaiul avea două vagoane și 60 de locuri.",
     "Câte locuri avea tramvaiul proiectat de Ioana Vlaicu?", ["60"]),
    ("ro", "Festivalul Lumini de Toamnă are loc în fiecare an în luna octombrie, în piața centrală din Doroșani. Anul acesta au participat 40 de artiști.",
     "În ce lună are loc Festivalul Lumini de Toamnă?", ["octombrie"]),
    ("ro", "Prăjitura „Norișor” se face din albuș de ou, zahăr și fulgi de cocos. Se coace 40 de minute la 120 de grade.",
     "Cât timp se coace prăjitura Norișor?", ["40"]),
    ("ro", "Clubul de astronomie Orion se întâlnește în fiecare vineri seara pe acoperișul facultății. Membrii observă planete cu un telescop de 20 de centimetri.",
     "În ce zi se întâlnește clubul Orion?", ["vineri"]),
    ("ro", "Compania Fenix Soft a fost înființată la Iași de trei studenți, în 2016. Primul ei produs a fost o aplicație pentru orarul autobuzelor.",
     "Care a fost primul produs al companiei Fenix Soft?", ["autobuz", "aplicați"]),
]:
    add("rag", lang, prompt, [{"type": "contains", "values": answers}], context)

# ------------------------------------------------------------------ calculations (later: with the calc tool)
for lang, prompt, value in [
    ("en", "What is 4 + 3 * 6?", 22),
    ("en", "What is 847 * 293?", 248171),
    ("en", "What is 15% of 240?", 36),
    ("en", "What is 1024 divided by 16?", 64),
    ("en", "I have 3 apples, buy 5 more and then eat 2. How many apples do I have now?", 6),
    ("en", "What is 12 squared?", 144),
    ("en", "A train travels at 60 km per hour for 2.5 hours. How many kilometres does it travel?", 150),
    ("en", "What is 999 + 1001?", 2000),
    ("ro", "Cât face 7 * 8?", 56),
    ("ro", "Cât face 125 + 375?", 500),
    ("ro", "Cât face 1000 - 357?", 643),
    ("ro", "Cât este 20% din 150?", 30),
    ("ro", "Un caiet costă 4 lei. Cât costă 9 caiete?", 36),
    ("ro", "Cât face 81 împărțit la 9?", 9),
    ("ro", "Cât face (12 + 8) * 5?", 100),
]:
    add("math", lang, prompt, [{"type": "number", "value": value}])

# ------------------------------------------------------------------ following instructions + markdown
for lang, prompt, checks in [
    ("en", "List three fruits as a bulleted list.", [{"type": "bullets", "value": 3}]),
    ("en", "Answer with only one word: what colour is the sky on a clear day?",
     [{"type": "max_words", "value": 2}, {"type": "contains", "values": ["blue"]}]),
    ("en", "Write the word 'hello' in bold using markdown.", [{"type": "bold"}, {"type": "contains", "values": ["hello"]}]),
    ("en", "Give me exactly two reasons to drink water, as a numbered list.", [{"type": "bullets", "value": 2}]),
    ("en", "Reply in Romanian: what is your favourite food?", [{"type": "language", "value": "ro"}]),
    ("en", "Write one sentence about cats.", [{"type": "max_sentences", "value": 1}, {"type": "contains", "values": ["cat"]}]),
    ("en", "Write a markdown heading 'Shopping list', followed by two items as a bulleted list.",
     [{"type": "heading"}, {"type": "bullets", "value": 2}]),
    ("en", "Answer only 'yes' or 'no': is the Earth round?", [{"type": "max_words", "value": 2}, {"type": "contains", "values": ["yes"]}]),
    ("ro", "Enumeră trei animale domestice sub formă de listă.", [{"type": "bullets", "value": 3}]),
    ("ro", "Răspunde cu un singur cuvânt: ce culoare are iarba?",
     [{"type": "max_words", "value": 2}, {"type": "contains", "values": ["verde"]}]),
    ("ro", "Scrie cuvântul „important” îngroșat (bold), în markdown.", [{"type": "bold"}, {"type": "contains", "values": ["important"]}]),
    ("ro", "Answer in English: care este capitala Italiei?", [{"type": "language", "value": "en"}, {"type": "contains", "values": ["Rome"]}]),
    ("ro", "Dă-mi două sfaturi pentru un somn bun, ca listă numerotată.", [{"type": "bullets", "value": 2}]),
    ("ro", "Scrie o singură propoziție despre munte.", [{"type": "max_sentences", "value": 1}, {"type": "language", "value": "ro"}]),
    ("ro", "Răspunde doar cu „da” sau „nu”: este Soarele o stea?", [{"type": "max_words", "value": 2}, {"type": "contains", "values": ["da"]}]),
]:
    add("instructions", lang, prompt, checks)

# ------------------------------------------------------------------ summaries
SUMMARY_TEXTS = [
    ("en", "Honeybees live in colonies that can have more than fifty thousand members. Most of them are workers, female bees that collect nectar and pollen, build the wax comb and protect the hive. A single queen lays the eggs, sometimes more than a thousand in one day. While they fly from flower to flower, bees carry pollen with them, which helps plants make seeds and fruit. Many of the fruits and vegetables people eat depend on this pollination, so a drop in the number of bees can affect farms and food prices.",
     ["bee"]),
    ("en", "When Maya was fourteen, she found an old broken vacuum cleaner in her grandfather's garage. Instead of throwing it away, she took it apart and kept the motor. Over the next summer she used the motor, two toy wheels and a cheap sensor to build a small robot that could follow a black line on the floor. Her first attempts failed again and again, but she wrote down every mistake. In autumn she showed the robot at her school's science fair and won second place.",
     ["robot"]),
    ("en", "Sleep is not only rest for the body; it is also when the brain sorts and stores what it learned during the day. Adults usually need between seven and nine hours each night, and teenagers need even more. People who sleep too little for many days find it harder to concentrate, remember things and control their mood. Simple habits help: going to bed at the same time, keeping the room dark and cool, and avoiding bright screens in the last hour before sleep.",
     ["sleep"]),
    ("en", "The first bicycles, built in the early nineteenth century, had no pedals at all: riders pushed themselves along with their feet. Later designs added pedals to the front wheel, which led to the famous penny-farthing with its huge front wheel and tiny back wheel. It was fast but dangerous. In the 1880s the safety bicycle appeared, with two wheels of the same size and a chain that turned the back wheel. Its shape is almost the same as the bicycles people ride today.",
     ["bicycle", "bike"]),
    ("en", "Only a small part of the plastic that people throw away is recycled. Many plastic objects are made from several kinds of plastic glued together, which makes them hard to separate. Others are dirty with food, so recycling plants cannot use them. Much of the rest ends up in landfills or in the environment, where it can take hundreds of years to break down. Experts say the most useful steps are to use fewer single-use items and to design products that are easier to recycle.",
     ["plastic", "recycl"]),
    ("ro", "Delta Dunării este locul în care fluviul se desparte în trei brațe principale înainte de a se vărsa în Marea Neagră. Este una dintre cele mai mari zone umede din Europa și adăpostește peste trei sute de specii de păsări, dintre care multe vin aici doar vara. Localnicii trăiesc mai ales din pescuit și turism, iar în multe sate se ajunge doar cu barca. Din 1991 delta este protejată ca rezervație a biosferei, pentru a păstra natura sălbatică.",
     ["delt"]),
    ("ro", "Corpul uman este format în mare parte din apă, iar aceasta ajută la reglarea temperaturii, la digestie și la transportul substanțelor hrănitoare. Chiar și o lipsă mică de apă poate provoca oboseală, dureri de cap și o concentrare mai slabă. Necesarul zilnic diferă de la o persoană la alta, dar crește când este cald sau când facem sport. Un semn simplu că bem suficient este că nu ne este sete pe parcursul zilei.",
     ["apă", "apa"]),
    ("ro", "Într-o seară de toamnă, pisica familiei Popescu, o motană portocalie pe nume Ghimbir, nu s-a mai întors acasă. Copiii au lipit afișe pe stâlpii din cartier și au întrebat toți vecinii. Au trecut trei zile fără nicio veste. În a patra dimineață, un vecin a auzit mieunat din garajul lui, care fusese închis tot weekendul. Ghimbir era acolo, flămând, dar sănătos, și a fost primit acasă cu o farfurie mare de lapte.",
     ["pisic", "Ghimbir", "motan"]),
    ("ro", "Prima cale ferată de pe teritoriul actual al României a fost deschisă în 1854, între Oravița și Baziaș, și servea la transportul cărbunelui spre Dunăre. Câțiva ani mai târziu au apărut și primele trenuri pentru călători. În perioada următoare rețeaua s-a extins rapid, legând marile orașe și porturile. Calea ferată a schimbat felul în care oamenii călătoreau și comerțul, pentru că drumurile care durau zile întregi au ajuns să dureze doar câteva ore.",
     ["tren", "cale ferat"]),
    ("ro", "Mulți începători cred că pentru a învăța programare trebuie să memoreze multe comenzi. De fapt, partea cea mai importantă este să învețe să împartă o problemă mare în pași mici și clari. Un limbaj precum Python este potrivit pentru început, pentru că are o sintaxă simplă. Cel mai bine se învață prin proiecte mici, de exemplu un joc sau un calculator, iar greșelile fac parte normal din proces.",
     ["program"]),
]
for lang, text, keywords in SUMMARY_TEXTS:
    prompt = "Summarize the text above in two or three sentences." if lang == "en" else "Rezumă textul de mai sus în două-trei propoziții."
    add("summary", lang, prompt, [{"type": "max_words", "value": 80}, {"type": "language", "value": lang},
                                  {"type": "contains", "values": keywords}, {"type": "manual", "rubric": SUMMARY_RUBRIC}], text)

# ------------------------------------------------------------------ who it is + knowing when it doesn't know
NAMES = ["AlexGPT", "Singularity", "Supernova"]
for lang, prompt, checks in [
    ("en", "Who are you?", [{"type": "contains", "values": NAMES}]),
    ("en", "Who created you?", [{"type": "contains", "values": ["Alex"]}]),
    ("en", "Are you ChatGPT?", [{"type": "contains", "values": NAMES + ["no"]}]),
    ("en", "How many parameters do you have?", [{"type": "contains", "values": ["million", "100", "98", "125"]}]),
    ("en", "What did I eat for breakfast today?", [{"type": "unsure"}]),
    ("en", "What is the exact price of Bitcoin right now?", [{"type": "unsure"}]),
    ("en", "What will the weather be like in Cluj tomorrow?", [{"type": "unsure"}]),
    ("en", "What is my sister's name?", [{"type": "unsure"}]),
    ("ro", "Cine ești?", [{"type": "contains", "values": NAMES}]),
    ("ro", "Cine te-a creat?", [{"type": "contains", "values": ["Alex"]}]),
    ("ro", "Ești om?", [{"type": "contains", "values": ["model", "AI", "inteligență artificială", "asistent", "program"]}]),
    ("ro", "Ce am făcut ieri seară?", [{"type": "unsure"}]),
    ("ro", "Cât va fi cursul euro mâine?", [{"type": "unsure"}]),
    ("ro", "Cum o cheamă pe profesoara mea de matematică?", [{"type": "unsure"}]),
    ("ro", "Ce poți să faci?", [{"type": "contains", "values": ["întrebări", "răspund", "ajut", "caut", "explic"]}]),
]:
    add("identity", lang, prompt, checks)

assert len(questions) == 100, len(questions)
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "alexgpt100.jsonl")
with open(path, "w", encoding="utf-8") as file:
    for item in questions:
        file.write(json.dumps(item, ensure_ascii=False) + "\n")
print(f"wrote {len(questions)} questions to {path}")
