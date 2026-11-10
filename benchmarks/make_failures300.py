"""Builds benchmarks/failures300.jsonl: ~300 questions split by FAILURE TYPE (CLAUDE.md "Known problems").
    .venv\\Scripts\\python.exe benchmarks/make_failures300.py
Every item:  {"id", "category", "kind", "turns": [messages...], "expect": regex or None, "min_matches", "lang", "split"}
  turns   the user's messages, in order; only the LAST answer is graded (the others build the history)
  kind    how it is graded (eval_failures.py):
            fact           right = expect matches;  abstain = the "no source" reply;  wrong = anything else
            false_premise  right = abstained or says it found nothing;  wrong = anything else
            chat           right = not abstained and in the right language;  wrong = abstained or wrong language
            identity       right = expect matches
            style          right = short (< 300 chars), no list, and expect matches (if given)
            list           right = at least min_matches different names from expect
            tool           right = expect matches
  split   "heldout" for every 4th item of a category (about 25%): NEVER tune on these, only report them."""
import json
import os
import re
import sys
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

FACTS_EN = [
    ("who wrote Hamlet?", "shakespeare"), ("who painted the Mona Lisa?", "leonardo|da vinci"),
    ("what is the capital of Australia?", "canberra"), ("what is the capital of Canada?", "ottawa"),
    ("what is the capital of Japan?", "tokyo"), ("how tall is Mount Everest?", r"8,?8\d\d|29,0\d\d"),
    ("when did World War II end?", "1945"), ("when did World War I begin?", "1914"),
    ("who discovered penicillin?", "fleming"), ("who developed the theory of relativity?", "einstein"),
    ("what is the largest planet in the solar system?", "jupiter"), ("what is the chemical symbol for gold?", r"\bau\b"),
    ("who was the first person to walk on the Moon?", "armstrong"), ("when did the Berlin Wall fall?", "1989"),
    ("what is the longest river in Africa?", r"\bnile"), ("how many bones are in the adult human body?", "206"),
    ("who wrote Pride and Prejudice?", "austen"), ("what is the capital of Romania?", "bucharest"),
    ("when was the Eiffel Tower completed?", "1889"), ("who composed the Moonlight Sonata?", "beethoven"),
    ("what is the boiling point of water in Celsius?", r"\b100\b"), ("who was the first president of the United States?", "washington"),
    ("what is the smallest planet in the solar system?", "mercury"), ("when did the Titanic sink?", "1912"),
    ("who invented the light bulb?", "edison|swan"), ("what is the capital of Italy?", r"\brome\b"),
    ("which ocean is the largest?", "pacific"), ("who wrote 1984?", "orwell"),
    ("what is the speed of light?", r"299|300,000|300 000"), ("when did Romania join NATO?", "2004"),
    ("when did Romania join the European Union?", "2007"), ("who founded Microsoft?", "gates|allen"),
    ("what is the currency of Japan?", r"\byen\b"), ("who painted The Starry Night?", "gogh"),
    ("what language is spoken in Brazil?", "portuguese"), ("what is the highest mountain in Romania?", "moldoveanu"),
    ("who wrote The Odyssey?", "homer"), ("when did Christopher Columbus reach the Americas?", "1492"),
    ("what is the capital of Spain?", "madrid"), ("who formulated the law of universal gravitation?", "newton"),
]
FACTS_RO = [
    ("cine a scris Luceafarul?", "eminescu"), ("care este capitala Frantei?", "paris"),
    ("cand a avut loc Marea Unire?", "1918"), ("cine a fost Stefan cel Mare?", "moldov|domn|voievod"),
    ("care este cel mai lung râu din România?", "dunar|mures"), ("unde se afla Delta Dunarii?", "tulcea|romania|dobrog|ucrain"),
    ("cine a inventat stiloul?", "poenaru"), ("când s-a născut Mihai Eminescu?", "1850"),
    ("care este capitala Germaniei?", "berlin"), ("cine a pictat Gioconda?", "leonardo|vinci"),
    ("ce limba se vorbeste in Austria?", "german"), ("cand a cazut Zidul Berlinului?", "1989"),
    ("cine a fost primul om pe Luna?", "armstrong"), ("care este cea mai mare planetă din sistemul solar?", "jupiter"),
    ("în ce an a murit Ștefan cel Mare?", "1504"), ("cine a compus Rapsodia romana?", "enescu"),
    ("care este capitala Republicii Moldova?", "chisinau"), ("cand a intrat Romania in Uniunea Europeana?", "2007"),
    ("cine a scris Amintiri din copilărie?", "creang"), ("unde este Castelul Bran?", "brasov|bran"),
    ("cine a construit Turnul Eiffel?", "eiffel"), ("când a avut loc Revoluția Română?", "1989"),
    ("care este cel mai inalt varf din Romania?", "moldoveanu"), ("cine a fost Constantin Brancusi?", "sculpt"),
    ("cine a inventat telefonul?", "bell|meucci"), ("ce este fotosinteza?", "lumin|plant|energ"),
    ("care este capitala Italiei?", r"\broma\b"), ("în ce an a aderat România la NATO?", "2004"),
    ("cine a scris Moara cu noroc?", "slavici"), ("cine a fost Henri Coanda?", "inventator|aviat|fizician|inginer"),
]
FACTS_HARD = [
    ("who invented the telephone?", "bell|meucci"), ("who invented the radio?", "marconi|tesla|popov"),
    ("who invented the airplane?", "wright"), ("who invented the printing press?", "gutenberg"),
    ("who discovered America?", "columbus|leif|erikson"), ("who discovered the structure of DNA?", "watson|crick|franklin"),
    ("who invented the World Wide Web?", "berners"), ("who invented the steam engine?", "watt|newcomen|savery"),
    ("who discovered radium?", "curie"), ("who invented the electric battery?", "volta"),
    ("who wrote Don Quixote?", "cervantes"), ("who wrote War and Peace?", "tolstoy"),
    ("who painted the Sistine Chapel ceiling?", "michelangelo"), ("who was the first emperor of Rome?", "augustus"),
    ("who killed Julius Caesar?", "brutus|cassius"), ("who founded Apple?", "jobs|wozniak"),
    ("who invented the telescope?", "lippershey|galileo"), ("who invented the computer?", "babbage|turing|zuse|neumann"),
    ("who invented dynamite?", "nobel"), ("who composed The Four Seasons?", "vivaldi"),
    ("when was the United Nations founded?", "1945"), ("when was NATO founded?", "1949"),
    ("when did the French Revolution begin?", "1789"), ("when did humans first land on the Moon?", "1969"),
    ("when did the Western Roman Empire fall?", r"\b476\b"), ("when was the euro introduced?", "1999|2002"),
    ("when was Google founded?", "1998"), ("when was Facebook founded?", "2004"),
    ("when did the Chernobyl disaster happen?", "1986"), ("when did Napoleon die?", "1821"),
]
FALSE_PREMISE = [
    "when did Napoleon visit Cluj?", "what did Mihai Eminescu say about Tokyo?", "when did Albert Einstein live in Bucharest?",
    "who did Stefan cel Mare fight in Brazil?", "when did the Beatles play in Jibou?", "how did Shakespeare describe Romania?",
    "când a vizitat Napoleon Clujul?", "when did Elon Musk buy Dacia?", "who built the Eiffel Tower in Cluj?",
    "when did Mozart visit Sibiu?", "when did Nikola Tesla live in Iasi?", "why did Dracula build the Eiffel Tower?",
    "when did Taylor Swift perform in Jibou?", "când a cântat Michael Jackson în Jibou?", "ce a spus Eminescu despre Japonia?",
    "when did Shakespeare visit Bucharest?", "what did Abraham Lincoln say about Romania?", "when did Leonardo da Vinci paint in Cluj?",
    "when did Eminescu travel to America?", "why did Einstein move to Sibiu?", "when did Mozart compose music in Iasi?",
    "when did Isaac Newton live in Paris?", "when did Cleopatra visit Romania?", "who did Napoleon marry in Bucharest?",
    "when did Brancusi win the Nobel Prize?", "when did Elvis Presley perform in Cluj?", "why did Tesla build a tower in Brasov?",
    "when did Picasso live in Timisoara?", "when did Gandhi visit Constanta?", "cand a jucat Hagi la Manchester United?",
    "când a vizitat Einstein Timișoara?", "de ce a construit Stefan cel Mare Turnul Eiffel?", "când a cântat Mozart la Cluj?",
    "când a locuit Shakespeare în Brașov?", "ce a inventat Eminescu in Japonia?", "when did Albert Einstein play for Barcelona?",
    "when did Mount Everest erupt?", "when did Romania win the FIFA World Cup?", "who was the king of Australia in 1500?",
    "when did the Moon landing happen in Romania?",
]
FOLLOWUP = [
    (["who is Alesso?", "where is he from?"], "swed|stockholm"), (["what is the capital of France?", "and of Germany?"], "berlin"),
    (["who wrote Hamlet?", "when was he born?"], "1564"), (["tell me about the Danube", "how long is it?"], r"2,?8\d\d|1,7\d\d"),
    (["who was Napoleon?", "when did he die?"], "1821"), (["what is the weather in Paris?", "and in Cluj?"], "cluj"),
    (["what time is it in Tokyo?", "and in London?"], r"\d{1,2}:\d{2}"), (["who painted the Mona Lisa?", "when was he born?"], "1452"),
    (["cine a scris Luceafarul?", "cand s-a nascut el?"], "1850"), (["care este capitala Frantei?", "dar a Italiei?"], r"\broma\b"),
    (["who invented the telephone?", "where was he born?"], "edinburgh|scotland"), (["who is Elon Musk?", "what companies does he run?"], "tesla|spacex"),
    (["what is NATO?", "when was it founded?"], "1949"), (["who was Albert Einstein?", "what prize did he win?"], "nobel"),
    (["what is Cluj-Napoca?", "how many people live there?"], r"\d{3}[ ,.]?\d{3}"), (["tell me about Mount Everest", "how tall is it?"], r"8,?8\d\d"),
    (["cine a fost Stefan cel Mare?", "cand a murit?"], "1504"), (["what is the capital of Japan?", "and of China?"], "beijing"),
    (["who wrote 1984?", "what else did he write?"], "animal farm"), (["what countries are in Scandinavia?", "and in the Baltics?"], "estonia|latvia|lithuania"),
    (["who discovered penicillin?", "when?"], "1928"), (["tell me about the Eiffel Tower", "when was it built?"], "1887|1889"),
    (["who is the author of Harry Potter?", "where was she born?"], "yate|england|gloucestershire|bristol"),
    (["what is the largest planet?", "how many moons does it have?"], r"\b\d{2}\b"),
    (["cine a fost Mihai Viteazul?", "cand a murit el?"], "1601"), (["ce este Dunarea?", "cat de lunga este?"], r"2.?8\d\d"),
    (["who founded Microsoft?", "when?"], "1975"), (["who was Mozart?", "where was he born?"], "salzburg"),
    (["what is the Louvre?", "where is it?"], "paris"), (["who was Nikola Tesla?", "where was he born?"], "smiljan|croatia|austria"),
]
STYLE_TOPICS = [["NATO", "the Danube", "Mount Everest"], ["Romania", "Paris", "the Moon"], ["Napoleon", "Mihai Eminescu", "Albert Einstein"],
                ["Japan", "Brazil", "Egypt"], ["photosynthesis", "black holes", "volcanoes"]]
STYLE_FINALS = [("What is your name?", "singularity"), ("how are you?", None), ("thanks!", None), ("Cum te cheamă?", "singularity")]
IDENTITY = [
    ("who are you exactly?", "singularity"), ("what's your name, bot?", "singularity"), ("who built you?", "alex"),
    ("who programmed you?", "alex"), ("are you made by OpenAI?", r"alex|singularity|\bno\b|nope"), ("are you Gemini?", r"\bno\b|nope|singularity"),
    ("how big are you?", "125|million|small"), ("how were you trained?", "train|alex|gpu|card|data"),
    ("can you browse the internet?", r"wiki|internet|search|\bno\b|can't|cannot"), ("are you a real person?", r"\bno\b|\bnot\b|model|\bai\b"),
    ("what can you help me with?", "chat|explain|question|summar"), ("do you have feelings?", r"\bno\b|\bnot\b|model|don't"),
    ("what's my favorite color?", "don't know|no way|tell me|haven't"), ("where do I work?", "don't know|no way|tell me|haven't"),
    ("what family of models are you from?", "cosmos"), ("cum te numești?", "singularity"), ("cine te-a programat?", "alex"),
    ("ești făcut de OpenAI?", r"alex|\bnu\b"), ("câți parametri ai?", "125|milioane"), ("ești o persoană reală?", r"\bnu\b|model"),
    ("unde lucrez eu?", "nu stiu|nu am cum|spune-mi|nu mi-ai"), ("cum ai fost antrenat?", "antren|alex|plac"),
    ("ce poți face pentru mine?", "pot|explic|rezum|vorbesc"), ("ai sentimente?", r"\bnu\b|model"),
    ("din ce familie de modele faci parte?", "cosmos"), ("what does CosmosLM mean?", "cosmos|famil|alex"),
    ("are you smarter than ChatGPT?", r"\bno\b|small|smaller"), ("do you remember our last conversation?", r"\bno\b|don't|only|can't"),
    ("care e numele tău?", "singularity"), ("tell me who made you", "alex"),
    ("yo, who am I chatting with?", "singularity"), ("introduce urself pls", "singularity"), ("what should I call u?", "singularity"),
    ("whose project are you?", "alex"), ("who's behind you?", "alex"), ("what kind of AI are you?", "language model|singularity|small"),
    ("are you GPT-4?", r"\bno\b|nope|singularity"), ("are you Claude?", r"\bno\b|nope|singularity"), ("how many params?", "125|million"),
    ("on what hardware were you trained?", "card|gpu|graphics|computer"), ("is a human typing these answers?", r"\bno\b|\bnot\b|model|\bai\b"),
    ("can you look things up online?", r"wiki|internet|search|\bno\b|can't|cannot"), ("what do you know about my job?", "don't know|no way|tell me|haven't"),
    ("what's my mom's name?", "don't know|no way|tell me|haven't"), ("tell me about CosmosLM", "cosmos|famil|alex"),
    ("cine te-a făcut pe tine, botule?", "alex"), ("cine e Alex?", "alex|student|cluj"), ("ce fel de AI ești?", "model|singularity"),
    ("ești Claude?", r"\bnu\b|singularity"), ("pe ce placă video ai fost antrenat?", "plac|gpu|calculator"),
    ("știi cum o cheamă pe mama mea?", "nu stiu|nu am cum|spune-mi|nu mi-ai"), ("poți căuta pe internet?", r"wiki|internet|caut|\bnu\b"),
    ("da-mi un rezumat despre tine", "singularity|alex|model"), ("what are your limits?", "small|mistake|wrong|facts|can't"),
    ("which company owns you?", r"alex|\bno\b|singularity|student"),
]
CHITCHAT = [
    ("hi there", "en"), ("good morning!", "en"), ("I'm bored", "en"), ("I just finished my exam!", "en"), ("thank you so much", "en"),
    ("see you later", "en"), ("I'm feeling sad today", "en"), ("can you cheer me up?", "en"), ("what should I do this weekend?", "en"),
    ("tell me something nice", "en"), ("I love pizza", "en"), ("it's raining today", "en"), ("I'm going to sleep", "en"),
    ("how was your day?", "en"), ("you're awesome", "en"),
    ("salut!", "ro"), ("buna dimineata", "ro"), ("ma plictisesc", "ro"), ("am terminat examenul!", "ro"), ("mersi mult", "ro"),
    ("pe curand", "ro"), ("sunt trist azi", "ro"), ("ce sa fac in weekend?", "ro"), ("spune-mi ceva frumos", "ro"),
    ("imi place pizza", "ro"), ("ploua azi", "ro"), ("ma duc la culcare", "ro"), ("cum a fost ziua ta?", "ro"),
    ("esti super", "ro"), ("noapte buna", "ro"),
]
LISTS = [
    ("what is the most visited country in the world?", "france", 1), ("what is the largest country in the world?", "russia", 1),
    ("what is the most populous country?", "india|china", 1), ("what is the smallest country in the world?", "vatican", 1),
    ("what is the tallest building in the world?", "burj khalifa", 1), ("what is the deepest ocean?", "pacific|mariana", 1),
    ("what is the largest desert in the world?", "sahara|antarctic", 1), ("what is the longest river in the world?", r"\bnile|amazon", 1),
    ("which country has the most islands?", "sweden|finland|norway|indonesia", 1), ("what is the biggest city in Romania?", "bucure|bucharest", 1),
    ("what is the largest lake in the world?", "caspian", 1), ("what is the fastest animal?", "cheetah|falcon", 1),
    ("care este cel mai mare oras din Romania?", "bucure", 1), ("care este cea mai mare tara din lume?", "rusia", 1),
    ("care este cel mai lung rau din lume?", r"\bnil|amazon", 1),
    ("list some songs by Calvin Harris", "summer|feel so close|this is what you came for|one kiss|we found love|outside|how deep", 2),
    ("list albums by Queen", "night at the opera|news of the world|jazz|innuendo|kind of magic|sheer heart attack|the game", 2),
    ("name some books by Mark Twain", "tom sawyer|huckleberry|prince and the pauper|connecticut yankee", 2),
    ("what are the largest cities in Europe?", "istanbul|moscow|london|paris|madrid|berlin", 2),
    ("which countries border Romania?", "hungary|ukraine|serbia|bulgaria|moldova", 2),
]
TOOLS = [
    ("what is 245 * 18?", "4410|4,410"), ("what is 15% of 240?", r"\b36\b"), ("what is the square root of 144?", r"\b12\b"),
    ("what is 2^10?", "1024|1,024"), ("1234 + 5678", "6912|6,912"), ("radical din 81", r"\b9\b"), ("cat face 7 x 8?", r"\b56\b"),
    ("what is 100 / 8?", r"12[.,]5"), ("10 km to miles", r"6[.,]21"), ("100 celsius to fahrenheit", r"\b212\b"),
    ("5 kg to pounds", r"11[.,]0"), ("1 mile to km", r"1[.,]609"), ("what time is it in Tokyo?", r"\d{1,2}:\d{2}"),
    ("ce ora e in Paris?", r"\d{1,2}:\d{2}"), ("weather in London", "°"), ("vremea in Iasi", "°"),
    ("roll 2 dice", r"\d"), ("flip a coin", "heads|tails|cap|pajur|ban"), ("alege dintre pizza si paste", "pizza|paste"),
    ("what happened today in history?", r"\d{3,4}"),
]


def fold(text):
    return "".join(c for c in unicodedata.normalize("NFKD", text.lower()) if not unicodedata.combining(c)).strip(" ?!.")


def is_romanian(text):
    from alexgpt.tools import is_romanian as vote
    return vote(text)


def main():
    from alexgpt.identity_rag import read_groups
    known_identity = {fold(q) for group in read_groups().values() for q in group["questions"]}
    items = []

    def add(category, kind, turns, expect=None, min_matches=1, lang=None):
        number = sum(1 for item in items if item["category"] == category)
        items.append({"id": f"{category}_{number + 1:02d}", "category": category, "kind": kind, "turns": turns,
                      "expect": expect, "min_matches": min_matches,
                      "lang": lang or ("ro" if is_romanian(turns[-1]) else "en"),
                      "split": "heldout" if number % 4 == 3 else "dev"})

    for question, expect in FACTS_EN:
        add("facts_en", "fact", [question], expect)
    for question, expect in FACTS_RO:
        add("facts_ro", "fact", [question], expect, lang="ro")
    for question, expect in FACTS_HARD:
        add("facts_hard", "fact", [question], expect)
    for question in FALSE_PREMISE:
        add("false_premise", "false_premise", [question])
    for turns, expect in FOLLOWUP:
        add("followup", "fact", turns, expect)
    for topics in STYLE_TOPICS:
        for final, expect in STYLE_FINALS:
            add("style_reset", "style", [f"tell me about {topic}" for topic in topics] + [final], expect)
    dropped = 0
    for question, expect in IDENTITY:
        if fold(question) in known_identity:            # must be NEW wordings: the RAG file must not contain them
            dropped += 1
            continue
        add("identity", "identity", [question], expect)
    for question, lang in CHITCHAT:
        add("chitchat", "chat", [question], lang=lang)
    for question, expect, min_matches in LISTS:
        add("superlatives_lists", "list" if min_matches > 1 else "fact", [question], expect, min_matches)
    for question, expect in TOOLS:
        add("tools", "tool", [question], expect)

    path = os.path.join(HERE, "failures300.jsonl")
    with open(path, "w", encoding="utf-8") as file:
        for item in items:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")
    heldout = sum(item["split"] == "heldout" for item in items)
    print(f"{len(items)} items ({heldout} held out) -> {path}.  identity questions dropped (already in identity_qa.txt): {dropped}")


if __name__ == "__main__":
    main()
