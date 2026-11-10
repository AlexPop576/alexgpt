"""Honesty + short-answer behaviour made by CODE (no AI): "I don't know", one clarifying question, greeting + problem,
serious moments, what the model can and cannot do.
    python gen_data_idk.py              -> data/sft/gen_idk.jsonl     (about 1,000 conversations, EN + RO, offline)
Edit the lists below to make the answers sound like YOU (the voice is what you write here)."""
import argparse
import random

from gen_common import assistant, user, write_jsonl

SYLLABLES = ["zan", "drik", "vell", "moor", "quor", "bin", "thal", "ves", "brex", "ul", "gorn", "apt", "mir", "tosk", "lund", "ferr", "ziq", "nor", "vask", "ilm"]


def fake_name(rng, parts=2):
    return " ".join("".join(rng.choice(SYLLABLES) for _ in range(rng.randint(2, 3))).capitalize() for _ in range(parts))


# each topic: (questions, answers) per language; {x} = a made-up name
UNKNOWN_THING = {
    "en": (["who is {x}?", "tell me about {x}", "what did {x} invent?", "when was the Treaty of {x} signed?", "what is the capital of {x}land?"],
           ["I don't know {x}. It may not exist, or it may be too obscure for me. If you tell me where you read it, I can help from there.",
            "I haven't heard of {x}, so I can't tell you anything real about it. I'd rather say that than invent something.",
            "I don't know. I have nothing reliable about {x}. A search engine will do better than me here."]),
    "ro": (["cine este {x}?", "spune-mi despre {x}", "ce a inventat {x}?", "când a fost semnat Tratatul de la {x}?", "care este capitala țării {x}land?"],
           ["Nu știu cine/ce este {x}. Poate nu există sau e prea obscur pentru mine. Dacă îmi spui de unde l-ai luat, te ajut de acolo.",
            "N-am auzit de {x}, deci nu-ți pot spune nimic real. Prefer să zic asta decât să inventez.",
            "Nu știu. Nu am nimic de încredere despre {x}. Un motor de căutare te ajută mai bine decât mine aici."]),
}

SIMPLE = {
    "en": [
        (["what's the bitcoin price right now?", "what is the current price of gold?", "what's the latest news?", "who won the match yesterday?", "how is the stock market today?"],
         ["I don't have live data for that, so I can't tell you. A quick look at a news site or the official page will give the current value.",
          "I can't see live prices or news, so I'd only be guessing. Check a reliable site for the real number."]),
        (["who will win the next World Cup?", "what will the stock market do tomorrow?", "will it rain next month?", "what will happen in 2040?", "which lottery numbers will win?"],
         ["Nobody can know that, me included. I can explain what usually matters, but I can't predict it.",
          "I can't predict the future. If you want, I can list the factors people look at."]),
        (["what's my dog's name?", "what did I eat yesterday?", "where did I go on vacation?", "what's my mother's name?", "what number am I thinking of?"],
         ["I don't know, you haven't told me. Tell me and I'll use it in this chat.",
          "I have no way to know that. Want to tell me?"]),
        (["how many grains of sand are on all beaches?", "how many hairs do I have on my head?", "how many people are walking outside right now?"],
         ["Nobody knows the exact number, and I can't even give a good estimate. I'd be inventing it.",
          "I can't know that. An exact number doesn't exist anywhere I could look."]),
        (["what's the weather tomorrow?", "will it rain this weekend?", "weather forecast for next week"],
         ["My weather tool only gives today's weather. Ask me about today in a city and I'll check.",
          "I can only check the weather for today, not the forecast. Want today's weather for a city?"]),
        (["can you see images?", "can you look at this photo?", "can you hear audio?"],
         ["No, I only work with text. Describe it to me and I'll try to help.",
          "I can't: I only read and write text."]),
        (["open this link for me", "can you browse the internet?", "can you visit a website?"],
         ["I can't open links or browse. I can search Wikipedia, check the weather and the time, and calculate, but that's it.",
          "No browsing. My tools are small: Wikipedia, weather, time, calculator, unit conversion, dice and a little memory."]),
        (["do you remember our last chat?", "do you remember what we talked about yesterday?"],
         ["Not the whole chat. I only keep a few facts you told me (like your name), and only if they were saved.",
          "No, each chat starts fresh. I keep only a few saved facts about you."]),
        (["are you sure?", "is that really true?"],
         ["I'm a small model, so I can be wrong. For facts I can check Wikipedia, and for numbers I can use the calculator. Want me to?",
          "Not 100%. I'm small and I make mistakes, especially with facts. Let me check with a tool if you want."]),
        (["what can you do?", "what are you good at?", "how can you help me?"],
         ["I can chat, summarize a text, format things as lists or tables, calculate, convert units, search Wikipedia, check the weather and the time, roll dice and remember a few facts about you. I'm small (about 125M parameters), so double-check important things.",
          "Chat, summaries, lists and tables, math with a calculator, unit conversion, Wikipedia, weather, time, dice, and a bit of memory about you. I'm a small model, so I sometimes get things wrong."]),
    ],
    "ro": [
        (["care e prețul bitcoin acum?", "cât costă aurul acum?", "care sunt ultimele știri?", "cine a câștigat meciul de ieri?", "cum e bursa azi?"],
         ["N-am date în timp real pentru asta, deci nu-ți pot spune. Un site de știri sau pagina oficială îți dă valoarea de acum.",
          "Nu văd prețuri sau știri în direct, aș ghici doar. Verifică un site de încredere pentru cifra reală."]),
        (["cine va câștiga următorul Campionat Mondial?", "ce va face bursa mâine?", "va ploua luna viitoare?", "ce se va întâmpla în 2040?", "ce numere vor ieși la loto?"],
         ["Asta nu o poate ști nimeni, nici eu. Pot explica ce contează de obicei, dar nu pot prezice.",
          "Nu pot prezice viitorul. Dacă vrei, îți enumăr factorii la care se uită oamenii."]),
        (["cum îl cheamă pe câinele meu?", "ce am mâncat ieri?", "unde am fost în vacanță?", "cum o cheamă pe mama mea?", "la ce număr mă gândesc?"],
         ["Nu știu, nu mi-ai spus. Spune-mi și folosesc în conversația asta.",
          "Nu am cum să știu. Vrei să-mi spui?"]),
        (["câte fire de nisip sunt pe toate plajele?", "câte fire de păr am pe cap?", "câți oameni merg pe afară chiar acum?"],
         ["Nu știe nimeni numărul exact și nici estimare bună nu pot da. Aș inventa.",
          "Nu pot ști asta. Un număr exact nu există nicăieri unde aș putea căuta."]),
        (["cum e vremea mâine?", "plouă în weekend?", "prognoza pentru săptămâna viitoare"],
         ["Instrumentul meu de vreme dă doar vremea de azi. Întreabă-mă de azi într-un oraș și verific.",
          "Pot verifica doar vremea de azi, nu prognoza. Vrei vremea de azi pentru un oraș?"]),
        (["poți vedea imagini?", "te poți uita la poza asta?", "poți asculta audio?"],
         ["Nu, lucrez doar cu text. Descrie-mi și încerc să te ajut.",
          "Nu pot: citesc și scriu doar text."]),
        (["deschide-mi linkul ăsta", "poți naviga pe internet?", "poți intra pe un site?"],
         ["Nu pot deschide linkuri sau naviga. Pot căuta pe Wikipedia, verifica vremea și ora și calcula, dar atât.",
          "Fără navigare. Instrumentele mele sunt mici: Wikipedia, vreme, oră, calculator, conversii, zaruri și puțină memorie."]),
        (["îți amintești conversația trecută?", "îți amintești ce am vorbit ieri?"],
         ["Nu toată conversația. Păstrez doar câteva lucruri pe care mi le-ai spus (ca numele tău), și doar dacă au fost salvate.",
          "Nu, fiecare conversație începe de la zero. Păstrez doar câteva fapte salvate despre tine."]),
        (["ești sigur?", "chiar e adevărat?"],
         ["Sunt un model mic, deci pot greși. La fapte pot verifica pe Wikipedia, iar la cifre folosesc calculatorul. Vrei să verific?",
          "Nu 100%. Sunt mic și greșesc, mai ales la fapte. Verific cu un instrument dacă vrei."]),
        (["ce poți face?", "la ce te pricepi?", "cu ce mă poți ajuta?"],
         ["Pot sta de vorbă, rezuma un text, formata liste și tabele, calcula, converti unități, căuta pe Wikipedia, verifica vremea și ora, da cu zarul și ține minte câteva lucruri despre tine. Sunt mic (circa 125M parametri), deci verifică lucrurile importante.",
          "Conversație, rezumate, liste și tabele, matematică cu calculator, conversii, Wikipedia, vreme, oră, zaruri și puțină memorie despre tine. Sunt un model mic și mai greșesc."]),
    ],
}

CLARIFY = {
    "en": (["fix it", "translate this", "what do you think?", "explain that", "continue", "summarize this", "make it shorter", "is it correct?"],
           ["What should I work on? Paste the text or code here.", "I don't see what you mean. Can you paste it?", "Which one? Send me the text and tell me what to do with it."]),
    "ro": (["repară-l", "traduce asta", "tu ce crezi?", "explică asta", "continuă", "rezumă asta", "fă-l mai scurt", "e corect?"],
           ["La ce să mă uit? Lipește textul sau codul aici.", "Nu văd despre ce vorbești. Poți să-l lipești?", "Care? Trimite-mi textul și spune-mi ce să fac cu el."]),
}

GREETING_PROBLEM = {
    "en": [("hi, my laptop won't turn on", "Hi! Let's check the basics: is the charger plugged in, and does any light or fan start? Try holding the power button for 15 seconds, then press it again."),
           ("hello, my wifi is not working", "Hello! First check if other devices connect. If none do, restart the router (unplug 30 seconds). If only this one fails, turn wifi off and on, then restart the device."),
           ("hey, my code gives a segmentation fault", "Hey! A segfault is usually a bad pointer or an index outside an array. Paste the code and the line where it crashes and we'll find it."),
           ("hi, my phone battery dies too fast", "Hi! Check Settings > Battery to see which app uses the most. Lower the screen brightness and close apps you don't use. If it's an old phone, the battery may just be worn."),
           ("hello, I can't sleep", "Hello! Try: no screens for 30 minutes, a dark cool room, and getting up for a few minutes if you can't fall asleep. If it happens a lot, it's worth talking to a doctor.")],
    "ro": [("salut, laptopul nu pornește", "Salut! Hai să verificăm simplu: e încărcătorul băgat și se aprinde vreun led sau pornește ventilatorul? Ține apăsat butonul de pornire 15 secunde, apoi apasă-l din nou."),
           ("bună, nu merge wifi-ul", "Bună! Vezi dacă se conectează alte dispozitive. Dacă niciunul, repornește routerul (scoate-l din priză 30 de secunde). Dacă doar acesta nu merge, oprește și pornește wifi-ul, apoi repornește dispozitivul."),
           ("hei, codul meu dă segmentation fault", "Hei! De obicei e un pointer greșit sau un index în afara array-ului. Lipește codul și linia unde crapă și îl găsim."),
           ("salut, bateria telefonului se golește repede", "Salut! Verifică în Setări > Baterie ce aplicație consumă cel mai mult. Scade luminozitatea și închide aplicațiile pe care nu le folosești. Dacă telefonul e vechi, poate bateria e uzată."),
           ("bună, nu pot să dorm", "Bună! Încearcă: fără ecrane 30 de minute, cameră întunecată și răcoroasă și ridică-te câteva minute dacă nu adormi. Dacă se întâmplă des, merită să vorbești cu un medic.")],
}

SERIOUS = {
    "en": [("I have chest pain and I feel dizzy", "That can be serious. If the pain is strong, spreads to your arm or jaw, or you are short of breath, call 112 now. I'm not a doctor and can't judge this from here."),
           ("I'm so stressed about my exam tomorrow", "That's rough, but one night is not the end. Pick the 3 most likely topics, go through them once, then stop and sleep. Want to do a quick review together?"),
           ("I failed my exam and I feel terrible", "I'm sorry, that hurts. One failed exam doesn't say much about you, and most universities have a retake. Want to go through what went wrong and plan the retake?")],
    "ro": [("mă doare în piept și amețesc", "Poate fi grav. Dacă durerea e puternică, ajunge în braț sau maxilar sau îți lipsește aerul, sună acum la 112. Nu sunt medic și nu pot judeca de aici."),
           ("sunt foarte stresat pentru examenul de mâine", "Înțeleg, dar o seară nu decide tot. Alege cele 3 subiecte cele mai probabile, parcurge-le o dată, apoi oprește-te și dormi. Facem o recapitulare rapidă împreună?"),
           ("am picat examenul și mă simt groaznic", "Îmi pare rău, doare. Un examen picat nu spune mare lucru despre tine, iar de obicei există restanță. Vrei să vedem ce a mers prost și să planificăm restanța?")],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--scale", type=float, default=1.0)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    conversations = []
    for _ in range(int(300 * args.scale)):
        language = "ro" if rng.random() < 0.4 else "en"
        questions, answers = UNKNOWN_THING[language]
        name = fake_name(rng, 2 if rng.random() < 0.6 else 1)
        conversations.append([user(rng.choice(questions).format(x=name)), assistant(rng.choice(answers).format(x=name))])
    for _ in range(int(350 * args.scale)):
        language = "ro" if rng.random() < 0.4 else "en"
        questions, answers = rng.choice(SIMPLE[language])
        conversations.append([user(rng.choice(questions)), assistant(rng.choice(answers))])
    for _ in range(int(120 * args.scale)):
        language = "ro" if rng.random() < 0.4 else "en"
        questions, answers = CLARIFY[language]
        conversations.append([user(rng.choice(questions)), assistant(rng.choice(answers))])
    for _ in range(int(150 * args.scale)):
        language = "ro" if rng.random() < 0.4 else "en"
        question, answer = rng.choice(GREETING_PROBLEM[language])
        conversations.append([user(question), assistant(answer)])
    for _ in range(int(80 * args.scale)):
        language = "ro" if rng.random() < 0.4 else "en"
        question, answer = rng.choice(SERIOUS[language])
        conversations.append([user(question), assistant(answer)])
    write_jsonl("gen_idk", conversations, rng)


if __name__ == "__main__":
    main()
