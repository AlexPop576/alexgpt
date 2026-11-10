"""Tool-call conversations made by CODE (no AI): every result is computed here, so every answer is correct.
    python gen_data_tools.py            -> data/sft/gen_tools.jsonl     (about 4,500 conversations, EN + RO, offline, ~1 s)
The shape of every call and result must match alexgpt/tools.py (read its docstrings).
Tool kinds: calc, convert, now, weather, dice, coin, pick, memory_save, memory (answers from memory / empty memory)."""
import argparse
import datetime
import random

from alexgpt.sft_format import system_prompt
from gen_common import (assistant, context, format_number, maybe_small_talk, system, tool_call, user, write_jsonl)


# ====================================================================== calc
OPERATOR_WORDS = {"en": {"+": "plus", "-": "minus", "*": "times", "/": "divided by"},
                  "ro": {"+": "plus", "-": "minus", "*": "ori", "/": "împărțit la"}}


def calc_case(rng, language):
    """-> (user text, call text, result text, answer text)"""
    kind = rng.choice(["binary"] * 6 + ["percent", "percent", "sqrt", "power", "paren", "paren", "shop", "split", "squared", "squared", "story", "story", "speed", "speed"])
    en = language == "en"
    if kind == "binary":
        operator = rng.choice(["+", "-", "*", "*", "/"])
        if operator in "+-":
            a, b = rng.randint(10, 9999), rng.randint(10, 9999)
            value = a + b if operator == "+" else a - b
        elif operator == "*":
            a, b = rng.randint(11, 999), rng.randint(2, 99)
            value = a * b
        else:
            b = rng.randint(2, 50)
            a = b * rng.randint(2, 400) if rng.random() < 0.7 else rng.randint(10, 999)
            value = a / b
        expression = f"{a} {operator} {b}"
        shown = rng.choice([operator, "x"]) if operator == "*" else operator
        word = OPERATOR_WORDS[language][operator]
        templates = ([f"{a} {shown} {b}", f"what is {a} {shown} {b}?", f"calculate {a} {shown} {b}", f"how much is {a} {word} {b}?",
                      f"compute {a} {shown} {b}", f"{a} {shown} {b} = ?", f"What is {a} {shown} {b}?", f"What is {a} {shown} {b}?",
                      f"Calculate {a} {shown} {b}.", f"How much is {a} {shown} {b}?"] if en else
                     [f"{a} {shown} {b}", f"cât face {a} {shown} {b}?", f"calculează {a} {shown} {b}", f"cât e {a} {word} {b}?",
                      f"{a} {shown} {b} = ?", f"care este rezultatul pentru {a} {shown} {b}?", f"Cât face {a} {shown} {b}?", f"Cât face {a} {shown} {b}?"])
        return rng.choice(templates), "calc: " + expression, format_number(value), answer_for(rng, language, expression, value)
    if kind == "squared":
        base = rng.randint(3, 40)
        text = (rng.choice([f"What is {base} squared?", f"{base} squared", f"what is {base} squared?"]) if en else
                rng.choice([f"Cât face {base} la pătrat?", f"{base} la pătrat", f"Cât este {base} la pătrat?"]))
        return text, f"calc: {base} ** 2", str(base * base), (f"{base} squared is {base * base}." if en else f"{base} la pătrat este {base * base}.")
    if kind == "story":
        have, bought, eaten = rng.randint(2, 30), rng.randint(2, 20), rng.randint(1, 9)
        eaten = min(eaten, have + bought - 1)
        thing_en, thing_ro = rng.choice([("apples", "mere"), ("pencils", "creioane"), ("stickers", "abțibilduri"), ("books", "cărți"), ("coins", "monede")])
        value = have + bought - eaten
        verb_en = "eat" if thing_en == "apples" else "lose"
        verb_ro = "mănânc" if thing_en == "apples" else "pierd"
        text = (f"I have {have} {thing_en}, buy {bought} more and then {verb_en} {eaten}. How many {thing_en} do I have now?" if en else
                f"Am {have} {thing_ro}, mai cumpăr {bought} și apoi {verb_ro} {eaten}. Câte {thing_ro} am acum?")
        return text, f"calc: {have} + {bought} - {eaten}", str(value), (f"You have {value} {thing_en} now." if en else f"Ai {value} {thing_ro} acum.")
    if kind == "speed":
        speed, hours = rng.choice([40, 50, 60, 70, 80, 90, 100, 120]), rng.choice([1.5, 2, 2.5, 3, 3.5, 4, 5])
        value = speed * hours
        hours_text = format_number(hours)
        text = (rng.choice([f"A car travels at {speed} km per hour for {hours_text} hours. How many kilometres does it travel?",
                            f"A bus drives {hours_text} hours at {speed} km/h. How far does it go?"]) if en else
                rng.choice([f"O mașină merge cu {speed} km pe oră timp de {hours_text} ore. Câți kilometri parcurge?",
                            f"Un autobuz merge {hours_text} ore cu {speed} km/h. Ce distanță parcurge?"]))
        return text, f"calc: {speed} * {hours_text}", format_number(value), (f"It travels {format_number(value)} km." if en else f"Parcurge {format_number(value)} km.")
    if kind == "percent":
        percent = rng.choice([5, 8, 10, 12, 15, 20, 25, 30, 40, 50, 75])
        base = rng.randint(2, 200) * rng.choice([10, 10, 20, 5])
        value = percent * base / 100
        text = (rng.choice([f"what is {percent}% of {base}?", f"{percent}% of {base}", f"how much is {percent} percent of {base}?"]) if en else
                rng.choice([f"cât e {percent}% din {base}?", f"{percent}% din {base}", f"cât reprezintă {percent} la sută din {base}?"]))
        expression = f"{percent}% of {base}"
        result = format_number(value)
        return text, "calc: " + expression, result, (f"{percent}% of {base} is {result}." if en else f"{percent}% din {base} este {result}.")
    if kind == "sqrt":
        root = rng.randint(2, 40)
        text = (rng.choice([f"square root of {root * root}", f"what is the square root of {root * root}?", f"sqrt({root * root})"]) if en else
                rng.choice([f"radical din {root * root}", f"cât este rădăcina pătrată din {root * root}?", f"sqrt({root * root})"]))
        return text, f"calc: sqrt({root * root})", str(root), (f"The square root of {root * root} is {root}." if en else f"Rădăcina pătrată din {root * root} este {root}.")
    if kind == "power":
        base, exponent = rng.randint(2, 12), rng.randint(2, 8)
        value = base ** exponent
        if value > 5_000_000:
            exponent = 3
            value = base ** exponent
        text = (rng.choice([f"{base} to the power of {exponent}", f"what is {base}^{exponent}?"]) if en else
                rng.choice([f"{base} la puterea {exponent}", f"cât face {base}^{exponent}?"]))
        return text, f"calc: {base} ** {exponent}", str(value), (f"{base}^{exponent} = {value}." if en else f"{base}^{exponent} = {value}.")
    if kind == "paren":
        a, b, c = rng.randint(2, 99), rng.randint(2, 99), rng.randint(2, 30)
        expression = f"({a} + {b}) * {c}"
        value = (a + b) * c
        text = rng.choice([f"calculate {expression}", f"what is {expression}?"]) if en else rng.choice([f"calculează {expression}", f"cât face {expression}?"])
        return text, "calc: " + expression, str(value), answer_for(rng, language, expression, value)
    if kind == "shop":
        count, price = rng.randint(2, 40), rng.randint(2, 150)
        item_en, item_ro, unit_en, unit_ro = rng.choice([("notebooks", "caiete", "lei", "lei"), ("coffees", "cafele", "lei", "lei"),
                                                         ("tickets", "bilete", "euros", "euro"), ("pens", "pixuri", "lei", "lei")])
        value = count * price
        text = (f"I buy {count} {item_en} at {price} {unit_en} each. How much do I pay?" if en else
                f"Cumpăr {count} {item_ro} a câte {price} {unit_ro} bucata. Cât plătesc în total?")
        return text, f"calc: {count} * {price}", str(value), (f"You pay {value} {unit_en} in total." if en else f"Plătești {value} {unit_ro} în total.")
    # split a bill
    people = rng.randint(2, 9)
    total = people * rng.randint(10, 120) if rng.random() < 0.7 else rng.randint(50, 900)
    value = total / people
    result = format_number(value)
    nice = format_number(value, 2)
    exact = result == nice
    text = (f"The bill is {total} lei and we are {people} people. How much does each pay?" if en else
            f"Nota de plată este {total} lei și suntem {people} persoane. Cât plătește fiecare?")
    if en:
        answer = f"Each pays {nice} lei." if exact else f"Each pays about {nice} lei."
    else:
        answer = f"Fiecare plătește {nice} lei." if exact else f"Fiecare plătește aproximativ {nice} lei."
    return text, f"calc: {total} / {people}", result, answer


def answer_for(rng, language, expression, value):
    result = format_number(value)
    long_decimal = "." in result and len(result.split(".")[1]) > 4
    shown = format_number(value, 4) if long_decimal else result
    if language == "en":
        return rng.choice([f"{expression} = {shown}.", f"That's {shown}.", f"The answer is {shown}."]) if not long_decimal else \
               rng.choice([f"{expression} ≈ {shown}.", f"That's about {shown}."])
    return rng.choice([f"{expression} = {shown}.", f"Face {shown}.", f"Rezultatul este {shown}."]) if not long_decimal else \
           rng.choice([f"{expression} ≈ {shown}.", f"Face aproximativ {shown}."])


def calc_conversation(rng, language):
    if rng.random() < 0.02:
        a = rng.randint(2, 900)
        text = f"{a} / 0" if language == "en" else rng.choice([f"{a} / 0", f"cât face {a} împărțit la 0?"])
        answer = ("You can't divide by zero: it's undefined." if language == "en" else "Nu se poate împărți la zero: nu este definit.")
        return [user(text), tool_call(f"calc: {a} / 0"), context("error: division by zero"), assistant(answer)]
    text, call, result, answer = calc_case(rng, language)
    return maybe_small_talk(rng, language, [user(text), tool_call(call), context(result), assistant(answer)])


# ====================================================================== convert
# unit -> (kind, factor to the base unit);  temperature is handled separately
UNITS = {
    "km": ("length", 1000), "m": ("length", 1), "cm": ("length", 0.01), "mm": ("length", 0.001), "miles": ("length", 1609.344),
    "feet": ("length", 0.3048), "inches": ("length", 0.0254),
    "kg": ("mass", 1), "g": ("mass", 0.001), "pounds": ("mass", 0.45359237), "ounces": ("mass", 0.028349523125),
    "liters": ("volume", 1), "ml": ("volume", 0.001), "gallons": ("volume", 3.785411784),
    "km/h": ("speed", 1 / 3.6), "mph": ("speed", 0.44704), "m/s": ("speed", 1),
    "hours": ("time", 3600), "minutes": ("time", 60), "seconds": ("time", 1),
    "celsius": ("temperature", None), "fahrenheit": ("temperature", None), "kelvin": ("temperature", None),
}
WORDS = {
    "en": {"km": ["km", "kilometers"], "m": ["meters", "m"], "cm": ["cm", "centimeters"], "mm": ["mm", "millimeters"],
           "miles": ["miles"], "feet": ["feet", "ft"], "inches": ["inches"], "kg": ["kg", "kilograms"], "g": ["grams", "g"],
           "pounds": ["pounds", "lbs"], "ounces": ["ounces", "oz"], "liters": ["liters", "l"], "ml": ["ml", "milliliters"],
           "gallons": ["gallons"], "km/h": ["km/h"], "mph": ["mph"], "m/s": ["m/s"], "hours": ["hours"], "minutes": ["minutes"],
           "seconds": ["seconds"], "celsius": ["celsius", "°C"], "fahrenheit": ["fahrenheit", "°F"], "kelvin": ["kelvin"]},
    "ro": {"km": ["km", "kilometri"], "m": ["metri", "m"], "cm": ["cm", "centimetri"], "mm": ["mm", "milimetri"],
           "miles": ["mile"], "feet": ["picioare", "ft"], "inches": ["inci"], "kg": ["kg", "kilograme"], "g": ["grame", "g"],
           "pounds": ["livre", "lbs"], "ounces": ["uncii", "oz"], "liters": ["litri", "l"], "ml": ["ml", "mililitri"],
           "gallons": ["galoane"], "km/h": ["km/h"], "mph": ["mph"], "m/s": ["m/s"], "hours": ["ore"], "minutes": ["minute"],
           "seconds": ["secunde"], "celsius": ["grade Celsius", "°C"], "fahrenheit": ["grade Fahrenheit", "°F"], "kelvin": ["kelvin"]},
}


def convert_value(value, source, target):
    if UNITS[source][0] == "temperature":
        celsius = value if source == "celsius" else (value - 32) * 5 / 9 if source == "fahrenheit" else value - 273.15
        return celsius if target == "celsius" else celsius * 9 / 5 + 32 if target == "fahrenheit" else celsius + 273.15
    return value * UNITS[source][1] / UNITS[target][1]


def convert_conversation(rng, language):
    kind = rng.choice(["length", "length", "mass", "volume", "speed", "time", "temperature", "temperature"])
    source, target = rng.sample([unit for unit in UNITS if UNITS[unit][0] == kind], 2)
    if kind == "temperature":
        value = rng.choice([0, 10, 15, 20, 25, 30, 37, 40, 100, -5, -10, 50, 72, 98.6, 212])
    else:
        value = rng.choice([1, 2, 3, 5, 10, 12, 15, 20, 25, 30, 42, 50, 100, 150, 200, 250, 500, 1000, 2.5, 0.5, 7.5])
    result_value = round(convert_value(value, source, target), 3)
    result_text = format_number(result_value, 3)
    value_text = format_number(value, 3)
    result = f"{value_text} {source} = {result_text} {target}"
    source_word, target_word = rng.choice(WORDS[language][source]), rng.choice(WORDS[language][target])
    if language == "en":
        text = rng.choice([f"convert {value_text} {source_word} to {target_word}", f"how many {target_word} is {value_text} {source_word}?",
                           f"{value_text} {source_word} in {target_word}"])
        approx = "" if float(result_text) == convert_value(value, source, target) else "about "
        answer = rng.choice([f"{value_text} {source_word} is {approx}{result_text} {target_word}.", f"{value_text} {source_word} = {approx}{result_text} {target_word}."])
    else:
        text = rng.choice([f"convertește {value_text} {source_word} în {target_word}", f"câți {target_word} sunt {value_text} {source_word}?",
                           f"{value_text} {source_word} în {target_word}"])
        approx = "" if float(result_text) == convert_value(value, source, target) else "aproximativ "
        answer = rng.choice([f"{value_text} {source_word} înseamnă {approx}{result_text} {target_word}.", f"{value_text} {source_word} = {approx}{result_text} {target_word}."])
    return maybe_small_talk(rng, language, [user(text), tool_call(f"convert: {value_text} {source} to {target}"), context(result), assistant(answer)])


# ====================================================================== now / weather
CITIES = [("Bucharest", "București", "Europe/Bucharest"), ("Cluj-Napoca", "Cluj-Napoca", "Europe/Bucharest"),
          ("London", "Londra", "Europe/London"), ("New York", "New York", "America/New_York"), ("Tokyo", "Tokyo", "Asia/Tokyo"),
          ("Paris", "Paris", "Europe/Paris"), ("Berlin", "Berlin", "Europe/Berlin"), ("Los Angeles", "Los Angeles", "America/Los_Angeles"),
          ("Sydney", "Sydney", "Australia/Sydney"), ("Dubai", "Dubai", "Asia/Dubai"), ("Madrid", "Madrid", "Europe/Madrid"),
          ("Rome", "Roma", "Europe/Rome"), ("Vienna", "Viena", "Europe/Vienna"), ("Chicago", "Chicago", "America/Chicago")]
WEATHER_ONLY = [("Timișoara", "Timișoara"), ("Iași", "Iași"), ("Brașov", "Brașov"), ("Constanța", "Constanța"), ("Oradea", "Oradea"),
                ("Sibiu", "Sibiu"), ("Budapest", "Budapesta"), ("Prague", "Praga"), ("Lisbon", "Lisabona"), ("Oslo", "Oslo"),
                ("Athens", "Atena"), ("Toronto", "Toronto"), ("Singapore", "Singapore"), ("Cairo", "Cairo")]
DAYS = {"en": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
        "ro": ["luni", "marți", "miercuri", "joi", "vineri", "sâmbătă", "duminică"]}
MONTHS = {"en": ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"],
          "ro": ["ianuarie", "februarie", "martie", "aprilie", "mai", "iunie", "iulie", "august", "septembrie", "octombrie", "noiembrie", "decembrie"]}


def now_conversation(rng, language):
    english_city, romanian_city, zone = rng.choice(CITIES)
    if rng.random() < 0.5:
        english_city, romanian_city, zone = CITIES[0]
        city_in_question = False
    else:
        city_in_question = True
    moment = datetime.datetime(2026, 1, 1) + datetime.timedelta(minutes=rng.randint(0, 60 * 24 * 700))
    day_en, month_en = DAYS["en"][moment.weekday()], MONTHS["en"][moment.month - 1]
    result = f"{day_en}, {moment.day} {month_en} {moment.year}, {moment:%H:%M} ({zone})"
    clock = f"{moment:%H:%M}"
    kind = rng.choice(["time", "time", "date", "day"])
    place = english_city if language == "en" else romanian_city
    if language == "en":
        if kind == "time":
            text = f"what time is it in {place}?" if city_in_question else rng.choice(["what time is it?", "what's the time?", "time?"])
            answer = f"It's {clock}" + (f" in {place}." if city_in_question else ".")
        elif kind == "date":
            text = rng.choice(["what's the date today?", "what is today's date?"]) if not city_in_question else f"what's the date in {place}?"
            answer = f"Today is {day_en}, {moment.day} {month_en} {moment.year}."
        else:
            text = rng.choice(["what day is it?", "what day of the week is it today?"]) if not city_in_question else f"what day is it in {place}?"
            answer = f"It's {day_en}."
    else:
        day_ro, month_ro = DAYS["ro"][moment.weekday()], MONTHS["ro"][moment.month - 1]
        if kind == "time":
            text = f"ce oră este în {place}?" if city_in_question else rng.choice(["ce oră e?", "cât este ceasul?", "ce ora e acum?"])
            answer = f"Este ora {clock}" + (f" în {place}." if city_in_question else ".")
        elif kind == "date":
            text = rng.choice(["ce dată e azi?", "ce dată avem astăzi?"]) if not city_in_question else f"ce dată este în {place}?"
            answer = f"Azi este {day_ro}, {moment.day} {month_ro} {moment.year}."
        else:
            text = rng.choice(["ce zi e azi?", "în ce zi a săptămânii suntem?"]) if not city_in_question else f"ce zi este în {place}?"
            answer = f"Este {day_ro}."
    return maybe_small_talk(rng, language, [user(text), tool_call(f"now: {zone}"), context(result), assistant(answer)])


CONDITIONS = [("clear sky", "cer senin", False), ("partly cloudy", "parțial noros", False), ("overcast", "înnorat", False),
              ("fog", "ceață", False), ("light rain", "ploaie ușoară", True), ("rain", "ploaie", True),
              ("heavy rain", "ploaie torențială", True), ("snow", "ninsoare", True), ("thunderstorm", "furtună", True)]


def weather_conversation(rng, language):
    english_city, romanian_city = rng.choice([(c[0], c[1]) for c in CITIES] + WEATHER_ONLY)
    place = english_city if language == "en" else romanian_city
    condition_en, condition_ro, rainy = rng.choice(CONDITIONS)
    if condition_en == "snow":
        current = rng.randint(-8, 1)
    else:
        current = rng.choice([rng.randint(-3, 12), rng.randint(8, 25), rng.randint(15, 33)])
    low, high = current - rng.randint(2, 8), current + rng.randint(1, 7)
    rain_chance = rng.choice([60, 70, 80, 90, 100]) if rainy else rng.choice([0, 0, 5, 10, 20, 30])
    wind = rng.randint(3, 40)
    result = f"{english_city}: {current}°C, {condition_en}, wind {wind} km/h; today min {low}°C, max {high}°C, rain chance {rain_chance}%"
    umbrella_question = rng.random() < 0.3
    if language == "en":
        text = (rng.choice([f"do I need an umbrella in {place}?", f"will it rain in {place} today?"]) if umbrella_question else
                rng.choice([f"weather in {place}", f"what's the weather like in {place}?", f"how's the weather in {place}?"]))
        advice = (" Take an umbrella." if rain_chance >= 50 else " Dress warm." if current <= 5 else " It's hot, drink water." if current >= 30 else "")
        summary = f"In {place} it's {current}°C, {condition_en}, wind {wind} km/h. Today between {low}°C and {high}°C, rain chance {rain_chance}%.{advice}"
        if umbrella_question:
            summary = (f"Yes, take one: the rain chance in {place} is {rain_chance}%." if rain_chance >= 50 else
                       f"Probably not: the rain chance in {place} is only {rain_chance}%.")
    else:
        text = (rng.choice([f"iau umbrelă în {place}?", f"plouă azi în {place}?"]) if umbrella_question else
                rng.choice([f"vremea în {place}", f"ce vreme e în {place}?", f"cum e vremea în {place}?"]))
        advice = (" Ia o umbrelă." if rain_chance >= 50 else " Îmbracă-te gros." if current <= 5 else " E cald, bea apă." if current >= 30 else "")
        summary = f"În {place} sunt {current}°C, {condition_ro}, vânt de {wind} km/h. Azi între {low}°C și {high}°C, șanse de ploaie {rain_chance}%.{advice}"
        if umbrella_question:
            summary = (f"Da, ia una: șansele de ploaie în {place} sunt {rain_chance}%." if rain_chance >= 50 else
                       f"Probabil nu: șansele de ploaie în {place} sunt doar {rain_chance}%.")
    roll = rng.random()
    if roll < 0.03:
        fake = rng.choice(["Qzxvland", "Brelmoria", "Xanthopolis", "Vrenkaville"])
        error = f"error: city '{fake}' not found"
        reply = (f"I couldn't find a city called {fake}. Check the spelling?" if language == "en" else f"Nu am găsit niciun oraș numit {fake}. Verifici ortografia?")
        return [user(f"weather in {fake}" if language == "en" else f"vremea în {fake}"), tool_call(f"weather: {fake}"), context(error), assistant(reply)]
    if roll < 0.10:   # no city in the question: ask first
        ask = ("Which city?" if language == "en" else "În ce oraș?")
        first = rng.choice(["what's the weather?", "how's the weather today?"]) if language == "en" else rng.choice(["cum e vremea?", "ce vreme e azi?"])
        return [user(first), assistant(ask), user(place), tool_call(f"weather: {english_city}"), context(result), assistant(summary)]
    return maybe_small_talk(rng, language, [user(text), tool_call(f"weather: {english_city}"), context(result), assistant(summary)])


# ====================================================================== dice / coin / pick
def dice_conversation(rng, language):
    count = rng.choice([1, 1, 1, 2, 2, 3, 4])
    sides = rng.choice([6, 6, 6, 20, 20, 4, 8, 10, 12, 100])
    rolls = [rng.randint(1, sides) for _ in range(count)]
    spec = f"{count}d{sides}"
    result = f"{spec}: {rolls[0]}" if count == 1 else f"{spec}: {' + '.join(map(str, rolls))} = {sum(rolls)}"
    if language == "en":
        text = (rng.choice(["roll a die", "roll a dice", "throw a die"]) if spec == "1d6" else rng.choice([f"roll {spec}", f"roll a d{sides}"]) if count == 1
                else rng.choice([f"roll {spec}", f"roll {count} d{sides}"]))
        answer = f"You rolled {rolls[0]}." if count == 1 else f"You rolled {sum(rolls)} ({' + '.join(map(str, rolls))})."
    else:
        text = (rng.choice(["dă cu zarul", "aruncă un zar", "dă un zar"]) if spec == "1d6" else rng.choice([f"aruncă {spec}", f"aruncă un d{sides}"]) if count == 1
                else rng.choice([f"aruncă {spec}", f"aruncă {count} zaruri cu {sides} fețe"]))
        answer = f"Ai dat {rolls[0]}." if count == 1 else f"Ai dat {sum(rolls)} ({' + '.join(map(str, rolls))})."
    return [user(text), tool_call(f"dice: {spec}"), context(result), assistant(answer)]


def coin_conversation(rng, language):
    side = rng.choice(["heads", "tails"])
    if language == "en":
        return [user(rng.choice(["flip a coin", "heads or tails?", "toss a coin"])), tool_call("coin:"), context(side), assistant(side.capitalize() + "!")]
    return [user(rng.choice(["dă cu banul", "aruncă o monedă", "cap sau pajură?"])), tool_call("coin:"), context(side),
            assistant("Cap!" if side == "heads" else "Pajură!")]


OPTIONS = {"en": [["pizza", "pasta", "sushi"], ["tea", "coffee"], ["cinema", "a walk", "gaming"], ["red", "blue", "green", "yellow"],
                  ["Python", "C", "Rust"], ["Monday", "Friday"], ["pancakes", "omelette", "cereal"], ["the mountains", "the sea", "the city"]],
           "ro": [["pizza", "paste", "sushi"], ["ceai", "cafea"], ["cinema", "o plimbare", "jocuri"], ["roșu", "albastru", "verde", "galben"],
                  ["Python", "C", "Rust"], ["luni", "vineri"], ["clătite", "omletă", "cereale"], ["munte", "mare", "oraș"]]}


def pick_conversation(rng, language):
    options = rng.choice(OPTIONS[language])
    choice = rng.choice(options)
    joined = ", ".join(options[:-1]) + (" or " if language == "en" else " sau ") + options[-1] if len(options) > 2 else (" or " if language == "en" else " sau ").join(options)
    if language == "en":
        text = rng.choice([f"pick one: {joined}", f"I can't decide: {joined}. You choose!", f"choose for me: {joined}"])
        answer = rng.choice([f"Go with {choice}!", f"I pick {choice}.", f"{choice[0].upper() + choice[1:]}!"])
    else:
        text = rng.choice([f"alege unul: {joined}", f"nu mă pot decide: {joined}. Alege tu!", f"alege pentru mine: {joined}"])
        answer = rng.choice([f"Alege {choice}!", f"Eu aleg {choice}.", f"{choice[0].upper() + choice[1:]}!"])
    return [user(text), tool_call("pick: " + " | ".join(options)), context(choice), assistant(answer)]


# ====================================================================== memory
NAMES = ["Alex", "Maria", "Andrei", "Ioana", "Mihai", "Elena", "David", "Sofia", "Radu", "Ana", "Daniel", "Laura", "Tudor", "Bianca",
         "Stefan", "Carmen", "Victor", "Irina", "Paul", "Diana", "Chris", "Sam", "Jordan", "Nora", "Luca", "Teo", "Mara", "Vlad", "Ruxandra", "Ben"]
MEMORY_CITIES = ["Cluj-Napoca", "București", "Iași", "Timișoara", "Brașov", "Sibiu", "Oradea", "London", "Berlin", "Paris", "Dublin", "Madrid"]
HOBBIES = {"en": ["chess", "football", "reading", "hiking", "drawing", "gaming", "cooking", "cycling", "music", "photography"],
           "ro": ["șah", "fotbal", "citit", "drumeții", "desen", "jocuri video", "gătit", "ciclism", "muzică", "fotografie"]}


def fact_sentence(language, key, value):
    """How the user tells a fact, and how the model repeats it back."""
    if language == "en":
        return {"name": f"my name is {value}", "age": f"I'm {value} years old", "city": f"I live in {value}", "hobby": f"my hobby is {value}"}[key]
    return {"name": f"mă cheamă {value}", "age": f"am {value} de ani", "city": f"locuiesc în {value}", "hobby": f"hobby-ul meu este {value}"}[key]


def memory_save_conversation(rng, language):
    key = rng.choice(["name", "name", "age", "city", "hobby"])
    value = {"name": lambda: rng.choice(NAMES), "age": lambda: str(rng.randint(18, 60)), "city": lambda: rng.choice(MEMORY_CITIES),
             "hobby": lambda: rng.choice(HOBBIES[language])}[key]()
    if key == "name" and rng.random() < 0.5:
        text = rng.choice(["call me {v}", "I'm {v}", "hi, I'm {v}"] if language == "en" else ["spune-mi {v}", "sunt {v}", "salut, sunt {v}"]).format(v=value)
    else:
        text = fact_sentence(language, key, value)
        text = text[0].upper() + text[1:] if rng.random() < 0.5 else text
    if language == "en":
        answers = {"name": f"Nice to meet you, {value}! I'll remember that.", "age": f"Got it, {value}. I'll remember that.",
                   "city": f"{value}, nice! I'll remember that.", "hobby": f"Cool, {value}! I'll remember that."}
    else:
        answers = {"name": f"Mă bucur să te cunosc, {value}! Țin minte.", "age": f"Am înțeles, {value} de ani. Țin minte.",
                   "city": f"{value}, frumos! Țin minte.", "hobby": f"Misto, {value}! Țin minte."}
    conversation = [user(text), tool_call(f"memory_save: {key}={value}"), context(f"saved: {key}={value}"), assistant(answers[key])]
    if key == "name" and rng.random() < 0.3:   # asked again in the same chat: the answer is already in the conversation
        conversation += [user("what's my name?" if language == "en" else "cum mă cheamă?"),
                         assistant(f"Your name is {value}." if language == "en" else f"Te cheamă {value}.")]
    return maybe_small_talk(rng, language, conversation, chance=0.1)


def memory_answer_conversation(rng, language):
    """The app puts the saved facts in the system prompt ("Memory: name=Alex, age=20"): the model just answers from it."""
    facts = {"name": rng.choice(NAMES)}
    if rng.random() < 0.6:
        facts["age"] = str(rng.randint(18, 60))
    if rng.random() < 0.5:
        facts["city"] = rng.choice(MEMORY_CITIES)
    if rng.random() < 0.3:
        facts["hobby"] = rng.choice(HOBBIES[language])
    memory_line = "Memory: " + ", ".join(f"{key}={value}" for key, value in facts.items())
    asked = rng.choice(list(facts) + ["all"])
    if language == "en":
        parts = {"name": f"your name is {facts['name']}", "age": f"you are {facts.get('age')} years old", "city": f"you live in {facts.get('city')}",
                 "hobby": f"your hobby is {facts.get('hobby')}"}
        questions = {"name": ["what's my name?", "do you remember my name?"], "age": ["how old am I?", "do you remember my age?"],
                     "city": ["where do I live?", "which city do I live in?"], "hobby": ["what's my hobby?", "what do I like to do?"],
                     "all": ["what do you know about me?", "what do you remember about me?"]}
        if asked == "all":
            answer = "I remember that " + ", ".join(parts[key] for key in facts) + "."
        else:
            answer = parts[asked][0].upper() + parts[asked][1:] + "."
    else:
        parts = {"name": f"te cheamă {facts['name']}", "age": f"ai {facts.get('age')} de ani", "city": f"locuiești în {facts.get('city')}",
                 "hobby": f"hobby-ul tău este {facts.get('hobby')}"}
        questions = {"name": ["cum mă cheamă?", "îți amintești cum mă cheamă?"], "age": ["câți ani am?", "știi câți ani am?"],
                     "city": ["unde locuiesc?", "în ce oraș locuiesc?"], "hobby": ["ce hobby am?", "ce îmi place să fac?"],
                     "all": ["ce știi despre mine?", "ce îți amintești despre mine?"]}
        if asked == "all":
            answer = "Știu că " + ", ".join(parts[key] for key in facts) + "."
        else:
            answer = parts[asked][0].upper() + parts[asked][1:] + "."
    return [system(system_prompt("Singularity") + "\n" + memory_line), user(rng.choice(questions[asked])), assistant(answer)]


def memory_empty_conversation(rng, language):
    """No memory line in the system prompt: the model checks with memory_get, finds nothing and says so honestly."""
    key = rng.choice(["name", "age", "city"])
    if language == "en":
        text = {"name": "what's my name?", "age": "how old am I?", "city": "where do I live?"}[key]
        answer = {"name": "I don't know your name yet. What should I call you?", "age": "I don't know your age yet, you haven't told me.",
                  "city": "I don't know where you live. Tell me and I'll remember."}[key]
    else:
        text = {"name": "cum mă cheamă?", "age": "câți ani am?", "city": "unde locuiesc?"}[key]
        answer = {"name": "Încă nu știu cum te cheamă. Cum să-ți spun?", "age": "Încă nu știu câți ani ai, nu mi-ai spus.",
                  "city": "Nu știu unde locuiești. Spune-mi și țin minte."}[key]
    return [user(text), tool_call(f"memory_get: {key}"), context(f"error: nothing remembered about '{key}'"), assistant(answer)]


# ====================================================================== main
KINDS = [("calc", calc_conversation, 2600), ("convert", convert_conversation, 600), ("now", now_conversation, 500),
         ("weather", weather_conversation, 650), ("dice", dice_conversation, 200), ("coin", coin_conversation, 100),
         ("pick", pick_conversation, 150), ("memory_save", memory_save_conversation, 450),
         ("memory_answer", memory_answer_conversation, 250), ("memory_empty", memory_empty_conversation, 120)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--scale", type=float, default=1.0, help="2.0 = twice as many conversations")
    args = parser.parse_args()
    rng = random.Random(args.seed)
    conversations = []
    for name, function, count in KINDS:
        made = 0
        for _ in range(int(count * args.scale)):
            language = "ro" if rng.random() < 0.4 else "en"
            conversations.append(function(rng, language))
            made += 1
        print(f"  {name:14s} {made}")
    write_jsonl("gen_tools", conversations, rng)


if __name__ == "__main__":
    main()
