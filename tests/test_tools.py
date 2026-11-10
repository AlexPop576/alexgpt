"""Checks alexgpt/tools.py:  python tests/test_tools.py   (add --online for the tools that need the internet)"""
import os
import sys
import tempfile

sys.path.insert(0, ".")
from alexgpt import tools  # noqa: E402

ONLINE = "--online" in sys.argv
results = {"pass": 0, "fail": 0, "todo": 0}


def check(name, function, expected=None, contains=None, starts_with=None, test=None):
    try:
        got = function()
    except NotImplementedError:
        results["todo"] += 1
        print(f"  TODO  {name}")
        return
    except Exception as error:
        results["fail"] += 1
        print(f"  FAIL  {name}: crashed with {type(error).__name__}: {error}   (a tool must never crash)")
        return
    ok = True
    if expected is not None:
        ok = got == expected
    if contains is not None:
        ok = ok and contains in got
    if starts_with is not None:
        ok = ok and got.startswith(starts_with)
    if test is not None:
        ok = ok and test(got)
    ok = ok and len(got) <= tools.MAX_RESULT_CHARS
    results["pass" if ok else "fail"] += 1
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}" + ("" if ok else f"   got {got!r}" + (f", wanted {expected!r}" if expected is not None else "")))


print("calc")
check("245 * 18", lambda: tools.calc("245 * 18"), "4410")
check("precedence", lambda: tools.calc("2 + 3 * 4"), "14")
check("parentheses + power", lambda: tools.calc("(3 + 4) ** 2"), "49")
check("division", lambda: tools.calc("10 / 4"), "2.5")
check("sqrt", lambda: tools.calc("sqrt(144)"), "12")
check("percent", lambda: tools.calc("15% of 80"), "12")
check("decimals", lambda: tools.calc("1 / 3"), "0.3333")
check("division by zero", lambda: tools.calc("5 / 0"), starts_with="error:")
check("NOT code: __import__", lambda: tools.calc("__import__('os').system('echo hi')"), starts_with="error:")
check("NOT code: names", lambda: tools.calc("abs.__class__"), starts_with="error:")

print("convert")
check("km to miles", lambda: tools.convert(5, "km", "miles"), "5 km = 3.107 miles")
check("celsius to fahrenheit", lambda: tools.convert(100, "celsius", "fahrenheit"), "100 celsius = 212 fahrenheit")
check("unknown unit", lambda: tools.convert(1, "km", "banana"), starts_with="error:")
check("mixed kinds", lambda: tools.convert(1, "km", "kg"), starts_with="error:")

print("now / dice / coin / pick")
check("now has the zone", lambda: tools.now("Asia/Tokyo"), contains="(Asia/Tokyo)")
check("now bad zone", lambda: tools.now("Mars/Base"), starts_with="error:")
check("dice 2d6", lambda: tools.dice("2d6"), test=lambda text: text.startswith("2d6: ") and " = " in text)
check("dice bad", lambda: tools.dice("banana"), starts_with="error:")
check("coin", lambda: tools.coin(), test=lambda text: text in ("heads", "tails"))
check("pick", lambda: tools.pick("pizza | pasta | sushi"), test=lambda text: text in ("pizza", "pasta", "sushi"))
check("pick one option", lambda: tools.pick("pizza"), starts_with="error:")

print("memory")
tools.MEMORY_FILE = os.path.join(tempfile.mkdtemp(), "memory.json")
check("save", lambda: tools.memory_save("name", "Alex"), "saved: name=Alex")
check("save 2", lambda: tools.memory_save("age", "20"), "saved: age=20")
check("get one", lambda: tools.memory_get("name"), "name=Alex")
check("get all", lambda: tools.memory_get(""), "name=Alex, age=20")
check("get missing", lambda: tools.memory_get("city"), starts_with="error:")

print("run_tool (the line the model writes)")
check("calc line", lambda: tools.run_tool("calc: 245 * 18"), "4410")
check("convert line", lambda: tools.run_tool("convert: 5 km to miles"), "5 km = 3.107 miles")
check("memory line", lambda: tools.run_tool("memory_save: city=Cluj"), "saved: city=Cluj")
check("unknown tool", lambda: tools.run_tool("hack: now"), starts_with="error: unknown tool")
check("no colon", lambda: tools.run_tool("hello"), starts_with="error:")

if ONLINE:
    print("online")
    check("wiki en", lambda: tools.wiki("Eiffel Tower", "en"), contains="Eiffel")
    check("wiki ro", lambda: tools.wiki("Turnul Eiffel", "ro"), contains="Eiffel")
    check("wiki nothing", lambda: tools.wiki("zzqxqxqxzz9999", "en"), starts_with="error:")
    check("weather", lambda: tools.weather("Cluj-Napoca"), contains="°C")
    check("weather bad city", lambda: tools.weather("Qzxqxqxzz"), starts_with="error:")
    check("today in history", lambda: tools.today_in_history(10, 24, "en"), contains="—")

print(f"\n{results['pass']} ok, {results['fail']} failed, {results['todo']} not written yet")
sys.exit(1 if results["fail"] else 0)
