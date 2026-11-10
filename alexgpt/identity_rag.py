"""Identity self-RAG (round 1 in CLAUDE.md, docs/fable_plan.md section 2).
Questions about the bot itself ("who made you?", "ești ChatGPT?", "what's my sister's name?") are answered from
sft_data/identity_qa.txt, the answers Alex wrote, instead of whatever the 126M model remembers.

How it works:
  1. read the groups:  ## group / Q: question / A: answer   (the same file build_sft_mix.py uses for SFT)
  2. turn every Q into a vector: TF-IDF over character trigrams + whole words (no sklearn needed)
  3. a new message -> the most similar Q (cosine).  Similar enough AND about "you/me" -> a random A of that group.

    match("who built you?")  ->  ("creator-en", 0.71, "Who made you?")
    reply("who built you?", "Singularity")  ->  "Alex built me from scratch..."
"""
import math
import os
import random
import re
import unicodedata
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QA_FILE = os.path.join(ROOT, "sft_data", "identity_qa.txt")
SIZE = {"en": "about 125 million", "ro": "aproximativ 125 de milioane"}

THRESHOLD = 0.35            # below this the message is not an identity question (tuned with tune_identity_rag.py:
                            # 59/229 leave-one-out right, 0 of 32 non-identity questions accepted)
# the message must be about "you" or "me", except for these groups ("Who is Alex?", "thanks!", "bye")
NO_PERSON_NEEDED = ("hello", "thanks", "bye", "compliment", "rude")
# these groups need their own word instead: "who is Alesso?" looks like "Who is Alex?" but is not about Alex
GROUP_WORDS = {"alex": re.compile(r"\balex\b"), "family": re.compile(r"\bcosmos|\byou\b|\btu\b|\bta\b|\btale\b|\bfrati")}
PERSON_WORDS = re.compile(r"\b(you|your|yours|yourself|u|ur|r u|tu|te|ti|tau|ta|tale|tai|esti|dumneavoastra|voi|"
                          r"poti|stii|ai|faci|vrei|sunteti|"
                          r"my|me|i|im|mine|eu|meu|mea|mele|mei|mi|imi|ma|am)\b")


def fold(text):
    """'Ești ChatGPT?' -> 'esti chatgpt?'"""
    return "".join(c for c in unicodedata.normalize("NFKD", text.lower()) if not unicodedata.combining(c))


def read_groups(path=QA_FILE):
    """-> {group: {"questions": [...], "answers": [...]}}   (the same rules as build_sft_mix.read_identity_groups)"""
    groups, current, last = {}, None, None
    with open(path, encoding="utf-8") as file:
        for line in file:
            line = line.rstrip("\n")
            if line.startswith("## "):
                current = line[3:].strip()
                groups[current] = {"questions": [], "answers": []}
            elif current and line.startswith("Q:"):
                groups[current]["questions"].append(line[2:].strip())
                last = None
            elif current and line.startswith("A:"):
                groups[current]["answers"].append(line[2:].strip())
                last = groups[current]["answers"]
            elif current and last is not None and not line.startswith("#"):
                last[-1] += "\n" + line                    # an answer on several lines (a list)
    for group in groups.values():
        group["answers"] = [answer.strip() for answer in group["answers"]]
    return groups


def features(text):
    """'who r u' -> character trigrams of ' who r u ' + the words 'w:who', 'w:r', 'w:u'"""
    plain = " " + re.sub(r"[^\w\s]", " ", fold(text)) + " "
    plain = re.sub(r"\s+", " ", plain)
    grams = [plain[i:i + 3] for i in range(len(plain) - 2)]
    words = ["w:" + word for word in plain.split()]
    stems = ["s:" + word[:5] for word in plain.split() if len(word) > 3]     # "created" and "creator" -> "s:creat"
    return Counter(grams + words + stems)


class IdentityIndex:
    def __init__(self, path=QA_FILE, skip_questions=()):
        self.groups = read_groups(path)
        skip = {fold(q).strip(" ?!.") for q in skip_questions}
        self.rows = []                                   # (group, question, features)
        for name, group in self.groups.items():
            for question in group["questions"]:
                if fold(question).strip(" ?!.") not in skip:
                    self.rows.append((name, question, features(question)))
        document_count = Counter()
        for _, _, counts in self.rows:
            document_count.update(counts.keys())
        total = len(self.rows)
        self.idf = {feature: math.log((1 + total) / (1 + count)) + 1 for feature, count in document_count.items()}
        self.vectors = [self.vector(counts) for _, _, counts in self.rows]
        # the centre of each group: the average of its questions (catches a new wording better than one question)
        self.centres = {}
        for (group, _, _), vector in zip(self.rows, self.vectors):
            centre = self.centres.setdefault(group, Counter())
            for feature, weight in vector.items():
                centre[feature] += weight
        for group, centre in self.centres.items():
            length = math.sqrt(sum(w * w for w in centre.values())) or 1.0
            self.centres[group] = {f: w / length for f, w in centre.items()}

    def vector(self, counts):
        """TF-IDF, length 1 (so the dot product IS the cosine)"""
        weights = {f: (1 + math.log(c)) * self.idf.get(f, math.log(1 + len(self.rows)) + 1) for f, c in counts.items()}
        length = math.sqrt(sum(w * w for w in weights.values())) or 1.0
        return {f: w / length for f, w in weights.items()}

    def match(self, text):
        """-> (group, score 0..1, the most similar question).
        score of a group = 0.5 * its closest question + 0.5 * its centre"""
        query = self.vector(features(text))

        def cosine(vector):
            return sum(weight * vector.get(feature, 0.0) for feature, weight in query.items())
        closest = {}                                     # group -> (score, question)
        for (group, question, _), vector in zip(self.rows, self.vectors):
            score = cosine(vector)
            if score > closest.get(group, (-1.0, None))[0]:
                closest[group] = (score, question)
        best = (None, 0.0, None)
        for group, (score, question) in closest.items():
            combined = 0.5 * score + 0.5 * cosine(self.centres[group])
            if combined > best[1]:
                best = (group, combined, question)
        return best

    def accepts(self, text, group, score):
        """similar enough, and about you/me (unless the group does not need it)"""
        if group is None or score < THRESHOLD:
            return False
        kind = group.rsplit("-", 1)[0]
        if kind in NO_PERSON_NEEDED:
            return True
        if kind in GROUP_WORDS:
            return GROUP_WORDS[kind].search(fold(text)) is not None
        return PERSON_WORDS.search(fold(text)) is not None


_INDEX = None


def index():
    global _INDEX
    if _INDEX is None:
        _INDEX = IdentityIndex()
    return _INDEX


def reply(text, model_name, rng=random):
    """-> (answer, group) from identity_qa.txt, or (None, None) if the message is not an identity question"""
    if len(text.split()) > 15:                           # a long message is about something else
        return None, None
    found = index()
    group, score, _ = found.match(text)
    if not found.accepts(text, group, score):
        return None, None
    language = "ro" if group.endswith("-ro") else "en"
    answer = rng.choice(found.groups[group]["answers"])
    answer = answer.replace("{NAME}", model_name).replace("{SIZE_RO}", SIZE["ro"]).replace("{SIZE}", SIZE[language])
    return answer, group
