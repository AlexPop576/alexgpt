class CharTokenizer:

    def __init__(self, text):

        self.chars = sorted(set(text))
        self.vocab_size = len(self.chars)

        self.stoi = {}
        for i, ch in enumerate(self.chars):
            self.stoi[ch] = i

        self.itos = {}
        for i, ch in enumerate(self.chars):
            self.itos[i] = ch

    def encode(self, s):
        result = []
        for ch in s:
            result.append(self.stoi[ch])
        return result

    def decode(self, ids):
        letters = []
        for i in ids:
            letters.append(self.itos[i])
        return "".join(letters)