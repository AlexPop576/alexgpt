class Value:
    def __init__(self, data, a=None, b=None, op=""):
        self.data = data
        self.grad = 0.0
        self.a = a
        self.b = b
        self.op = op

    def backward_step(self):
        if self.op == "+":
            self.a.grad += self.grad
            self.b.grad += self.grad
        elif self.op == "*":
            self.a.grad += self.grad * self.b.data
            self.b.grad += self.grad * self.a.data

def add(x, y):
    return Value(x.data + y.data, x, y, "+")

def mul(x, y):
    return Value(x.data * y.data, x, y, "*")

