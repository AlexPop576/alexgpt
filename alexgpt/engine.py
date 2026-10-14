import math

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
        elif self.op == "tanh":
            self.a.grad += (1 - self.data ** 2) * self.grad

    def build_order(self, order, visited):
        if self in visited:
            return
        visited.add(self)
        if self.a is not None:
            self.a.build_order(order, visited)
        if self.b is not None:
            self.b.build_order(order, visited)
        order.append(self)

    def backward(self):
        order = []
        visited = set()
        self.build_order(order, visited)

        self.grad = 1.0
        for node in reversed(order):
            node.backward_step()

def add(x, y):
    return Value(x.data + y.data, x, y, "+")

def mul(x, y):
    return Value(x.data * y.data, x, y, "*")

def tanh(x):
    t = math.tanh(x.data)
    return Value(t, x, None, "tanh")
