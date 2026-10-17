import numpy as np


class Tensor:
    def __init__(self, data, a=None, b=None, op=""):
        self.data = data                    
        self.grad = np.zeros(data.shape)    
        self.a = a                          
        self.b = b                          
        self.op = op                        

    def backward_step(self):
        if self.op == "@":
            self.a.grad += self.grad @ self.b.data.T
            self.b.grad += self.a.data.T @ self.grad
        elif self.op == "+bias":
            self.a.grad += self.grad
            self.b.grad += self.grad.sum(axis=0)
        elif self.op == "tanh":
            self.a.grad += (1 - self.data ** 2) * self.grad
        elif self.op == "embed":
            batch_size = self.ids.shape[0]
            context_size = self.ids.shape[1]
            embed_dim = self.a.data.shape[1]
            grad_per_char = self.grad.reshape(batch_size, context_size, embed_dim)
            np.add.at(self.a.grad, self.ids, grad_per_char)
        elif self.op == "cross_entropy":
            batch_size = self.targets.shape[0]
            grad_logits = self.probs.copy()
            grad_logits[np.arange(batch_size), self.targets] -= 1
            self.a.grad += grad_logits / batch_size * self.grad
        elif self.op == "+":
            self.a.grad += self.grad
            self.b.grad += self.grad
        elif self.op == "T":
            self.a.grad += self.grad.T
        elif self.op == "softmax":
            T = self.data.shape[0]
            for t in range(T):
                w = self.data[t]                 
                g = self.grad[t]                 
                average = (w * g).sum()
                self.a.grad[t] += w * (g-average) * self.scale

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

        self.grad = np.ones(self.data.shape)
        for node in reversed(order):
            node.backward_step()

def transpose(x):
    return Tensor(x.data.T, x, None, "T")

def add(x, y):
    return Tensor(x.data + y.data, x, y, "+")

def matmul(x, y):
    return Tensor(x.data @ y.data, x, y, "@")


def add_bias(x, bias):
    return Tensor(x.data + bias.data, x, bias, "+bias")


def tanh(x):
    return Tensor(np.tanh(x.data), x, None, "tanh")

def embed(table, ids):
    batch_size = ids.shape[0]
    context_size = ids.shape[1]
    embed_dim = table.data.shape[1]
    vectors = table.data[ids]                        
    flat = vectors.reshape(batch_size, context_size * embed_dim)
    out = Tensor(flat, table, None, "embed")
    out.ids = ids                                   
    return out


def cross_entropy(logits, targets):
    shifted = logits.data - logits.data.max(axis=1, keepdims=True)   
    exps = np.exp(shifted)
    probs = exps / exps.sum(axis=1, keepdims=True)                   
    batch_size = targets.shape[0]
    correct_probs = probs[np.arange(batch_size), targets]
    loss = -np.log(correct_probs).mean()
    out = Tensor(np.array(loss), logits, None, "cross_entropy")
    out.probs = probs                                
    out.targets = targets
    return out

def causal_softmax(x, scale):
    T = x.data.shape[0]
    weights = np.zeros(x.data.shape)
    for t in range(T):
        row = x.data[t] * scale              # a NEW row (multiplying makes a copy), so x.data stays untouched
        # 1. for every s > t:  row[s] = -np.inf
        for s in range(x.data.shape[1]):
            if s > t:
                row[s] = -np.inf
        # 2. exps = np.exp(row - row.max())
        exps = np.exp(row - row.max())
        # 3. weights[t] = exps / exps.sum()
        weights[t] = exps / exps.sum()
    out = Tensor(weights, x, None, "softmax")
    out.scale = scale                        # remember it for backward
    return out