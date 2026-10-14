import random
from alexgpt.engine import Value, add, mul, tanh

class Neuron:
    def __init__(self, n_inputs):
        self.weights = []
        for i in range (n_inputs):
            self.weights.append(Value(random.uniform(-1, 1)))
        self.bias = Value(random.uniform(-1,1))

    def forward(self, inputs):
        weighted_sum = self.bias
        for i in range (len(self.weights)):
            weighted_sum = add(weighted_sum, mul(self.weights[i], inputs[i]))
        return tanh(weighted_sum)

    def parameters(self):
        param = []
        for weight in self.weights:
            param.append(weight)
        param.append(self.bias)
        return param
