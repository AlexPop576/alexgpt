import random
from alexgpt.engine import Value, add, mul, tanh

class Neuron:
    def __init__(self, num_inputs):
        self.weights = []
        for _ in range (num_inputs):
            self.weights.append(Value(random.uniform(-1, 1)))
        self.bias = Value(random.uniform(-1,1))

    def forward(self, inputs):
        weighted_sum = self.bias
        for i in range (len(self.weights)):
            weighted_sum = add(weighted_sum, mul(self.weights[i], inputs[i]))
        return tanh(weighted_sum)

    def parameters(self):
        params = []
        for weight in self.weights:
            params.append(weight)
        params.append(self.bias)
        return params

class Layer:
    def __init__(self, num_inputs, num_neurons):
        self.neurons = []
        for _ in range (num_neurons):
            self.neurons.append(Neuron(num_inputs))

    def forward(self, inputs):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.forward(inputs))
        return outputs

    def parameters(self):
        params = []
        for neuron in self.neurons:
            params.extend(neuron.parameters())
        return params


class MLP:
    def __init__(self, num_inputs, layer_sizes):
        self.layers = []
        previous_size = num_inputs
        for size in layer_sizes:
            self.layers.append(Layer(previous_size, size))
            previous_size = size

    def forward(self, inputs):
        values = inputs
        for layer in self.layers:
            values = layer.forward(values)
        return values

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params