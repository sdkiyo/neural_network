from operator import mul
from math import tanh
from random import random, seed

BIAS = 1

class Neuron:
    def __init__(self, num_inputs):
        self.delta = 0.0    # Delta is how much this neuron affected output
        self.output = 0.0   # Saved during fire
        # We add a bias weight - This allows the graph of the activation to shift left or right.
        # See: https://stackoverflow.com/questions/2480650/role-of-bias-in-neural-networks
        self.weights = [random() for i in range(num_inputs + BIAS)]

    # Activation function for neuron - I'm using a hyperbolic tangent
    # but you can use a sigmoid and many others - you need to have
    # its matching derivative for the derivative() function
    def fire(self, inputs):
        # Translates to tanh(dot_product(inputs, weights)) excluding bias weight
        self.output = tanh(sum(map(mul, inputs, self.weights[:-1])))
        return self.output

    # The answer will be the slope of the tangent line to the curve at that point.
    # i.e: how much does this affect the rate of change
    # See: http://www.ugrad.math.ubc.ca/coursedoc/math100/notes/derivs/deriv6.html
    def derivative(self):
        return 1.0 - self.output ** 2

class Layer:
    def __init__(self, num_neurons, num_inputs):
        self.neurons = [Neuron(num_inputs) for i in range(0, num_neurons)]

class ANN:
    # Takes in a list of the shape - for example [2, 2, 1] means:
    #   - 2 inputs
    #   - A hidden layer of 2 neurons
    #   - 1 output neuron (A neuron for however many outputs you need)
    #
    # This can use as many layers of arbitrary size as necessary
    def __init__(self, shape):
        self.layers = list()
        for i in range(1, len(shape)):
            self.layers.append(Layer(shape[i], shape[i - 1]))

    # Feeds inputs through first layer - then uses outputs from layer
    # to feed through next layer
    def feed_forward(self, inputs):
        x = inputs
        for layer in self.layers:
            new_inputs = []
            for neuron in layer.neurons:
                new_inputs.append(neuron.fire(x))
            x = new_inputs
        return x

    # Calculate errors and set neuron deltas by feeding backwards
    def back_propagate(self, expected):
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            errors = list()
            if i != len(self.layers) - 1:
                for j in range(len(layer.neurons)):
                    error = 0.0
                    for neuron in self.layers[i + 1].neurons:
                        error += (neuron.weights[j] * neuron.delta)
                    errors.append(error)
            else:
                for j in range(len(layer.neurons)):
                    neuron = layer.neurons[j]
                    errors.append(expected[j] - neuron.output)

            for j in range(len(layer.neurons)):
                neuron = layer.neurons[j]
                neuron.delta = errors[j] * neuron.derivative()

    # Update weights based on deltas
    def update_weights(self, inputs, learning_rate):
        for i in range(len(self.layers)):
            if i != 0: # If not first hidden layer
                inputs = [neuron.output for neuron in self.layers[i - 1].neurons]
            for neuron in self.layers[i].neurons:
                for j in range(len(inputs)):
                    neuron.weights[j] += learning_rate * neuron.delta * inputs[j]
                # Update bias weight
                neuron.weights[-1] += learning_rate * neuron.delta

    def train(self, dataset, learning_rate, epochs):
        for epoch in range(epochs):
            sum_error = 0.0
            for row in dataset:
                inputs = row[0]
                expected = row[1]
                outputs = self.feed_forward(inputs)
                sum_error += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])
                self.back_propagate(expected)
                self.update_weights(inputs, learning_rate)
            print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, learning_rate, sum_error))

    def predict(self, inputs):
        return self.feed_forward(inputs)

if __name__ == "__main__":
    seed(1)

    # Left are inputs, right are expected
    # --------------------------------------------------
    # Inputs/Expected are expected to be wrapped in list
    # regardless if only one input or one output
    multi_output_dataset = [
        [[0, 0], [0, 0]],
        [[1, 0], [0, 1]],
        [[1, 1], [1, 1]],
        [[0, 1], [1, 0]]
    ]

    mo_nn = ANN([2, 2, 2])
    mo_nn.train(multi_output_dataset, 0.2, 1000)

    xor_dataset = [
        [[0, 0], [0]],
        [[0, 1], [1]],
        [[1, 0], [1]],
        [[1, 1], [0]]
    ]

    xor_nn = ANN([2, 2, 1])
    xor_nn.train(xor_dataset, 0.2, 1000)

    print("Multi output predictions:")
    for row in multi_output_dataset:
        print("%s -> %s" % (row[0], mo_nn.predict(row[0])))

    print("XOR predictions")
    for row in xor_dataset:
        print("%s -> %s" % (row[0], xor_nn.predict(row[0])))
