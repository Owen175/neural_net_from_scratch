# Neural net from scratch - inspired by Sebastian Lague but coded from scratch.
import math
from random import random

class InputData:
    def __init__(self, data):
        self.data = data
        self.output = None
    def set_expected_output(self, output):
        self.output = output

class Layer:
    def __init__(self, incoming, outgoing):
        self.incoming = incoming
        self.outgoing = outgoing

        range_val = 1 / math.sqrt(self.incoming)
        self.weights = [[2 * (random() - 0.5) * range_val for _ in range(self.outgoing)] for _ in range(self.incoming)]
        # self.weights[0][1] is the weight from the first node to the second
        # Good practice is to start your weights in the range of [-y, y] where y=1/sqrt (n)
        # (n is the number of inputs to a given neuron).
        # https://medium.com/swlh/weight-initialization-technique-in-neural-networks-fc3cbcd03046
        self.biases = [0] * self.outgoing

        self.grad_weights = [[0] * self.outgoing] * self.incoming
        self.grad_biases = [0] * self.outgoing

    def process(self, ipt):
        output = self.biases[:]
        for i in range(len(self.weights)):
            for j in range(len(self.weights[0])):
                output[j] += ipt[i] * self.weights[i][j]
class NeuralNet:
    def __init__(self, structure):
        self.layers = []
        inc = structure[0]
        for struct in structure[1:]:
            self.layers.append(Layer(inc, struct))
            inc = struct

    def compute(self, inp):
        data = inp.data
        for layer in self.layers:
            data = layer.process(data)
        return data
