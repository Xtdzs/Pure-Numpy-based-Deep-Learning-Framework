import numpy as np


class Layer:
    def __init__(self, *args, **kwargs):
        self.name = 'Layer'
        self.input = None
        self.output = None
        self.d_input = None
        self.d_output = None
        self.weights = None
        self.biases = None
        self.d_weights = None
        self.d_biases = None

    def forward(self, input):
        pass

    def backward(self, d_output):
        pass

    def update(self):
        pass

    def get_params(self):
        return self.weights.size + self.biases.size

    def get_output_shape(self):
        pass
