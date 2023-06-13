import numpy as np


class Dropout:
    def __init__(self, dropout_ratio=0.5):
        self.name = 'Dropout'
        self.dropout_ratio = dropout_ratio
        self.input = None
        self.output = None
        self.d_input = None
        self.d_output = None
        self.mask = None

    def forward(self, input):
        self.mask = np.random.rand(*input.shape) >= self.dropout_ratio
        self.output = np.multiply(input, self.mask) / (1 - self.dropout_ratio)
        return self.output

    def backward(self, d_input):
        self.d_output = np.multiply(d_input, self.mask) / (1 - self.dropout_ratio)
        return self.d_output

    def update(self):
        pass
