import numpy as np


class Flatten:
    def __init__(self):
        self.name = 'Flatten'
        self.input = None
        self.output = None
        self.d_input = None
        self.d_output = None

    def forward(self, input):
        self.input = input
        self.output = input.reshape(input.shape[0], -1).T
        return self.output

    def backward(self, d_input):
        self.d_input = d_input.T.reshape(self.input.shape)
        return self.d_input

    def update(self):
        pass