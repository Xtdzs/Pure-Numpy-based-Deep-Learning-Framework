import numpy as np
from Layers.Flatten import Flatten
from Layers.Dense import Dense
from Layers.Conv2D import Conv2D
from Layers.Pool import Pool
from Models.Model import Model


class LeNet5(Model):
    def __init__(self, input_shape, num_classes, decay_rate=0, **kwargs):
        super().__init__(**kwargs)
        self.name = 'LeNet5'
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.decay_rate = decay_rate
        self.input = None
        self.output = None
        self.total_output = None
        self.layers = [
            Conv2D(6, (5, 5), input_shape=input_shape, activation='relu', padding='same'),
            Pool((2, 2), stride=2, mode='max'),
            Conv2D(16, (5, 5), activation='relu', padding='valid'),
            Pool((2, 2), stride=2, mode='max'),
            Flatten(),
            Dense(400, 120, activation='relu'),
            Dense(120, 84, activation='relu'),
            Dense(84, num_classes, activation='softmax')
        ]

    def forward(self, input):
        if input.shape[1] != 32 or input.shape[2] != 32:
            raise ValueError('Input shape must be (-1, 32, 32, 1), but got {}'.format(input.shape))
        if self.layers[0].name == 'Dense':
            input = input.T
        else:
            pass
        i = 1
        for layer in self.layers:
            layer.interation = i
            input = layer.forward(input)
            i += 1
            if layer.name == 'Dropout' or layer.name == 'Pool' or layer.name == 'Conv2D' or layer.name == 'Flatten':
                i -= 1
        self.output = input
        return self.output

    def backward(self, d_input):
        for layer in reversed(self.layers):
            d_input = layer.backward(d_input)

    def update(self):
        for layer in self.layers:
            layer.update()

    def get_params(self):
        total_params = 0
        for layer in self.layers:
            total_params += layer.get_params()
        return total_params
