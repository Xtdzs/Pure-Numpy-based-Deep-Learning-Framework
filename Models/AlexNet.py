import numpy as np
from Layers.Dense import Dense
from Layers.Flatten import Flatten
from Layers.Conv2D import Conv2D
from Layers.Pool import Pool
from Models.Model import Model


class AlexNet(Model):
    def __init__(self, input_shape, num_classes, version=0, decay_rate=0, **kwargs):
        super().__init__(**kwargs)
        self.name = 'AlexNet'
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.decay_rate = decay_rate
        self.input = None
        self.output = None
        self.total_output = None
        if input_shape[0] == 227 & input_shape[1] == 227 & input_shape[2] == 3:
            self.version = 0
            self.layers = [
                Conv2D(96, (11, 11), input_shape=input_shape, activation='relu', padding='same', stride=4),
                Pool((3, 3), stride=2, mode='max'),
                Conv2D(256, (5, 5), activation='relu', padding='same'),
                Pool((3, 3), stride=2, mode='max'),
                Conv2D(384, (3, 3), activation='relu', padding='same'),
                Conv2D(384, (3, 3), activation='relu', padding='same'),
                Conv2D(256, (3, 3), activation='relu', padding='same'),
                Pool((3, 3), stride=2, mode='max'),
                Flatten(),
                Dense(9216, 4096, activation='relu'),
                Dense(4096, 4096, activation='relu'),
                Dense(4096, num_classes, activation='softmax')
            ]
        elif input_shape[0] == 224 & input_shape[1] == 224 & input_shape[2] == 3:
            self.version = 1
            self.layers = [
                Conv2D(48, (11, 11), input_shape=input_shape, activation='relu', padding='same', stride=4),
                Pool((3, 3), stride=2, mode='max'),
                Conv2D(128, (5, 5), activation='relu', padding='same'),
                Pool((3, 3), stride=2, mode='max'),
                Conv2D(192, (3, 3), activation='relu', padding='same'),
                Conv2D(192, (3, 3), activation='relu', padding='same'),
                Conv2D(128, (3, 3), activation='relu', padding='same'),
                Pool((3, 3), stride=2, mode='max'),
                Flatten(),
                Dense(6272, 2048, activation='relu'),
                Dense(2048, 2048, activation='relu'),
                Dense(2048, num_classes, activation='softmax')
            ]
        else:
            raise ValueError('Input shape must be (-1, 227, 227, 3) or (-1, 224, 224, 3), but got {}'.format(input_shape))

    def forward(self, input):
        if input.shape != self.input_shape:
            raise ValueError('Input shape must be (-1, 227, 227, 3) or (-1, 224, 224, 3), but got {}'.format(input.shape))
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