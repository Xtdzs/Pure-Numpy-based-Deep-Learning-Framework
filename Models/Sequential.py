import numpy as np
import Preprocessing
from Models.Model import Model


class Sequential(Model):
    def __init__(self, *layers, decay_rate=0, **kwargs):
        super().__init__(**kwargs)
        self.name = 'Sequential'
        self.layers = list(layers)
        self.decay_rate = decay_rate
        self.input = None
        self.output = None
        self.total_output = None

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, input):
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

    # def gradient_check(self, X, Y, epsilon=1e-7):
    #     print('Gradient Checking...')
    #     for layer in self.layers:
    #         if layer.name == 'Dropout':
    #             continue
    #         print('Checking', layer.name)
    #         layer.forward(X)
    #         layer.backward(Y - layer.output)
    #         for i, param in enumerate(layer.params):
    #             param += epsilon
    #             layer.forward(X)
    #             loss1 = np.mean(np.square(Y - layer.output))
    #             param -= 2 * epsilon
    #             layer.forward(X)
    #             loss2 = np.mean(np.square(Y - layer.output))
    #             param += epsilon
    #             numerical_gradient = (loss1 - loss2) / (2 * epsilon)
    #             print('Analytical Gradient:', layer.grads[i], 'Numerical Gradient:', numerical_gradient)

    def accuracy(self, Y_hat, Y):
        return np.sum(Y_hat == Y) / len(Y)

    def predict(self, X):
        Y = self.forward(X).T
        if self.layers[-1].activation == 'softmax':
            Y = np.argmax(Y, axis=1)
            return Y
        return Y

    def evaluate(self, Y_hat, Y):
        return np.mean(np.square(Y_hat - Y))

    def save(self, filename):
        np.savez(filename, layers=self.layers)

    def load(self, filename):
        data = np.load(filename, allow_pickle=True)
        self.layers = data['layers']
        self.input = None
        self.output = None

    def __str__(self):
        return 'Sequential Model'

    def __repr__(self):
        return 'Sequential Model'

    def __len__(self):
        return len(self.layers)

    def __getitem__(self, index):
        return self.layers[index]
