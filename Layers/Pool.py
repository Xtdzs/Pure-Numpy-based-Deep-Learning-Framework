import numpy as np
from Layers.Layer import Layer


class Pool(Layer):
    interation = 0

    def __init__(self, pool_size=(2, 2), strides=None, padding='valid', data_format=None, method='max',
                 input_shape=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        Pool.interation += 1
        self.iteration = Pool.interation
        self.name = 'Pool'
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
        self.input = None
        self.output = None
        self.d_input = None
        self.d_output = None
        self.mask = None
        self.method = method
        self.input_shape = input_shape

    def forward(self, input):
        self.input = input
        self.output = np.zeros(
            (input.shape[0], input.shape[1] // self.pool_size[0], input.shape[2] // self.pool_size[1], input.shape[3]))
        self.mask = np.zeros((input.shape[0], input.shape[1], input.shape[2], input.shape[3]))

        if self.method == 'max':
            for i in range(self.output.shape[1]):
                for j in range(self.output.shape[2]):
                    self.output[:, i, j, :] = np.max(input[:, i * self.pool_size[0]:(i + 1) * self.pool_size[0],
                                                     j * self.pool_size[1]:(j + 1) * self.pool_size[1], :], axis=(1, 2))
                    self.mask[:, i * self.pool_size[0]:(i + 1) * self.pool_size[0],
                    j * self.pool_size[1]:(j + 1) * self.pool_size[1], :] \
                        = (input[:, i * self.pool_size[0]:(i + 1) * self.pool_size[0],
                           j * self.pool_size[1]:(j + 1) * self.pool_size[1], :] == self.output[:, i, j, :][:, None,
                                                                                    None, :])
                    self.mask[:, i * self.pool_size[0]:(i + 1) * self.pool_size[0],
                    j * self.pool_size[1]:(j + 1) * self.pool_size[1], :] \
                        = self.mask[:, i * self.pool_size[0]:(i + 1) * self.pool_size[0],
                          j * self.pool_size[1]:(j + 1) * self.pool_size[1], :] / \
                          np.sum(self.mask[:, i * self.pool_size[0]:(i + 1) * self.pool_size[0],
                                 j * self.pool_size[1]:(j + 1) * self.pool_size[1], :], axis=(1, 2))[:, None, None, :]

        elif self.method == 'average':
            for i in range(self.output.shape[1]):
                for j in range(self.output.shape[2]):
                    self.output[:, i, j, :] = np.mean(input[:, i * self.pool_size[0]:(i + 1) * self.pool_size[0],
                                                      j * self.pool_size[1]:(j + 1) * self.pool_size[1], :],
                                                      axis=(1, 2))
        else:
            raise ValueError('Invalid pooling method.')

        return self.output

    def backward(self, d_output):
        self.d_input = np.zeros_like(self.input)

        if self.method == 'max':
            for i in range(self.output.shape[1]):
                for j in range(self.output.shape[2]):
                    self.d_input[:, i * self.pool_size[0]:(i + 1) * self.pool_size[0],
                    j * self.pool_size[1]:(j + 1) * self.pool_size[1], :] \
                        += d_output[:, i, j, :][:, None, None, :] \
                           * self.mask[:, i * self.pool_size[0]:(i + 1) * self.pool_size[0],
                             j * self.pool_size[1]:(j + 1) * self.pool_size[1], :]

        elif self.method == 'average':
            for i in range(self.output.shape[1]):
                for j in range(self.output.shape[2]):
                    self.d_input[:, i * self.pool_size[0]:(i + 1) * self.pool_size[0],
                    j * self.pool_size[1]:(j + 1) * self.pool_size[1], :] += d_output[:, i, j, :][:, None, None, :] / \
                                                                             self.pool_size[0] / self.pool_size[1]
        else:
            raise ValueError('Invalid pooling method.')

        return self.d_input

    def update(self):
        pass

    def get_params(self):
        return 0

    def output_shape(self):
        return f'({self.input_shape[0] // self.pool_size[0]}, {self.input_shape[1] // self.pool_size[1]}, {self.input_shape[2]})'

    def get_interation(self):
        return self.iteration
