import numpy as np
from Layers.Layer import Layer


class Conv2D(Layer):
    iteration = 0

    def __init__(self, filters, kernel_size, strides=(1, 1), input_shape=None, padding='valid', data_format=None,
                 dilation_rate=(1, 1), learning_rate=0.01,
                 activation='linear', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        Conv2D.iteration += 1
        self.iteration = Conv2D.iteration
        self.input_shape = input_shape
        self.name = 'Conv2D'
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
        self.dilation_rate = dilation_rate
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.learning_rate = learning_rate
        self.input = None
        self.output = None
        self.Z = None
        self.d_input = None
        self.d_output = None
        self.d_Z = None
        self.weights = None
        self.biases = None
        self.d_weights = None
        self.d_biases = None

        if kernel_initializer == 'glorot_uniform':
            self.weights = np.random.randn(kernel_size[0], kernel_size[1], input_shape[2], filters) * np.sqrt(
                2 / (input_shape[0] * input_shape[1] * input_shape[2]))
        elif kernel_initializer == 'glorot_normal':
            self.weights = np.random.randn(kernel_size[0], kernel_size[1], input_shape[2], filters) * np.sqrt(
                1 / (input_shape[0] * input_shape[1] * input_shape[2]))
        elif kernel_initializer == 'he_uniform':
            self.weights = np.random.randn(kernel_size[0], kernel_size[1], input_shape[2], filters) * np.sqrt(
                2 / input_shape[3])
        elif kernel_initializer == 'he_normal':
            self.weights = np.random.randn(kernel_size[0], kernel_size[1], input_shape[2], filters) * np.sqrt(
                1 / input_shape[3])
        elif kernel_initializer == 'zeros':
            self.weights = np.zeros((kernel_size[0], kernel_size[1], input_shape[2], filters))
        elif kernel_initializer == 'ones':
            self.weights = np.ones((kernel_size[0], kernel_size[1], input_shape[2], filters))
        elif kernel_initializer == 'random':
            self.weights = np.random.randn(kernel_size[0], kernel_size[1], input_shape[2], filters)
        else:
            raise ValueError(f'Kernel initializer {kernel_initializer} not supported')

        if bias_initializer == 'zeros':
            self.biases = np.zeros((1, 1, 1, filters))
        elif bias_initializer == 'ones':
            self.biases = np.ones((1, 1, 1, filters))
        elif bias_initializer == 'random':
            self.biases = np.random.randn(1, 1, 1, filters)
        else:
            raise ValueError(f'Bias initializer {bias_initializer} not supported')

    def get_params(self):
        return self.weights.size + self.biases.size

    def get_output_shape(self):
        return f'({self.input.shape[1] - self.kernel_size[0] + 1}, {self.input.shape[2] - self.kernel_size[1] + 1}, {self.filters})'

    def act_func(self, x):
        if self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.activation == 'tanh':
            return np.tanh(x)
        elif self.activation == 'softmax':
            return np.exp(x) / np.sum(np.exp(x), axis=0)
        elif self.activation == 'linear':
            return x
        else:
            raise ValueError(f'Activation function {self.activation} not supported')

    def act_func_deriv(self, x):
        if self.activation == 'relu':
            return np.where(x <= 0, 0, 1)
        elif self.activation == 'sigmoid':
            return x * (1 - x)
        elif self.activation == 'tanh':
            return 1 - x ** 2
        elif self.activation == 'softmax':
            return 1
        elif self.activation == 'linear':
            return 1
        else:
            raise ValueError(f'Activation function {self.activation} not supported')

    def forward(self, input):
        self.input = input
        if self.padding == 'valid':
            self.Z = np.zeros((self.input.shape[0], self.input.shape[1] - self.kernel_size[0] + 1,
                               self.input.shape[2] - self.kernel_size[1] + 1, self.filters))
            self.output = np.zeros(
                (self.input.shape[0], ((self.input.shape[1] - self.kernel_size[0]) // self.strides[0] + 1),
                 ((self.input.shape[2] - self.kernel_size[1]) // self.strides[1] + 1), self.filters))
        elif self.padding == 'same':
            self.Z = np.zeros((self.input.shape[0], self.input.shape[1], self.input.shape[2], self.filters))
            self.output = np.zeros((self.input.shape[0], self.input.shape[1], self.input.shape[2], self.filters))
            self.input = np.pad(self.input, ((0, 0), (self.kernel_size[0] // 2, self.kernel_size[0] // 2),
                                             (self.kernel_size[1] // 2, self.kernel_size[1] // 2), (0, 0)), 'constant')

        else:
            raise ValueError(f'Padding {self.padding} not supported')

        for f in range(self.filters):
            for i in range(self.output.shape[1]):
                for j in range(self.output.shape[2]):
                    self.Z[:, i, j, f] = np.sum(
                        np.multiply(self.input[:, i:i + self.kernel_size[0], j:j + self.kernel_size[1], :],
                                    self.weights[:, :, :, f]), axis=(1, 2, 3)) + self.biases[0, 0, 0, f]

        if self.activation is not None:
            self.output = self.act_func(self.Z)
        else:
            self.output = self.Z

        return self.output

    def backward(self, d_output):
        self.d_output = d_output
        if self.activation is not None:
            self.d_Z = self.act_func_deriv(self.output) * d_output
        else:
            self.d_Z = d_output

        self.d_weights = np.zeros_like(self.weights)
        self.d_biases = np.zeros_like(self.biases)
        self.d_input = np.zeros_like(self.input)

        for f in range(self.filters):
            for i in range(self.output.shape[1]):
                for j in range(self.output.shape[2]):
                    self.d_weights[:, :, :, f] += np.sum(
                        self.input[:, i:i + self.kernel_size[0], j:j + self.kernel_size[1]] * self.d_Z[:, i, j, f][:,
                                                                                              None, None, None],
                        axis=0) / self.input.shape[0]
                    self.d_input[:, i:i + self.kernel_size[0], j:j + self.kernel_size[1]] += self.weights[:, :, :,
                                                                                             f] * self.d_Z[:, i, j, f][
                                                                                                  :, None, None, None]
                    self.d_biases[0, 0, 0, f] += np.sum(self.d_Z[:, i, j, f]) / self.input.shape[0]

        if self.padding == 'same':
            self.d_input = self.d_input[:, self.kernel_size[0] // 2:-self.kernel_size[0] // 2,
                           self.kernel_size[1] // 2:-self.kernel_size[1] // 2]

        return self.d_input

    def update(self):
        self.weights -= self.learning_rate * self.d_weights
        self.biases -= self.learning_rate * self.d_biases

    def output_shape(self):
        return f'({self.input_shape[0] - self.kernel_size[0] + 1}, {self.input_shape[1] - self.kernel_size[1] + 1}, {self.filters})'

    def get_interation(self):
        return self.iteration
