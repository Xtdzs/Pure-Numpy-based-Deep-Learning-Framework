import numpy as np
from Layers.Layer import Layer


class Conv1D(Layer):
    interation = 0

    def __init__(self, filters=1, kernel_size=1, strides=1, padding='valid', data_format=None, dilation_rate=1,
                 activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',
                 kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
                 bias_constraint=None, input_shape=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        Conv1D.interation += 1
        self.iteration = Conv1D.interation
        self.name = 'Conv1D'
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
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint
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
        self.input_shape = input_shape
        self.learning_rate = None

        if kernel_initializer == 'glorot_uniform':
            self.weights = np.random.randn(kernel_size, input_shape[1], filters) * np.sqrt(
                2 / (input_shape[0] * input_shape[1]))
        elif kernel_initializer == 'zeros':
            self.weights = np.zeros((kernel_size, input_shape[1], filters))
        else:
            raise ValueError('Invalid kernel_initializer')

        if bias_initializer == 'zeros':
            self.biases = np.zeros((filters,))
        else:
            raise ValueError('Invalid bias_initializer')

        if kernel_regularizer is not None:
            raise ValueError('Invalid kernel_regularizer')

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
        if input.ndim != 3:
            input = np.expand_dims(input, axis=2)
        self.input = input
        if self.padding == 'valid':
            self.output = np.zeros((input.shape[0], (input.shape[1] - self.kernel_size) // self.strides + 1,
                                    self.filters))
            self.Z = np.zeros((input.shape[0], (input.shape[1] - self.kernel_size) // self.strides + 1,
                               self.filters))
        elif self.padding == 'same':
            self.output = np.zeros((input.shape[0], input.shape[1], self.filters))
            self.Z = np.zeros((input.shape[0], input.shape[1], self.filters))
            zeros = np.zeros((input.shape[0], input.shape[1]  + self.kernel_size * 2 - 2, self.filters))
            zeros[:, self.kernel_size - 1: 1 - self.kernel_size, :] = input
            self.input = zeros
        else:
            raise ValueError('Invalid padding')

        for f in range(self.filters):
            for i in range(self.output.shape[1]):
                self.Z[:, i, f] = np.sum(
                    np.multiply(self.input[:, i * self.strides:i * self.strides + self.kernel_size, :], self.weights[:, :, f])
                    , axis=(1, 2)) + self.biases[f]

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
                self.d_weights[:, :, f] += np.sum(
                    self.input[:, i * self.strides:i * self.strides + self.kernel_size, :] * self.d_output[:, i, f][:, None, None], axis=0) / self.input.shape[0]
                self.d_input[:, i * self.strides:i * self.strides + self.kernel_size] += self.weights[:, :, f] * self.d_Z[:, i, f][:, None, None]
                if self.use_bias:
                    self.d_biases[f] += np.sum(self.d_output[:, i, f], axis=0)

        if self.padding == 'same':
            self.d_input = self.d_input[:, self.kernel_size // 2:-self.kernel_size // 2, :]

        return self.d_input

    def update(self):
        self.weights -= self.learning_rate * self.d_weights
        self.biases -= self.learning_rate * self.d_biases

    def get_params(self):
        return self.weights.size + self.biases.size

    def output_shape(self):
        return f'({(self.input_shape[0] - self.kernel_size) // self.strides + 1},{self.filters})'
