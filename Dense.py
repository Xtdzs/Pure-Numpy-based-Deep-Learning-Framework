import numpy as np


class Dense:
    def __init__(self, input_size, output_size, activation, learning_rate=0.01, lambd=0, momentum=False, beta_1=0,
                 RMSprop=False, beta_2=0, Adam=False, decay_rate=0, input_shape=None):
        self.name = 'Dense'
        self.last_is_flatten = False
        self.learning_rate = learning_rate
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.lambd = lambd
        self.momentum = momentum
        self.RMSprop = RMSprop
        self.Adam = Adam
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.decay_rate = decay_rate
        self.epsilon = 1e-8
        self.interation = 1
        self.params = []
        self.grads = []

        # He initialization
        if self.activation == 'relu':
            self.weights = np.random.randn(self.input_size, self.output_size) * np.sqrt(2 / self.input_size)
        elif self.activation == 'sigmoid':
            self.weights = np.random.randn(self.input_size, self.output_size) * np.sqrt(1 / self.input_size)
        elif self.activation == 'tanh':
            self.weights = np.random.randn(self.input_size, self.output_size) * np.sqrt(1 / self.input_size)
        elif self.activation == 'softmax':
            self.weights = np.random.randn(self.input_size, self.output_size) * np.sqrt(2 / self.input_size)

        self.biases = np.zeros((output_size, 1))
        self.input = None
        self.output = None
        self.Z = None
        self.d_input = None
        self.d_output = None
        self.d_Z = None
        self.d_weights = None
        self.d_biases = None
        self.v_d_weights = None
        self.v_d_biases = None
        self.s_d_weights = None
        self.s_d_biases = None
        self.v_d_weights_corrected = None
        self.v_d_biases_corrected = None
        self.s_d_weights_corrected = None
        self.s_d_biases_corrected = None

    def get_params(self):
        return self.weights.size + self.biases.size

    def output_shape(self):
        return f"(None, {self.output_size})"

    def act_func(self, x):
        if self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.activation == 'tanh':
            return np.tanh(x)
        elif self.activation == 'softmax':
            return np.exp(x) / np.sum(np.exp(x), axis=0)
        else:
            raise Exception('Invalid activation function')

    def act_func_deriv(self, x):
        if self.activation == 'relu':
            return np.where(x > 0, 1, 0)
        elif self.activation == 'sigmoid':
            return x * (1 - x)
        elif self.activation == 'tanh':
            return 1 - np.power(x, 2)
        elif self.activation == 'softmax':
            return 1
        else:
            raise Exception('Invalid activation function')

    def forward(self, input):
        self.input = input
        self.Z = np.dot(self.weights.T, self.input) + self.biases
        self.output = self.act_func(self.Z) + self.lambd * np.sum(np.square(self.weights)) / (2 * self.input.shape[1])
        return self.output

    def backward(self, d_input):
        self.d_input = d_input
        self.d_Z = self.act_func_deriv(self.output) * self.d_input
        self.d_weights = np.dot(self.d_Z, self.input.T).T / self.input.shape[1] + self.lambd * self.weights / self.input.shape[1]
        self.d_biases = np.sum(self.d_Z, axis=1, keepdims=True) / self.input.shape[1]
        self.d_output = np.dot(self.weights, self.d_Z)

        if self.momentum:
            self.momentum_do()
        elif self.RMSprop:
            self.RMSprop_do()
        elif self.Adam:
            self.Adam_do()

        # self.grads = []
        # self.grads.append(self.v_d_weights.reshape(-1, 1))
        # self.grads.append(self.v_d_biases.reshape(-1, 1))
        return self.d_output

    def momentum_do(self):
        self.v_d_weights = np.reshape(np.zeros((self.input_size, self.output_size)), (self.input_size, self.output_size))
        self.v_d_biases = np.reshape(np.zeros((self.output_size, 1)), (self.output_size, 1))

        self.v_d_weights = self.beta_1 * self.v_d_weights + (1 - self.beta_1) * self.d_weights
        self.v_d_biases = self.beta_1 * self.v_d_biases + (1 - self.beta_1) * self.d_biases
        self.d_weights = self.v_d_weights
        self.d_biases = self.v_d_biases

    def RMSprop_do(self):
        self.s_d_weights = np.reshape(np.zeros((self.input_size, self.output_size)), (self.input_size, self.output_size))
        self.s_d_biases = np.reshape(np.zeros((self.output_size, 1)), (self.output_size, 1))

        self.s_d_weights = self.beta_2 * self.s_d_weights + (1 - self.beta_2) * np.square(self.d_weights)
        self.s_d_biases = self.beta_2 * self.s_d_biases + (1 - self.beta_2) * np.square(self.d_biases)
        self.d_weights /= np.sqrt(self.s_d_weights + self.epsilon)
        self.d_biases /= np.sqrt(self.s_d_biases + self.epsilon)

    def Adam_do(self):
        self.v_d_weights = np.reshape(np.zeros((self.input_size, self.output_size)), (self.input_size, self.output_size))
        self.v_d_biases = np.reshape(np.zeros((self.output_size, 1)), (self.output_size, 1))
        self.s_d_weights = np.reshape(np.zeros((self.input_size, self.output_size)), (self.input_size, self.output_size))
        self.s_d_biases = np.reshape(np.zeros((self.output_size, 1)), (self.output_size, 1))
        self.v_d_weights_corrected = np.reshape(np.zeros((self.input_size, self.output_size)), (self.input_size, self.output_size))
        self.v_d_biases_corrected = np.reshape(np.zeros((self.output_size, 1)), (self.output_size, 1))
        self.s_d_weights_corrected = np.reshape(np.zeros((self.input_size, self.output_size)), (self.input_size, self.output_size))
        self.s_d_biases_corrected = np.reshape(np.zeros((self.output_size, 1)), (self.output_size, 1))

        self.v_d_weights = self.beta_1 * self.v_d_weights + (1 - self.beta_1) * self.d_weights
        self.v_d_biases = self.beta_1 * self.v_d_biases + (1 - self.beta_1) * self.d_biases
        self.v_d_weights_corrected = self.v_d_weights / (1 - np.power(self.beta_1, self.interation))
        self.s_d_weights = self.beta_2 * self.s_d_weights + (1 - self.beta_2) * np.square(self.d_weights)
        self.s_d_biases = self.beta_2 * self.s_d_biases + (1 - self.beta_2) * np.square(self.d_biases)
        self.s_d_weights_corrected = self.s_d_weights / (1 - np.power(self.beta_2, self.interation))
        self.d_weights = self.v_d_weights_corrected / (np.sqrt(self.s_d_weights_corrected) + self.epsilon)
        self.d_biases = self.v_d_biases_corrected / (np.sqrt(self.s_d_biases_corrected) + self.epsilon)

    def update(self):
        self.weights -= self.learning_rate * self.d_weights
        self.biases -= self.learning_rate * self.d_biases
        self.params = []
        self.params.append(self.weights.reshape(-1, 1))
        self.params.append(self.biases.reshape(-1, 1))
