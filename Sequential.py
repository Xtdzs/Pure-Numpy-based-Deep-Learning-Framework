import numpy as np
import Preprocessing


class Sequential:
    def __init__(self, *layers, decay_rate=0):
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

    def train(self, X, Y, epochs, batch_size=1, learning_rate=0.01, patience=None):
        global best_loss, epochs_without_improvement
        self.total_output = np.zeros((self.layers[-1].output_size, Y.shape[1]))

        if self.layers[-1].activation == 'softmax':
            Y = Preprocessing.to_one_hot(Y)

        if patience is not None:
            best_loss = float('inf')
            epochs_without_improvement = 0

        for layer in self.layers:
            layer.learning_rate = learning_rate

        for epoch in range(epochs):
            print('Epoch:', epoch + 1, '/', epochs)
            for i in range(0, X.shape[0] - batch_size, batch_size):
                self.input = X[i:i + batch_size]
                self.output = self.forward(self.input)
                d_Y = self.output - Y[:, i:i + batch_size]
                for layer in self.layers:
                    layer.learning_rate = 1 / (1 + (epoch+1) * self.decay_rate) * layer.learning_rate
                self.backward(d_Y)
                self.update()
                # 清除上一行输出
                print('\r', end='')
                # 打印进度条
                progress = int((i + 1) / (X.shape[0] - batch_size) * 40)
                print('[' + '=' * progress + '>' + '.' * (40 - progress - 1) + ']', end='')

            Y_hat = self.forward(X)
            loss = np.mean(np.square(Y - Y_hat))
            Y_hat = np.argmax(Y_hat, axis=0)
            Y_targ = np.argmax(Y, axis=0)
            accuracy = self.accuracy(Y_hat, Y_targ)

            # Check for improvement
            if patience is not None:
                if loss < best_loss:
                    best_loss = loss
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

            print(' -Loss:', loss, ' -Accuracy:', accuracy)

            # Check early stopping condition
            if patience is not None and epochs_without_improvement >= patience:
                print(f'Early stopping after {epoch + 1} epochs')
                break

    def summary(self):
        total_params = 0
        print("_________________________________________________________________")
        print("{:<25s}{:<30s}{:<10s}".format("Layer (type)", "Output Shape", "Param #"))
        print("=================================================================")
        for i, layer in enumerate(self.layers):
            if layer.name == 'Dropout' or layer.name == 'Flatten':
                continue
            output_shape = layer.output_shape()
            print("{:<25s}{:<30s}{:<10d}".format(f"{layer.name}_{i + 1}", output_shape, layer.get_params()))
            if i < len(self.layers) - 1:
                print("_________________________________________________________________")
            total_params += layer.get_params()
        print("=================================================================")
        print(f"Total params: {total_params}")
        print(f"Trainable params: {total_params}")
        print(f"Non-trainable params: 0")
        print("_________________________________________________________________")

    def get_params(self):
        total_params = 0
        for layer in self.layers:
            total_params += layer.get_params()
        return total_params

    def gradient_check(self, X, Y, epsilon=1e-7):
        print('Gradient Checking...')
        for layer in self.layers:
            if layer.name == 'Dropout':
                continue
            print('Checking', layer.name)
            layer.forward(X)
            layer.backward(Y - layer.output)
            for i, param in enumerate(layer.params):
                param += epsilon
                layer.forward(X)
                loss1 = np.mean(np.square(Y - layer.output))
                param -= 2 * epsilon
                layer.forward(X)
                loss2 = np.mean(np.square(Y - layer.output))
                param += epsilon
                numerical_gradient = (loss1 - loss2) / (2 * epsilon)
                print('Analytical Gradient:', layer.grads[i], 'Numerical Gradient:', numerical_gradient)

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
