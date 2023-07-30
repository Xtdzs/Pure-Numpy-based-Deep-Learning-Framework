import numpy as np
import Preprocessing


class Model:
    def __init__(self, decay_rate=0, *args, **kwargs):
        self.total_output = None
        self.decay_rate = decay_rate
        self.layers = []
        self.input = None
        self.output = None

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, X):
        self.input = X
        for layer in self.layers:
            X = layer.forward(X)
        self.output = X
        return X

    def backward(self, Y):
        for layer in reversed(self.layers):
            Y = layer.backward(Y)
        return Y

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
            for i in range(0, X.shape[0] - batch_size + 1, batch_size):
                self.input = X[i:i + batch_size]
                self.output = self.forward(self.input)
                d_Y = self.output - Y[:, i:i + batch_size]
                for layer in self.layers:
                    layer.learning_rate = 1 / (1 + (epoch + 1) * self.decay_rate) * layer.learning_rate
                self.backward(d_Y)
                self.update()
                # 清除上一行输出
                print('\r', end='')
                # 打印进度条
                progress = int((i + 1) / (X.shape[0] - batch_size + 1) * 40)
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
            print("{:<25s}{:<30s}{:<10d}".format(f"{layer.name}_{layer.iteration}", output_shape, layer.get_params()))
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

    def accuracy(self, Y_hat, Y):
        return np.sum(Y_hat == Y) / len(Y)

    def update(self):
        pass
