import numpy as np
from n3.activation_function import (
    sigmoid,
    identityFunction
)


"""
Layer container has two types, sequencial and netwrok.
Model contains one or more layer containers.
"""


class Model:
    def __init__(self):
        self.net = {'W1': np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]]),
                    'b1': np.array([0.1, 0.2, 0.3]),
                    'W2': np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]]),
                    'b2': np.array([0.1, 0.2]),
                    'W3': np.array([[0.1, 0.3], [0.2, 0.4]]),
                    'b3': np.array([0.1, 0.2])}

    def forward(self, x):
        a1 = np.dot(x, self.net['W1']) + self.net['b1']
        z1 = sigmoid(a1)
        a2 = np.dot(z1, self.net['W2']) + self.net['b2']
        z2 = sigmoid(a2)
        a3 = np.dot(z2, self.net['W3']) + self.net['b3']
        y = identityFunction(a3)
        return y

    def predict(self, x, is_training=True):
        """
        如何寫介面讓子類別繼承就好？
        或是，以 delegate 的形式，事後給予函式的內涵即可？
        """
        return self.net

    def accuracy(self, x, y):
        y_hat = self.predict(x, is_training=False)
        y_hat = np.argmax(y_hat, axis=1)
        if y.ndim != 1:
            y = np.argmax(y, axis=1)

        accuracy = np.sum(y_hat == y) / float(x.shape[0])
        return accuracy
