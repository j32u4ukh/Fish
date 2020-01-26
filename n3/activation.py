import numpy as np
from enum import Enum

from n3.activation_function import (
    softmax
)
from n3.loss_function import (
    crossEntropyError
)


class EActivation(Enum):
    Relu = "relu"
    Sigmoid = "sigmoid"
    Tanh = "tanh"


class Relu:
    def __init__(self):
        self.mask_to_zero = None

    def forward(self, x):
        self.mask_to_zero = (x <= 0)
        out = x.copy()  # Is this necessary?
        out[self.mask_to_zero] = 0
        return out

    def backward(self, dout):
        dout[self.mask_to_zero] = 0
        dx = dout  # Is this necessary?
        return dx


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1.0 / (1.0 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        """
        dx = dout * self.out**2 * exp(-x)
        ∵ self.out = 1 / (1 + exp(-x))
        ∴ 可將 dx 簡化成下式
        dx = dout * (1 - self.out) * self.out

        :param dout: 反向傳遞來的導數
        :return:
        """
        dx = dout * (1.0 - self.out) * self.out
        return dx


# https://medium.com/@14prakash/back-propagation-is-very-simple-who-made-it-complicated-97b794c97e5c
# https://datascience.stackexchange.com/questions/29735/how-to-apply-the-gradient-of-softmax-in-backprop
# class Softmax:
#     def __init__(self):
#         self.y_hat = None  # softmax的輸出
#
#     def forward(self, x):
#         self.y_hat = softmax(x)
#
#         return self.y_hat
#
#     def backward(self, dout):
#         """
#         gradient = y_hat_i - y_i
#
#         :param dout: 反向傳播回來的梯度值，扮演上方 y_i 的角色
#         :return: softmax 層的梯度值
#         """
#         #
#         dx = self.y_hat - dout
#
#         return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y_hat = None  # softmax的輸出
        self.y = None  # 監督數據

    def forward(self, x, y):
        self.y = y
        self.y_hat = softmax(x)
        self.loss = crossEntropyError(self.y, self.y_hat)

        return self.loss

    def backward(self):
        """
        在誤差函數為 crossEntropyError 的情況下，求取出來的反向梯度，
        才會如同下方所寫，此為刻意設計所得。

        :return:
        """
        batch_size = self.y.shape[0]

        # 監督數據是 one-hot encoding 的情況
        if self.y.size == self.y_hat.size:
            dx = (self.y_hat - self.y) / batch_size

        # 監督數據是類別值，而非 one-hot encoding 的情況
        else:
            dx = self.y_hat.copy()
            dx[np.arange(batch_size), self.y] -= 1
            dx = dx / batch_size

        return dx
