import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
import matplotlib.pylab as plt
import numpy as np
from functions import *
from gradient import numerical_gradient


# 均方誤差
def mean_squared_error(y_hat, y):
    return 0.5 * np.sum((y_hat - y) ** 2)


# answer = 2
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
# guess = 2
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
# 0.09750000000000003
print(mean_squared_error(np.array(y), np.array(t)))
# guess = 7
y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
# 0.5975
print(mean_squared_error(np.array(y), np.array(t)))


# 交叉熵誤差
def cross_entropy_error(y_hat, y):
    delta = 1e-7
    # y 只在正確處為 1，其他地方都是 0
    # log(0) 為負無窮，為避免出現負無窮，加上極小值 delta
    return -np.sum(y * np.log(y_hat + delta))


# answer = 2
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
# guess = 2
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
# 0.510825457099338
print(cross_entropy_error(np.array(y), np.array(t)))
# guess = 7
y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
# 2.302584092994546
print(cross_entropy_error(np.array(y), np.array(t)))

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("x_train.shape:", x_train.shape, ", y_train.shape:", y_train.shape)
print("x_test.shape:", x_test.shape, ", y_test.shape:", y_test.shape)
# x_train.shape: (60000, 28, 28) , y_train.shape: (60000,)
# x_test.shape: (10000, 28, 28) , y_test.shape: (10000,)

x_train_normal = x_train.reshape(-1, 784) / 255
x_test_normal = x_test.reshape(-1, 784) / 255
y_train_OneHot = np_utils.to_categorical(y_train)
y_test_OneHot = np_utils.to_categorical(y_test)
print("x_train_normal.shape:", x_train_normal.shape, ", y_train_OneHot.shape:", y_train_OneHot.shape)
print("x_test_normal.shape:", x_test_normal.shape, ", y_test_OneHot.shape:", y_test_OneHot.shape)
# x_train_normal.shape: (60000, 784) , y_train_OneHot.shape: (60000, 10)
# x_test_normal.shape: (10000, 784) , y_test_OneHot.shape: (10000, 10)

train_size = x_train_normal.shape[0]
batch_size = 10
print("train_size:", train_size)
# train_size: 60000

# np.random.choice(60000, 10)会从0 到59999 之间随机选择10 个数字。
batch_mask = np.random.choice(train_size, batch_size)
# [39247 42320 38636 37229 16461  7848 16546 55897 36003 48516]
x_batch = x_train_normal[batch_mask]
y_batch = y_train_OneHot[batch_mask]
print("batch_mask")
print(batch_mask.shape)  # (10,)
print(batch_mask)
print("x_batch")
print(x_batch.shape)  # (10, 784)
print(x_batch)
print("y_batch")
print(y_batch.shape)  # (10, 10)
print(y_batch)


def crossEntropyError(y_hat, y):
    # log(0) 為負無窮，為避免出現負無窮，加上極小值 delta
    delta = 1e-7

    # 如果是一維資料，轉換成二維資料
    if y_hat.ndim == 1:
        y = y.reshape(1, y.size)
        y_hat = y_hat.reshape(1, y_hat.size)

    # 監督数据是 one-hot-vector 的情况下，轉換為標籤形式
    if y.size == y_hat.size:
        y = y.argmax(axis=1)

    batch_size = y_hat.shape[0]

    return -np.sum(np.log(y[np.arange(batch_size), t] + delta)) / batch_size


def differential(func, x):
    h = 1e-4
    plus = func(x + h)
    minus = func(x - h)
    derivative = (plus - minus) / (2 * h)
    return derivative


def numerical_diff(f, x):
    h = 1e-4  # 0.0001
    return (f(x + h) - f(x - h)) / (2 * h)


def function_1(x):
    return 0.01 * x ** 2 + 0.1 * x


# %%
x = np.arange(0.0, 20.0, 0.1)  # 以0.1为单位，从0到20的数组x
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x, y)
plt.show()
# %%
print("5\t", numerical_diff(function_1, 5))
print("10\t", numerical_diff(function_1, 10))


# %%
def function_2(x):
    return x[0] ** 2 + x[1] ** 2


def numerical_gradient(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)  # 生成和x形状相同的数组

    for idx in range(x.size):
        tmp_val = x[idx]
        # f(x+h)的计算，只修改 x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)
        # f(x-h)的计算，只修改 x[idx]
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val  # 还原值
    return grad


# %%
print(numerical_gradient(function_2, np.array([3.0, 4.0])))
print(numerical_gradient(function_2, np.array([0.0, 2.0])))
print(numerical_gradient(function_2, np.array([3.0, 0.0])))
# %%
# numerical_gradient(function_2, np.array([3.0, 0.0]))
matrix = np.random.randint(1, 10, size=(3, 2, 4))
for index, m in enumerate(matrix):
    print(index)
    print(m.shape)
    print(m)
    print()


# %%
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    print_step = step_num // 10
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
        if i % print_step == 0:
            print("step {}: {}".format(i, x))
    return x


# %%
init_x = np.array([-3.0, 4.0])
new_x = gradient_descent(function_2, init_x=init_x, lr=0.01, step_num=1000)
print(new_x)
# %%
# coding: utf-8
import numpy as np
import matplotlib.pylab as plt


def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    x_history = []

    for i in range(step_num):
        x_history.append(x.copy())

        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x, np.array(x_history)


def function_2(x):
    return x[0] ** 2 + x[1] ** 2


init_x = np.array([-3.0, 4.0])

lr = 0.1
step_num = 20
x, x_history = gradient_descent(function_2, init_x, lr=lr, step_num=step_num)
print(x_history)

plt.plot([-5, 5], [0, 0], '--b')
plt.plot([0, 0], [-5, 5], '--b')
plt.plot(x_history[:, 0], x_history[:, 1], 'or')

plt.xlim(-3.5, 3.5)
plt.ylim(-4.5, 4.5)
plt.xlabel("X0")
plt.ylabel("X1")
plt.show()
# %%
import sys, os

sys.path.append(os.path.join(os.getcwd(), "fish"))
from fish.activate_function import *


# %%
class simpleNet:
    def __init__(self, row, column):
        self.W = np.random.randn(row, column)  # 用高斯分布进行初始化

    def predict(self, x):
        return np.dot(x, self.W)

    def softmax(self, a):
        c = np.max(a)
        exp_a = np.exp(a - c)  # 溢出对策
        sum_exp_a = np.sum(exp_a)
        y = exp_a / sum_exp_a
        return y

    def cross_entropy_error(self, y, t):
        delta = 1e-7
        return -np.sum(t * np.log(y + delta))

    def numerical_gradient(self, f, x):
        h = 1e-4  # 0.0001
        grad = np.zeros_like(x)

        it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            tmp_val = x[idx]
            x[idx] = float(tmp_val) + h
            fxh1 = f(x)  # f(x+h)

            x[idx] = tmp_val - h
            fxh2 = f(x)  # f(x-h)
            grad[idx] = (fxh1 - fxh2) / (2 * h)

            x[idx] = tmp_val  # 还原值
            it.iternext()

        return grad

    def loss(self, x, t):
        z = self.predict(x)
        y = self.softmax(z)
        _loss = self.cross_entropy_error(y, t)
        return _loss


# %%
net = simpleNet(2, 3)
print(net.W)
# %%
x = np.array([0.6, 0.9])
p = net.predict(x)
print(p)
print(np.argmax(p))
# %% 正确解标签
t = np.array([0, 0, 1])
print(net.loss(x, t))


# %%
def f(W):
    return net.loss(x, t)


dW = net.numerical_gradient(x, net.W)
print(dW)


# %%

# %%
class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size,
                 weight_init_std=0.01):
        # 初始化权重
        self.params = {}
        self.params['W1'] = weight_init_std * \
                            np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * \
                            np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        return y

    # x:输入数据, t:监督数据
    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    # x:输入数据, t:监督数据
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        grads = {'W1': numerical_gradient(loss_W, self.params['W1']),
                 'b1': numerical_gradient(loss_W, self.params['b1']),
                 'W2': numerical_gradient(loss_W, self.params['W2']),
                 'b2': numerical_gradient(loss_W, self.params['b2'])}
        return grads

