from collections import OrderedDict

import tensorflow.examples.tutorials.mnist.input_data as input_data

from gradient import numerical_gradient
from layers import *


class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y
        return out

    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x
        return dx, dy


# %%
class AddLayer:
    def __init__(self):
        pass

    @staticmethod
    def forward(x, y):
        out = x + y
        return out

    @staticmethod
    def backward(dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy


apple = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1

mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
add_apple_orange_layer = AddLayer()
mul_tax_layer = MulLayer()

apple_price = mul_apple_layer.forward(apple, apple_num)
price = mul_tax_layer.forward(apple_price, tax)

print(price)  # 220
# backward
dprice = 1
dapple_price, dtax = mul_tax_layer.backward(dprice)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)
print(dapple, dapple_num, dtax)  # 2.2 110 200
# %% forward
apple_price = mul_apple_layer.forward(apple, apple_num)  # (1)
orange_price = mul_orange_layer.forward(orange, orange_num)  # (2)
all_price = add_apple_orange_layer.forward(apple_price, orange_price)  # (3)
price = mul_tax_layer.forward(all_price, tax)  # (4)
# %%
print(price)  # 715
# %% backward
dprice = 1
dall_price, dtax = mul_tax_layer.backward(dprice)  # (4)
dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price)  # (3)
dorange, dorange_num = mul_orange_layer.backward(dorange_price)  # (2)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)  # (1)
# %%
print(dapple_num, dapple, dorange, dorange_num, dtax)  # 110 2.2 3.3 165 650


class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size,
                 weight_init_std=0.01):
        # 初始化權重
        self.params = {'W1': weight_init_std * np.random.randn(input_size, hidden_size),
                       'b1': np.zeros(hidden_size),
                       'W2': weight_init_std * np.random.randn(hidden_size, output_size),
                       'b2': np.zeros(output_size)}

        # 生成層
        # OrderedDict是有序字典，
        # “有序”是指它可以記住向字典裏添加元素的順序。
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    # x:輸入數據, t:監督數據
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    # x:輸入數據, t:監督數據
    def numerical_gradient(self, x, t):
        loss_w = lambda W: self.loss(x, t)

        grads = {'W1': numerical_gradient(loss_w, self.params['W1']),
                 'b1': numerical_gradient(loss_w, self.params['b1']),
                 'W2': numerical_gradient(loss_w, self.params['W2']),
                 'b2': numerical_gradient(loss_w, self.params['b2'])}

        return grads

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 設定
        grads = {'W1': self.layers['Affine1'].dW,
                 'b1': self.layers['Affine1'].db,
                 'W2': self.layers['Affine2'].dW,
                 'b2': self.layers['Affine2'].db}

        return grads


mnist = input_data.read_data_sets("data/MNIST_data/", one_hot=True)
x_train = mnist.train.images / 255
t_train = mnist.train.labels
x_test = mnist.test.images / 255
t_test = mnist.test.labels
# %%
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
# %%
x_batch = x_train[:3]
t_batch = t_train[:3]
# %%
grad_numerical = network.numerical_gradient(x_batch, t_batch)
grad_backprop = network.gradient(x_batch, t_batch)
# %%
for key in grad_numerical.keys():
    diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
    print(key + ":" + str(diff))
# %% 使用誤差反向傳播法的學習　數據沿用前面的
iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1
train_loss_list = []
train_acc_list = []
test_acc_list = []
iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    # 通過誤差反向傳播法求梯度
    grad = network.gradient(x_batch, t_batch)
    # 更新
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(train_acc, test_acc)


import numpy as np

img = np.linspace(1, 60, 60).reshape((3, 4, 5))
print("img.shape:", img.shape)
print(img)
