from collections import OrderedDict

import matplotlib.pyplot as plt
import tensorflow.examples.tutorials.mnist.input_data as input_data

from multi_layer_net import MultiLayerNet
from multi_layer_net_extend import MultiLayerNetExtend
from optimizer import *
from optimizer import SGD
from trainer import Trainer
from util import smooth_curve


def f(x, y):
    return x ** 2 / 20.0 + y ** 2


def df(x, y):
    return x / 10.0, 2.0 * y


init_pos = (-7.0, 2.0)
params = {'x': init_pos[0], 'y': init_pos[1]}
grads = {'x': 0, 'y': 0}
# %%
optimizers = OrderedDict()
optimizers["SGD"] = SGD(lr=0.95)
optimizers["Momentum"] = Momentum(lr=0.1)
optimizers["AdaGrad"] = AdaGrad(lr=1.5)
optimizers["Adam"] = Adam(lr=0.3)

idx = 1
for key in optimizers:
    optimizer = optimizers[key]
    x_history = []
    y_history = []
    params['x'], params['y'] = init_pos[0], init_pos[1]

    for i in range(30):
        x_history.append(params['x'])
        y_history.append(params['y'])

        grads['x'], grads['y'] = df(params['x'], params['y'])
        optimizer.update(params, grads)

    x = np.arange(-10, 10, 0.01)
    y = np.arange(-5, 5, 0.01)

    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)

    # for simple contour line
    mask = Z > 7
    Z[mask] = 0

    # plot
    plt.subplot(2, 2, idx)
    idx += 1
    plt.plot(x_history, y_history, 'o-', color="red")
    plt.contour(X, Y, Z)
    plt.ylim(-10, 10)
    plt.xlim(-10, 10)
    plt.plot(0, 0, '+')
    # colorbar()
    # spring()
    plt.title(key)
    plt.xlabel("x")
    plt.ylabel("y")

plt.show()
# %% MNIST 資料集 比較優化方法差異

# %%
# 0:讀入MNIST數據==========
mnist = input_data.read_data_sets("data/MNIST_data/", one_hot=True)
x_train = mnist.train.images / 255
t_train = mnist.train.labels
x_test = mnist.test.images / 255
t_test = mnist.test.labels

train_size = x_train.shape[0]
batch_size = 128
max_iterations = 2000
# %% 1:進行實驗的設置==========
optimizers = {'SGD': SGD(), 'Momentum': Momentum(), 'AdaGrad': AdaGrad(), 'Adam': Adam()}
# optimizers['RMSprop'] = RMSprop()

networks = {}
train_loss = {}
for key in optimizers.keys():
    networks[key] = MultiLayerNet(
        input_size=784, hidden_size_list=[100, 100, 100, 100],
        output_size=10)
    train_loss[key] = []
# %% 2:開始訓練==========
for i in range(max_iterations):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    for key in optimizers.keys():
        grads = networks[key].gradient(x_batch, t_batch)
        optimizers[key].update(networks[key].params, grads)

        loss = networks[key].loss(x_batch, t_batch)
        train_loss[key].append(loss)

    if i % 100 == 0:
        print("===========" + "iteration:" + str(i) + "===========")
        for key in optimizers.keys():
            loss = networks[key].loss(x_batch, t_batch)
            print(key + ":" + str(loss))
# %% 3.繪制圖形==========
markers = {"SGD": "o", "Momentum": "x", "AdaGrad": "s", "Adam": "D"}
x = np.arange(max_iterations)
for key in optimizers.keys():
    plt.plot(x, smooth_curve(train_loss[key]), marker=markers[key], markevery=100, label=key)
plt.xlabel("iterations")
plt.ylabel("loss")
# plt.ylim(0, 1)
plt.legend()
plt.show()
# %% 初始權重差異比較


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def ReLU(x):
    return np.maximum(0, x)


def tanh(x):
    return np.tanh(x)


# %%
input_data = np.random.randn(1000, 100)  # 1000個數據
node_num = 100  # 各隱藏層的節點（神經元）數
hidden_layer_size = 5  # 隱藏層有5層
# %%
x = input_data
activations = {}  # 激活值的結果保存在這裏

for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i - 1]

    # 改變初始值進行實驗！
    #    w = np.random.randn(node_num, node_num) * 1
    #    w = np.random.randn(node_num, node_num) * 0.01
    #    w = np.random.randn(node_num, node_num) * np.sqrt(1.0 / node_num)
    w = np.random.randn(node_num, node_num) * np.sqrt(2.0 / node_num)

    a = np.dot(x, w)

    # 將激活函數的種類也改變，來進行實驗！
    #    z = sigmoid(a)
    z = ReLU(a)
    #    z = tanh(a)

    activations[i] = z
# %% 繪制直方圖
for i, a in activations.items():
    plt.subplot(1, len(activations), i + 1)
    plt.title(str(i + 1) + "-layer")
    if i != 0:
        plt.yticks([], [])
    # plt.xlim(0.1, 1)
    plt.ylim(0, 7000)
    plt.hist(a.flatten(), 30, range=(0, 1))
plt.show()

# %% 0:讀入MNIST數據==========
mnist = input_data.read_data_sets("data/MNIST_data/", one_hot=True)
x_train = mnist.train.images / 255
t_train = mnist.train.labels
x_test = mnist.test.images / 255
t_test = mnist.test.labels
# %%
train_size = x_train.shape[0]
batch_size = 500
max_iterations = 2000
# %% 1:進行實驗的設置==========
weight_init_types = {'std=0.01': 0.01, 'Xavier': 'sigmoid', 'He': 'relu'}
optimizer = SGD(lr=0.01)

networks = {}
train_loss = {}
for key, weight_type in weight_init_types.items():
    networks[key] = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100, 100],
                                  output_size=10, weight_init_std=weight_type)
    train_loss[key] = []
# %% 2:開始訓練==========
for i in range(max_iterations):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    for key in weight_init_types.keys():
        grads = networks[key].gradient(x_batch, t_batch)
        optimizer.update(networks[key].params, grads)

        loss = networks[key].loss(x_batch, t_batch)
        train_loss[key].append(loss)

    if i % 100 == 0:
        print("===========" + "iteration:" + str(i) + "===========")
        for key in weight_init_types.keys():
            loss = networks[key].loss(x_batch, t_batch)
            print(key + ":" + str(loss))
# %% 3.繪制圖形==========
markers = {'std=0.01': 'o', 'Xavier': 'x', 'He': 's'}
x = np.arange(max_iterations)
for key in weight_init_types.keys():
    plt.plot(x, smooth_curve(train_loss[key]), marker=markers[key], markevery=100, label=key)
plt.xlabel("iterations")
plt.ylabel("loss")
plt.ylim(2.225, 2.325)
plt.legend()
plt.show()
# %% <權重衰減> 為了再現過擬合，減少學習數據
x_train = x_train[:300]
t_train = t_train[:300]
# %% weight decay（權值衰減）的設定 =======================
# weight_decay_lambda = 0 # 不使用權值衰減的情況
weight_decay_lambda = 0.1
# %%
network = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100], output_size=10,
                        weight_decay_lambda=weight_decay_lambda)
optimizer = SGD(lr=0.01)
max_epochs = 201
train_size = x_train.shape[0]
batch_size = 100
iter_per_epoch = max(train_size / batch_size, 1)
# %%
train_loss_list = []
train_acc_list = []
test_acc_list = []
epoch_cnt = 0
for i in range(1000000000):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grads = network.gradient(x_batch, t_batch)
    optimizer.update(network.params, grads)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        print("epoch:" + str(epoch_cnt) + ", train acc:" + str(train_acc) + ", test acc:" + str(test_acc))

        epoch_cnt += 1
        if epoch_cnt >= max_epochs:
            break
# %% 3.繪制圖形==========
markers = {'train': 'o', 'test': 's'}
x = np.arange(max_epochs)
plt.plot(x, train_acc_list, marker='o', label='train', markevery=10)
plt.plot(x, test_acc_list, marker='s', label='test', markevery=10)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 0.2)
plt.legend(loc='lower right')
plt.show()
# %%

# %%
mnist = input_data.read_data_sets("data/MNIST_data/", one_hot=True)
x_train = mnist.train.images / 255
t_train = mnist.train.labels
x_test = mnist.test.images / 255
t_test = mnist.test.labels
# %% 定是否使用Dropuout，以及比例 ========================
use_dropout = True  # 不使用Dropout的情況下為False
dropout_ratio = 0.2
# %%
network = MultiLayerNetExtend(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100],
                              output_size=10, use_dropout=use_dropout, dropout_ration=dropout_ratio)
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=301, mini_batch_size=100,
                  optimizer='sgd', optimizer_param={'lr': 0.01}, verbose=True)
trainer.train()

train_acc_list, test_acc_list = trainer.train_acc_list, trainer.test_acc_list
# %% 繪制圖形==========
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, marker='o', label='train', markevery=10)
plt.plot(x, test_acc_list, marker='s', label='test', markevery=10)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()


class Ke:
    def __init__(self):
        print("init")

    def __call__(self, *args, **kwargs):
        print("__call__")
        for idx, arg in enumerate(args):
            print(idx, arg)


ke = Ke()("p1", "p2")
