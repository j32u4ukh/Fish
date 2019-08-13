#%%
import numpy as np
import matplotlib.pyplot as plt
#%%
def step_function(x):
    return np.array(x > 0, dtype=np.int)
#%%
x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1) # 指定y轴的范围
plt.show()
#%%
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
#%%
x = np.array([-1.0, 1.0, 2.0])
print(sigmoid(x))
#%%
x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1) # 指定y轴的范围
plt.show()
#%%
def relu(x):
    return np.maximum(0, x)
#%%
x = np.arange(-5.0, 5.0, 0.1)
_step_function = step_function(x)
_sigmoid = sigmoid(x)
_relu = relu(x)
plt.plot(x, _step_function, label = "step_function")
plt.plot(x, _sigmoid, label = "sigmoid")
plt.plot(x, _relu, label = "relu")
plt.legend()
plt.show()
#%%
A = np.array([1, 2, 3, 4])
print("A", A)
# np.ndim 取得維度
print("np.ndim(A)", np.ndim(A))
print("A.shape", A.shape)
print("A.shape[0]", A.shape[0])
print("A.size", A.size)
#%%
B = np.array([[1,2], 
              [3,4], 
              [5,6]])
print(B)
print(np.ndim(B))
print(B.shape)
print("B.size", B.size)
#%%
A = np.array([[1,2], 
              [3,4]])
B = np.array([[5,6], 
              [7,8]])
print(A.shape)
print(B.shape)
print(np.dot(A, B))
#%%
A = np.array([[1, 2], 
              [3, 4], 
              [5, 6]])
B = np.array([[7], 
              [8]])
print("A.shape", A.shape)
print("B.shape", B.shape)
print("np.dot(A, B)")
print(np.dot(A, B))
#%% 3 层神经网络的实现
X = np.array([1.0, 0.5])
W1 = np.array([[0.1, 0.3, 0.5], 
               [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])
print(W1.shape) # (2, 3)
print(X.shape) # (2,)
print(B1.shape) # (3,)
A1 = np.dot(X, W1) + B1
#%%
Z1 = sigmoid(A1)
print(A1) # [0.3, 0.7, 1.1]
print(Z1) # [0.57444252, 0.66818777, 0.75026011]
#%%
W2 = np.array([[0.1, 0.4], 
               [0.2, 0.5], 
               [0.3, 0.6]])
B2 = np.array([0.1, 0.2])
print(Z1.shape) # (3,)
print(W2.shape) # (3, 2)
print(B2.shape) # (2,)
A2 = np.dot(Z1, W2) + B2
Z2 = sigmoid(A2)
print(A2) # [0.51615984 1.21402696]
print(Z2) # [0.62624937 0.7710107 ]
print(Z2.shape) # (2,)
#%%
def identity_function(x):
    return x
#%%
W3 = np.array([[0.1, 0.3], 
               [0.2, 0.4]])
B3 = np.array([0.1, 0.2])
A3 = np.dot(Z2, W3) + B3
Y = identity_function(A3) # 或者Y = A3
print(Y) # [0.31682708 0.69627909]
#%%
a = np.array([0.3, 2.9, 4.0])
exp_a = np.exp(a) # 指数函数
print(exp_a) # [ 1.34985881 18.17414537 54.59815003]
#%%
sum_exp_a = np.sum(exp_a) # 指数函数的和
print(sum_exp_a) # 74.1221542101633
#%%
y = exp_a / sum_exp_a
print(y) # [0.01821127 0.24519181 0.73659691]
print(np.sum(y)) # 1.0
#%%
def softmax(x):        
    x = x - np.max(x) # 防止數值溢出   
    exp_x = np.exp(x)
    sum_exp_x = np.sum(exp_x)
    y = exp_x / sum_exp_x
    return y
#%%
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("x_train.shape:", x_train.shape, ", y_train.shape:", y_train.shape)
print("x_test.shape:", x_test.shape, ", y_test.shape:", y_test.shape)
#x_train.shape: (60000, 28, 28) , y_train.shape: (60000,)
#x_test.shape: (10000, 28, 28) , y_test.shape: (10000,)
#%%
import pickle
def init_network():
    with open("data/sample_weight.pkl", "rb") as f:
        network = pickle.load(f)
    return network   
#%%
def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)
    return y
#%%
network = init_network()
#%%
x = x_test.reshape(-1, 784)
t = y_test.copy()
accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    p = np.argmax(y) # 获取概率最高的元素的索引
    if p == t[i]:
        accuracy_cnt += 1
print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
#%%
x_normalize = x / 255
accuracy_cnt = 0
for i in range(len(x_normalize)):
    y = predict(network, x_normalize[i])
    p = np.argmax(y) # 获取概率最高的元素的索引
    if p == t[i]:
        accuracy_cnt += 1
print("Accuracy:" + str(float(accuracy_cnt) / len(x_normalize)))
#%%
batch_size = 5 # 批数量
accuracy_cnt = 0
for i in range(0, len(x_normalize), batch_size):
    x_batch = x_normalize[i : i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p == t[i : i+batch_size])
    if i == 0:
        print("y_batch")
        print(y_batch)
        print("p")
        print(p)
        print("t[i : i+batch_size]")
        print(t[i : i+batch_size])
        print("p == t[i : i+batch_size]")
        print(p == t[i : i+batch_size])
print("Accuracy:" + str(float(accuracy_cnt) / len(x_normalize)))
#%%

#%%

#%%

#%%

#%%

#%%

