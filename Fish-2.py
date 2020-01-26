import numpy as np
import matplotlib.pyplot as plt


"""
AND
(0, 0) -> 0
(1, 0) -> 0
(0, 1) -> 0
(1, 1) -> 1

Not AND
(0, 0) -> 1
(1, 0) -> 1
(0, 1) -> 1
(1, 1) -> 0

OR
(0, 0) -> 0
(1, 0) -> 1
(0, 1) -> 1
(1, 1) -> 1

XOR
(0, 0) -> 0
(1, 0) -> 1
(0, 1) -> 1
(1, 1) -> 0
"""


def AND(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp = x1 * w1 + x2 * w2
    if tmp <= theta:
        print(0)
    elif tmp > theta:
        print(1)


# %%
AND(0, 0)  # 輸出0
AND(1, 0)  # 輸出0
AND(0, 1)  # 輸出0
AND(1, 1)  # 輸出1
# %%
x = np.array([0, 1])
w = np.array([0.5, 0.5])
b = -0.7
# %%
print("w * x")
print(w * x)
print("np.sum(w*x)")
print(np.sum(w * x))
print("np.sum(w*x) + b")
print(np.sum(w * x) + b)
print("np.sum(np.multiply(x, w)) + b")
print(np.sum(np.multiply(x, w)) + b)

x = np.array([0, 1])
w = np.array([[0.5],
              [0.5]])
b = -0.7

y = x.dot(w) + b
print(y)


# %%
def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    elif tmp > 0:
        return 1


def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    elif tmp > 0:
        return 1


def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    elif tmp > 0:
        return 1


def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y


print(XOR(0, 0))  # 輸出0
print(XOR(1, 0))  # 輸出1
print(XOR(0, 1))  # 輸出1
print(XOR(1, 1))  # 輸出0


def plotPerceptron(perceptron, num=20):
    x1 = np.linspace(0, 1, num=20)
    x2 = np.linspace(0, 1, num=20)

    plt.title(str(perceptron))
    for h in x1:
        for v in x2:
            y = perceptron(h, v)
            if y == 0:
                plt.plot(h, v, 'b.')
            else:
                plt.plot(h, v, 'r.')
    plt.legend()
    plt.show()


plotPerceptron(AND)
plotPerceptron(NAND)
plotPerceptron(OR)
plotPerceptron(XOR)
