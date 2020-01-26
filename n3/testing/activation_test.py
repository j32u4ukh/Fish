import numpy as np
from n3.activation_function import (
    sigmoid,
    stepFunction,
    relu
)
import matplotlib.pyplot as plt


def plotFunction(func, x=None):
    if x is None:
        x = np.arange(-3.0, 3.0, 0.1)

    y = func(x)
    plt.scatter(x, x, c="b", label="origin")
    plt.scatter(x, y, c="r", label="activated")
    plt.legend(loc="best")
    plt.show()


def stepFunctionTest():
    plotFunction(stepFunction)


def sigmoidTest():
    plotFunction(sigmoid)


def reluTest():
    plotFunction(relu)


if __name__ == "__main__":
    # stepFunctionTest()
    # sigmoidTest()
    reluTest()
