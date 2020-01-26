import numpy as np

from n3.loss_function import (
    meanSquaredError,
    crossEntropyError
)


def meanSquaredErrorTest():
    # label = 2
    y = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    # predict = 2
    y_hat = np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])
    print("predict1 class:", np.argmax(y_hat))

    # mse = 0.09750000000000003
    print("meanSquaredError1:", meanSquaredError(y, y_hat))

    # predict = 7
    y_hat = np.array([0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0])
    print("predict2 class:", np.argmax(y_hat))

    # mse = 0.5975
    print("meanSquaredError2:", meanSquaredError(y, y_hat))


def crossEntropyErrorTest():
    # label = 2
    y = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])

    # predict = 2
    y_hat = np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])
    print("predict1 class:", np.argmax(y_hat))

    # cee = 0.510825457099338
    print("crossEntropyError:", crossEntropyError(y, y_hat))

    # predict = 7
    y_hat = np.array([0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0])
    print("predict2 class:", np.argmax(y_hat))

    # cee = 2.302584092994546
    print("crossEntropyError:", crossEntropyError(y, y_hat))


if __name__ == "__main__":
    # meanSquaredErrorTest()
    crossEntropyErrorTest()
