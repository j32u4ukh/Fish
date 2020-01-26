import numpy as np


def _numerical_gradient_1d(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val  # 还原值

    return grad


def numerical_gradient_2d(f, x):
    if x.ndim == 1:
        return _numerical_gradient_1d(f, x)
    else:
        grad = np.zeros_like(x)

        for idx, _x in enumerate(x):
            grad[idx] = _numerical_gradient_1d(f, _x)

        return grad


def numerical_gradient(f, x):
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


def test_nditer(x):
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        print(idx)
        it.iternext()


if __name__ == "__main__":
    def f(x):
        return x[0] ** 2 + x[1] ** 2


    x = np.array([3.0, 2.0])
    grad = numerical_gradient(f, x)
    print(grad)

    test_nditer(np.random.randint(0, 12, size=(3, 2, 2)))
