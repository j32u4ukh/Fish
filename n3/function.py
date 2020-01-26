import numpy as np


def numericalGradient(f, x):
    """
    數值微分的優點是實現簡單，因此，一般情況下不太容易出錯。
    而誤差反向傳播法的實現很覆雜，容易出錯。
    所以，經常會比較數值微分的結果和誤差反向傳播法的結果，以確認誤差反向傳播法的實現是否正確。

    :param f: 網路架構中的 f 即為 loss function
    :param x:
    :return:
    """
    delta = 1e-8

    # 生成和 x 形狀相同的陣列
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        # 只修改 x[idx]，計算 f(x+h)
        x[idx] = tmp_val + delta
        fxh1 = f(x)
        # 只修改 x[idx]，計算 f(x-h)
        x[idx] = tmp_val - delta
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * delta)

        # 還原 x[idx]
        x[idx] = tmp_val
    return grad


def gradientDescent(f, x, lr=0.01, step_num=100):
    print_step = step_num // 10

    for i in range(step_num):
        grad = numericalGradient(f, x)
        x -= lr * grad
        if i % print_step == 0:
            print("step {}: {}".format(i, x))
    return x


if __name__ == "__main__":
    def numericalGradientTest():
        def func(x):
            return np.sum(x ** 2)

        x = np.linspace(1, 12, 12).reshape((3, 4))
        print(x.size)
        print(numericalGradient(func, np.array([3.0, 4.0])))


    def gradientDescentTest():
        def func(x):
            return np.sum(x ** 2)

        x = np.array([-3.0, 4.0])
        x_pron = gradientDescent(func, x, lr=1e-1, step_num=100)
        print(x_pron)


    # numericalGradientTest()
    # gradientDescentTest()

    x = np.linspace(1, 12, 12).reshape((2, 2, 3))
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished:
        idx = it.multi_index
        print(idx)
        it.iternext()
