import numpy as np


# 均方誤差
def meanSquaredError(y, y_hat):
    return 0.5 * np.sum((y - y_hat) ** 2)


# 交叉熵誤差
def crossEntropyError(y, y_hat):
    """
    error = sum( y_i * log(y_hat_i) )
    交叉熵誤差原須將相乘的數值累加起來，但 y 只在正確標籤處為 1，其他都是 0，
    因此相當於取得與正確標籤相同索引值的 y_hat 的 log 值。

    y = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    correct_index = 2

    y_hat = np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])
    error = log( y_hat[ correct_index ] )

    batch_error = sum( error_j ) / batch_size
    """
    # y 只在正確處為 1，其他地方都是 0
    # log(0) 為負無窮，為避免出現負無窮，加上極小值 delta
    delta = 1e-8

    if y_hat.ndim == 1:
        y = y.reshape(1, y.size)
        y_hat = y_hat.reshape(1, y_hat.size)

    # 監督數據是 one-hot encoding 的情況下，轉換為正確解標簽的索引
    if y_hat.size != y.size:
        y = y.argmax(axis=1)

    batch_size = y_hat.shape[0]

    """
    令 y = [2, 7, 0, 9, 4]，則 y_hat[np.arange(batch_size), y]
    則會產生 [ y[0, 2], y[1, 7], y[2, 0], y[3, 9], y[4, 4] ]。
    """
    return -np.sum(np.log(y_hat[np.arange(batch_size), y] + delta)) / batch_size
