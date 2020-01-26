import numpy as np


def smooth_curve(x):
    """用於使損失函數的圖形變圓滑

    參考：http://glowingpython.blogspot.jp/2012/02/convolution-with-numpy.html
    """
    window_len = 11
    s = np.r_[x[window_len - 1:0:-1], x, x[-1:-window_len:-1]]
    w = np.kaiser(window_len, 2)
    y = np.convolve(w / w.sum(), s, mode='valid')
    return y[5:len(y) - 5]


def shuffle_dataset(x, t):
    """打亂數據集

    Parameters
    ----------
    x : 訓練數據
    t : 監督數據

    Returns
    -------
    x, t : 打亂的訓練數據和監督數據
    """
    permutation = np.random.permutation(x.shape[0])
    x = x[permutation, :] if x.ndim == 2 else x[permutation, :, :, :]
    t = t[permutation]

    return x, t


def conv_output_size(input_size, filter_size, stride=1, pad=0):
    return (input_size + 2 * pad - filter_size) / stride + 1


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """

    :param input_data: 由(數據量, 通道, 高, 長)的4維數組構成的輸入數據
    :param filter_h: 濾波器的高
    :param filter_w: 濾波器的寬
    :param stride: 步幅
    :param pad: 填充
    :return: col : 2維數組
    """
    N, C, H, W = input_data.shape
    #    // >> 整除 EX 5//2 = 2；5/2=2.5
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    # (batch_size * out_h * out_w, channel * f_height * f_width)
    col = col.transpose([0, 4, 5, 1, 2, 3]).reshape(N * out_h * out_w, -1)
    return col


# %%
def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """

    Parameters
    ----------
    col :
    input_shape : 輸入數據的形狀（例：(10, 1, 28, 28)）
    filter_h :
    filter_w
    stride
    pad

    Returns
    -------

    """
    N, C, H, W = input_shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]


if __name__ == "__main__":
    value = conv_output_size(input_size=7, filter_size=3, stride=3, pad=2)
    print(value)
