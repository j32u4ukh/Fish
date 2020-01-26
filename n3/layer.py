import numpy as np
from n3.activation import EActivation
from n3.utils.image import (
    tensorToMat2d,
    mat2dToTensor
)


class Dense:
    """
    全連接層
    """
    def __init__(self, w, b):
        self.w = w
        self.b = b

        self.x = None
        self.origin_x_shape = None

        # 權重和偏置參數的導數
        self.dw = None
        self.db = None

    def forward(self, x):
        # 紀錄原始張量
        self.origin_x_shape = x.shape
        batch_size = x.shape[0]
        self.x = x.reshape(batch_size, -1)

        out = np.dot(self.x, self.w) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.w.T)
        self.dw = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        # 還原輸入數據的形狀（對應張量）
        dx = dx.reshape(*self.origin_x_shape)
        return dx


class BatchNormalization:
    """
    http://arxiv.org/abs/1502.03167
    """

    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum

        # Conv層的情況下為4維，全連接層的情況下為2維
        self.input_shape = None

        # 測試時使用的平均值和方差
        self.running_mean = running_mean
        self.running_var = running_var

        # backward時使用的中間數據
        self.batch_size = None
        self.xc = None
        self.xn = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, train_flg=True):
        self.input_shape = x.shape
        if x.ndim != 2:
            batch_size, channel, height, width = x.shape
            x = x.reshape(batch_size, -1)

        out = self.forward_(x, train_flg)

        return out.reshape(*self.input_shape)

    def forward_(self, x, train_flg):
        if self.running_mean is None:
            batch_size, n_dense = x.shape
            self.running_mean = np.zeros(n_dense)
            self.running_var = np.zeros(n_dense)

        if train_flg:
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc ** 2, axis=0)
            std = np.sqrt(var + 10e-7)
            xn = xc / std

            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            xc = x - self.running_mean
            xn = xc / np.sqrt(self.running_var + 10e-7)

        out = self.gamma * xn + self.beta
        return out

    def backward(self, dout):
        if dout.ndim != 2:
            batch_size, channel, height, width = dout.shape
            dout = dout.reshape(batch_size, -1)

        dx = self.backward_(dout)

        dx = dx.reshape(*self.input_shape)
        return dx

    def backward_(self, dout):
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size

        self.dgamma = dgamma
        self.dbeta = dbeta

        return dx


class Dropout:
    """
    http://arxiv.org/abs/1207.0580
    """

    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, is_training=True):
        if is_training:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask


class Convolution:
    def __init__(self, w, b, stride=1, padding=0):
        self.w = w
        # b.shape = (n_fliter, 1, 1)
        self.b = b
        self.stride = stride
        self.padding = padding

        # 中間數據（backward時使用）
        self.x = None
        self.mat_2d = None
        self.mat_w = None

        # 權重和偏置參數的梯度
        self.dw = None
        self.db = None

    def forward(self, x):
        # self.w.shape = (output_channel, input_channel, height, width)
        n_fliter, f_channel, f_height, f_width = self.w.shape
        batch_size, channel, height, width = x.shape
        out_h = 1 + int((height + 2 * self.padding - f_height) / self.stride)
        out_w = 1 + int((width + 2 * self.padding - f_width) / self.stride)

        # mat_2d.shape = (batch_size * out_h * out_w, channel * f_height * f_width)
        mat_2d = tensorToMat2d(x, f_height, f_width, self.stride, self.padding)

        # mat_w.shape = (n_fliter, channel * f_height * f_width).T
        #             = (channel * f_height * f_width, n_fliter)
        mat_w = self.w.reshape(n_fliter, -1).T

        # out.shape = (batch_size * out_h * out_w, n_fliter)
        out = np.dot(mat_2d, mat_w) + self.b

        # transpose(0, 3, 1, 2) 改變4維資料的軸的順序
        # out.shape = (batch_size, out_h, out_w, n_fliter)
        #          -> (batch_size, n_fliter, out_h, out_w)
        out = out.reshape(batch_size, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.mat_2d = mat_2d
        self.mat_w = mat_w

        return out

    def backward(self, dout):
        n_fliter, f_channel, f_height, f_width = self.w.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, n_fliter)

        self.db = np.sum(dout, axis=0)
        self.dw = np.dot(self.mat_2d.T, dout)
        self.dw = self.dw.transpose([1, 0]).reshape((n_fliter, f_channel, f_height, f_width))

        dcol = np.dot(dout, self.mat_w.T)
        dx = mat2dToTensor(dcol, self.x.shape, f_height, f_width, self.stride, self.padding)

        return dx


class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, padding=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.padding = padding

        self.x = None
        self.arg_max = None

    def forward(self, x):
        batch_size, channel, height, width = x.shape
        out_h = int(1 + (height - self.pool_h) / self.stride)
        out_w = int(1 + (width - self.pool_w) / self.stride)

        # 展開(1): mat_2d.shape = (batch_size * out_h * out_w, channel * pool_h * pool_w)
        mat_2d = tensorToMat2d(x, self.pool_h, self.pool_w, self.stride, self.padding)

        # mat_2d.shape = (batch_size * out_h * out_w * channel,  pool_h * pool_w)
        mat_2d = mat_2d.reshape(-1, self.pool_h * self.pool_w)

        # 最大值(2): out.shpae = (batch_size * out_h * out_w * channel,)
        out = np.max(mat_2d, axis=1)

        # argmax: 返回最大值所在的索引值，是在指定的軸上的索引值
        # arg_max.shpae = (batch_size * out_h * out_w * channel,)
        arg_max = np.argmax(mat_2d, axis=1)

        # 轉換(3)
        # (batch_size, out_h, out_w, channel) -> (batch_size, channel, out_h, out_w)
        out = out.reshape((batch_size, out_h, out_w, channel)).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        return out

    def backward(self, dout):
        # dout.shape = (batch_size, out_h, out_w, channel)
        dout = dout.transpose(0, 2, 3, 1)

        pool_size = self.pool_h * self.pool_w

        # dmax.shape = (batch_size * out_h * out_w * channel, self.pool_h * self.pool_w)
        dmax = np.zeros((dout.size, pool_size))

        # self.arg_max 是在 axis=1 上的最大值所在的索引值
        # arg_max.shpae = (batch_size * out_h * out_w * channel,)
        # arg_max.size = batch_size * out_h * out_w * channel
        # 索引值上的位置改成 dout 當中的數值，不影響 dmax 的 shape
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()

        # tuple 的相加，直接拼接到後面: (1, 2) + (3, 4) = (1, 2, 3, 4)
        # dmax.shape = (batch_size, out_h, out_w, channel, self.pool_h * self.pool_w)
        dmax = dmax.reshape(dout.shape + (pool_size,))

        # dcol.shape = (batch_size * out_h * out_w, channel * self.pool_h * self.pool_w)
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = mat2dToTensor(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.padding)

        return dx


def initWeight(input_size, output_size, scale=None, e_activation=None):
    # 使用ReLU的情況下推薦的初始值(paper: He)
    if e_activation == EActivation.Relu:
        scale = np.sqrt(2.0 / input_size)

    # 使用sigmoid的情況下推薦的初始值(paper: Xavier)
    # Xavier 的原始論文中，不只考慮了前一層的神經元數量，也考慮了後一層的數量，但這裡將它簡化
    elif e_activation == EActivation.Sigmoid or e_activation == EActivation.Tanh:
        scale = np.sqrt(1.0 / input_size)

    return scale * np.random.randn(input_size, output_size)

