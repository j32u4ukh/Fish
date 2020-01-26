import numpy as np


def tensorToMat2d(input_data, f_height, f_width, stride=1, padding=0):
    """
    im2col: 為避免卷積運算時，使用多層迴圈，將 4 維數組轉為 2 維數據

    :param input_data: 由(數據量, 通道, 高, 長)的4維數組構成的輸入數據
    :param f_height: 濾波器的高
    :param f_width: 濾波器的寬
    :param stride: 步幅
    :param padding: 填充
    :return: col : 2維數組
    """
    batch_size, channel, height, width = input_data.shape
    out_h = int((height + 2 * padding - f_height) / stride) + 1
    out_w = int((width + 2 * padding - f_width) / stride) + 1

    img = np.pad(input_data, [(0, 0), (0, 0), (padding, padding), (padding, padding)], 'constant')
    col = np.zeros((batch_size, channel, f_height, f_width, out_h, out_w))

    for y in range(f_height):
        y_max = y + stride * out_h
        for x in range(f_width):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y: y_max: stride, x: x_max: stride]

    # (batch_size, channel, f_height, f_width, out_h, out_w)
    # -> (batch_size, out_h, out_w, channel, f_height, f_width)
    # -> (batch_size * out_h * out_w, channel * f_height * f_width)
    col = col.transpose([0, 4, 5, 1, 2, 3]).reshape(batch_size * out_h * out_w, -1)
    return col


def mat2dToTensor(_tensor, input_shape, f_height, f_width, stride=1, padding=0):
    """
    col2im: tensorToMat2d 的反運算，將 2 維數組轉為 4 維數據

    :param _tensor:
    :param input_shape: 原始數據的形狀，例：(10, 1, 28, 28)
    :param f_height: 濾波器的高
    :param f_width: 濾波器的寬
    :param stride: 步幅
    :param padding: 填充
    :return: 4 維數據
    """
    batch_size, channel, height, width = input_shape.shape
    out_h = int((height + 2 * padding - f_height) / stride) + 1
    out_w = int((width + 2 * padding - f_width) / stride) + 1

    _tensor = _tensor.reshape(batch_size, out_h, out_w, channel, f_height, f_width)

    # _tensor.shape = (batch_size, channel, f_height, f_width, out_h, out_w)
    _tensor = _tensor.transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((batch_size,
                    channel,
                    height + 2 * padding + stride - 1,
                    width + 2 * padding + stride - 1))

    for y in range(f_height):
        y_max = y + stride * out_h
        for x in range(f_width):
            x_max = x + stride * out_w
            img[:, :, y: y_max: stride, x: x_max: stride] += _tensor[:, :, y, x, :, :]

    return img[:, :, padding: height + padding, padding: width + padding]
