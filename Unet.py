# 按照unet的这个架构搭建而来
# https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/

import numpy as np


def calc_conv_output_size(input_h, input_w, filter_h, filter_w, stride=1, pad=0):
    """计算卷积输出特征图的高和宽"""
    out_h = (input_h + 2 * pad - filter_h) // stride + 1
    out_w = (input_w + 2 * pad - filter_w) // stride + 1
    return out_h, out_w


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """
    把输入张量展开成二维矩阵，方便后续用矩阵乘法实现卷积

    input_data shape: (N, C, H, W)
    return shape: (N * out_h * out_w, C * filter_h * filter_w)
    """
    N, C, H, W = input_data.shape
    out_h, out_w = calc_conv_output_size(H, W, filter_h, filter_w, stride, pad)

    # 只在高和宽两个维度补 0
    img = np.pad(
        input_data,
        [(0, 0), (0, 0), (pad, pad), (pad, pad)],
        mode="constant"
    )

    # 用来收集所有卷积窗口中的元素
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w),
                   dtype=input_data.dtype)

    # 遍历卷积核内部的每一个位置
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    # 调整维度顺序，然后把每个窗口拉平成一行
    col = col.transpose(0, 4, 5, 1, 2, 3)
    col = col.reshape(N * out_h * out_w, -1)

    return col


class Conv:
    def __init__(self, w, b, stride=1, pad=0):
        """
        w shape: (FN, C, FH, FW)
        b shape: (FN,)
        """
        self.w = w
        self.b = b
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        """
        x shape: (N, C, H, W)
        return shape: (N, FN, out_h, out_w)
        """
        N, C, H, W = x.shape
        FN, C_w, FH, FW = self.w.shape

        assert C == C_w, "Input channel must match filter channel"
       # 只有在输入数据通道数等于卷积核要求通道数时运行，不然报错
        out_h, out_w = calc_conv_output_size(
            H, W, FH, FW, self.stride, self.pad)

        # 1. 输入窗口展开: (N*out_h*out_w, C*FH*FW)
        x_col = im2col(x, FH, FW, self.stride, self.pad)
        # 2. 把每个卷积核拉平成一行，组成权重矩阵
        w_col = self.w.reshape(FN, -1)
        # 3. 矩阵乘法: (N*out_h*out_w, FN)
        out = x_col @ w_col.T + self.b.reshape(1, FN)
        # 4. 恢复成卷积输出形状: (N, FN, out_h, out_w)
        out = out.reshape(N, out_h, out_w, FN)
        out = out.transpose(0, 3, 1, 2)
        return out


class ReLU:
    def forward(self, x):
        return np.maximum(0, x)


class MaxPooling:  # 下采样
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        """
        x shape: (N, C, H, W)
        return shape: (N, C, out_h, out_w)
        """
        N, C, H, W = x.shape
        out_h, out_w = calc_conv_output_size(
            H, W, self.pool_h, self.pool_w, self.stride, self.pad)
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        # 拉一行
        col = col.reshape(-1, self.pool_h*self.pool_w)
        # maximum
        out = np.max(col, axis=1)
        # 恢复
        out = out.reshape(N, out_h, out_w, C)
        out = out.transpose(0, 3, 1, 2)
        return out


class Upsample:  # 上采样
    def __init__(self, scale=2):
        self.scale = scale

    def forward(self, x):
        """
        x shape: (N, C, H, W)
        return shape: (N, C, H*scale, W*scale)
        """
        x = np.repeat(x, repeats=self.scale, axis=2)  # axis-->H复制scale倍
        x = np.repeat(x, repeats=self.scale, axis=3)  # axis-->W
        return x


def concat_channel(x1, x2):
    # 沿通道维拼接
    return np.concatenate([x1, x2], axis=1)


class Doubleconv:
    def __init__(self, in_ch, out_ch):
        self.conv1 = Conv(np.random.randn(out_ch, in_ch, 3, 3) * 0.01,
                          np.zeros(out_ch), stride=1, pad=1)

        self.relu1 = ReLU()
        self.conv2 = Conv(np.random.randn(out_ch, out_ch, 3, 3) * 0.01,
                          np.zeros(out_ch), stride=1, pad=1)
        self.relu2 = ReLU()

    def forward(self, x):
        x = self.conv1.forward(x)
        x = self.relu1.forward(x)
        x = self.conv2.forward(x)
        x = self.relu2.forward(x)
        return x

# build unet


class unet:
    def __init__(self, in_ch=3, out_ch=1):
        self.enc1 = Doubleconv(in_ch, 4)
        self.enc2 = Doubleconv(4, 8)
        self.pool1 = self.pool2 = MaxPooling(2, 2, 2)
        self.bottleneck = Doubleconv(8, 16)
        self.up1 = Upsample(2)
        self.dec1 = Doubleconv(16 + 8, 8)  # 上采样和原来的拼接，通道数相加
        self.up2 = Upsample(2)
        self.dec2 = Doubleconv(8 + 4, 4)
        self.out_conv = Conv(np.random.randn(out_ch, 4, 1, 1) * 0.01,
                             np.zeros(out_ch), stride=1, pad=0)

    def forward(self, x):
        x1 = self.enc1.forward(x)
        p1 = self.pool1.forward(x1)
        x2 = self.enc2.forward(p1)
        p2 = self.pool2.forward(x2)
        # 瓶颈
        x3 = self.bottleneck.forward(p2)
        # upupup
        u1 = self.up1.forward(x3)
        c1 = concat_channel(u1, x2)
        d1 = self.dec1.forward(c1)
        u2 = self.up2.forward(d1)
        c2 = concat_channel(u2, x1)
        d2 = self.dec2.forward(c2)
        out = self.out_conv.forward(d2)
        return out


if __name__ == "__main__":
    # 仅在作为主程序时运行
    x = np.random.randn(2, 3, 32, 32)
    model = unet(in_ch=3, out_ch=1)
    y = model.forward(x)

    print("input shape:", x.shape)
    print("output shape:", y.shape)
