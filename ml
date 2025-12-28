import matplotlib.pyplot as plt
import numpy as np

d1 = np.load("./data/mnist-data.npz")
d2 = np.load("./data/spam-data.npz")

#mnist-data.npz 和 spam-data.npz 文件内部包含了多个数组，这些数组可能存储了训练数据、测试数据、标签等信息。通过 np.load 加载后，d1 和 d2 实际上是 NpzFile 对象，可以像字典一样通过键访问里面的数组。
#例如，d1['training_data'] 可以用于访问 mnist 数据集的训练数据部分。

../上一级目录
./当前目录
