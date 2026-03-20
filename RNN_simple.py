""" 不用pytorch实现的RNN模型，从XOR开始"""
import numpy as np


class SIMPLE_RNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # 下面这些都是初始化随机生成的权重和偏置
        # 输入 -> 隐藏
        self.W_xh = np.random.randn(hidden_size, input_size) * 0.1
        # 隐藏 -> 隐藏
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.1
        # 隐藏 -> 输出
        self.W_hy = np.random.randn(output_size, hidden_size) * 0.1
        # 隐藏层偏置
        self.b_h = np.zeros((hidden_size, 1))
        # 输出层偏置
        self.b_y = np.zeros((output_size, 1))

        self.zero_grad()

    # 激活函数（可以换 tanh、sigmoid 等）
    def ReLU(self, x):
        return np.maximum(0, x)

    def ReLU_derivative(self, x):
        return (x > 0).astype(float)

    def init_hidden(self):
        return np.zeros((self.hidden_size, 1))

    def zero_grad(self):
        self.dW_xh = np.zeros_like(self.W_xh)
        self.dW_hh = np.zeros_like(self.W_hh)
        self.dW_hy = np.zeros_like(self.W_hy)
        self.db_h = np.zeros_like(self.b_h)
        self.db_y = np.zeros_like(self.b_y)

    def forward_step(self, x_t, h_prev):
        z_t = np.dot(self.W_xh, x_t) + np.dot(self.W_hh, h_prev) + self.b_h
        h_t = self.ReLU(z_t)
        y_t = np.dot(self.W_hy, h_t) + self.b_y
        cache = (x_t, h_prev, z_t, h_t, y_t)
        return h_t, y_t, cache

    # 返回非线性处理后的隐藏状态和输出
    # 前向传播
    def forward(self, x_seq):  # 假设 x_seq 是列表
        h = self.init_hidden()  # 初始化隐藏状态
        hs = []  # 用于存储每个时间步的隐藏状态
        ys = []  # 用于存储每个时间步的输出
        caches = []  # 训练时反向传播要用到

        for x_t in x_seq:
            h, y, cache = self.forward_step(x_t, h)
            hs.append(h)
            ys.append(y)
            caches.append(cache)
        return hs, ys, caches

    def loss_function(self, y_pred_seq, y_true_seq):
        # mse（这里按整个序列的平均损失来算）
        loss = 0.0
        for y_pred, y_true in zip(y_pred_seq, y_true_seq):
            loss += np.mean((y_pred - y_true) ** 2)
        return loss / len(y_pred_seq)

    def output_loss_gradient(self, y_pred, y_true):
        return 2 * (y_pred - y_true) / y_pred.size

    #反向传播
    def backward(self, y_true_seq, caches):
        self.zero_grad()
        dh_next = np.zeros((self.hidden_size, 1))

        for t in reversed(range(len(caches))):
            x_t, h_prev, z_t, h_t, y_t = caches[t]
            y_true_t = y_true_seq[t]

            # 输出层梯度
            dy = self.output_loss_gradient(y_t, y_true_t)
            self.dW_hy += np.dot(dy, h_t.T)
            self.db_y += dy

            # 当前时间步的隐藏层梯度 = 输出层传回来的 + 下一时刻传回来的
            dh = np.dot(self.W_hy.T, dy) + dh_next
            dz = dh * self.ReLU_derivative(z_t)

            self.dW_xh += np.dot(dz, x_t.T)
            self.dW_hh += np.dot(dz, h_prev.T)
            self.db_h += dz

            dh_next = np.dot(self.W_hh.T, dz)

    def clip_gradients(self, max_value=5.0):
        for grad in [self.dW_xh, self.dW_hh, self.dW_hy, self.db_h, self.db_y]:
            np.clip(grad, -max_value, max_value, out=grad)

    def update_parameters(self, learning_rate):
        self.W_xh -= learning_rate * self.dW_xh
        self.W_hh -= learning_rate * self.dW_hh
        self.W_hy -= learning_rate * self.dW_hy
        self.b_h -= learning_rate * self.db_h
        self.b_y -= learning_rate * self.db_y

    def train_one_sample(self, x_seq, y_true_seq, learning_rate):
        hs, ys, caches = self.forward(x_seq)
        loss = self.loss_function(ys, y_true_seq)
        self.backward(y_true_seq, caches)
        self.clip_gradients()
        self.update_parameters(learning_rate)
        return loss, hs, ys

    def fit(self, X_train, Y_train, epochs=1000, learning_rate=0.01, print_every=100):
        loss_history = []

        for epoch in range(epochs):
            total_loss = 0.0

            for x_seq, y_seq in zip(X_train, Y_train):
                loss, _, _ = self.train_one_sample(x_seq, y_seq, learning_rate)
                total_loss += loss

            avg_loss = total_loss / len(X_train)
            loss_history.append(avg_loss)

            if (epoch + 1) % print_every == 0 or epoch == 0:
                print(f"epoch {epoch + 1}/{epochs}, loss={avg_loss:.6f}")

        return loss_history

    def predict(self, x_seq):
        _, ys, _ = self.forward(x_seq)
        return ys

#为什么没有把训练集和测试集分开？因为这个py主要是演示rnn框架的，我觉得这个部分并不是很重要。
def build_running_xor_dataset():
    X_train = []
    Y_train = []

    samples = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1], 
    ]
    for seq in samples:
        x_seq = [np.array([[value]], dtype=float) for value in seq]

        running_xor = []
        current_value = 0
        for value in seq:
            current_value = current_value ^ value
            running_xor.append(np.array([[current_value]], dtype=float))

        X_train.append(x_seq)
        Y_train.append(running_xor)

    return X_train, Y_train


if __name__ == "__main__":
    np.random.seed(42)

    X_train, Y_train = build_running_xor_dataset()

    rnn = SIMPLE_RNN(input_size=1, hidden_size=8, output_size=1)
    rnn.fit(X_train, Y_train, epochs=2000, learning_rate=0.05, print_every=200)

    print("\n训练完成后的预测结果：")
    for x_seq, y_seq in zip(X_train, Y_train):
        y_pred_seq = rnn.predict(x_seq)
        x_values = [int(x[0, 0]) for x in x_seq]
        y_true_values = [round(float(y[0, 0]), 4) for y in y_seq]
        y_pred_values = [round(float(y[0, 0]), 4) for y in y_pred_seq]
        print(f"输入序列: {x_values}, 目标: {y_true_values}, 预测: {y_pred_values}")
