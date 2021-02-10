'''
基于误差反向传播的两层网络实现，速度比数值微分求梯度快
'''
import sys
sys.path.append("F:\study\python\python_programming\Code_python_deeplearning")
from dataset.mnist import load_mnist
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from common.functions import sigmoid, softmax
from common.gradient import numerical_gradient
from collections import OrderedDict

# 显示图像
def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

# ----------交叉熵损失函数(mini-batch)-------- #
# 输入：预测结果y，实际标记t(需要时one_hot形式)
# 输出：交叉熵损失函数值
# ------------------------------------------- #
def cross_entropy_error(y, t):
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)
    batch_size = y.shape[0]  # mini-batch的大小
    return -np.sum(t * np.log(y+1e-7)) / batch_size

# ----------计算图乘法层实现-------- #
# 输入：两个数x y，反向传播时的上游梯度
# 输出：前向传播相乘结果out，反向传播时两个梯度dx dy
# -------------------------------- #
class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None
    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y
        return out
    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x
        return dx, dy

# ----------计算图加法层实现-------- #
# 输入：两个数x y，反向传播时的上游梯度
# 输出：前向传播相加结果out，反向传播时两个梯度dx dy
# -------------------------------- #
class AddLayer:
    def __init__(self):
        pass
    def forward(self, x, y):
        out = x + y
        return out
    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy

# ----------计算图ReLU层实现-------- #
# 输入：x，反向传播时的上游梯度
# 输出：前向传播结果out，反向传播时梯度dx
# -------------------------------- #
class ReLU:
    def __init__(self):
        self.Mask = None
    def forward(self, x):
        self.Mask = (x<=0)
        out = x.copy()
        out[self.Mask] = 0
        return out
    def backward(self, dout):
        dout[self.Mask] = 0
        dx = dout
        return dx

# ----------计算图sigmoid层实现-------- #
# 输入：x，反向传播时的上游梯度
# 输出：前向传播结果out，反向传播时梯度dx
# ------------------------------------ #
class sigmoid:
    def __init__(self):
        self.out = None
    def forward(self, x):
        out = sigmoid(x)
        self.out = out
        return out
    def backward(self, dout):
        dx = dout * self.out * (1-self.out)
        return dx

# ----------计算图Affine层(XW+B)实现-------- #
# 输入：横向量x，横向量偏置b，权重W(需要维度符合要求)，反向传播时的上游梯度，能满足mini-batch
# 输出：前向传播结果out，反向传播时梯度dx dW dB
# --------------------------------------------- #
class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None
    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b
        return out
    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        # dx = dx.reshape(*self.original_x_shape)  # 还原输入数据的形状（对应张量）
        return dx

# ----------计算图Affine层(XW+B)实现-------- #
# 输入：横向量x，标签t
# 输出：前向传播输出及其损失loss，反向传播时梯度dx
# --------------------------------------------- #
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None # softmax的输出
        self.t = None # 监督数据

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size: # 监督数据是one-hot-vector的情况
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        
        return dx

# --------简单的两层网络-------- #
# 输入：输入输出及隐藏层权重大小，输入横向量x，实际结果t
# 输出：预测结果predict，交叉熵损失结果loss
# ----------------------------- #
class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        # 权重初始化
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.random.randn(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.random.randn(output_size)
        # 生成层
        self.layers = OrderedDict() #  OrderedDict是有序字典，可以记住向字典里添加元素的顺序
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['ReLU'] = ReLU()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        # 输出层
        self.lastlayer = SoftmaxWithLoss()
    def predict(self, x):
        # 经过所有生成层
        for layer in self.layers.values():
            x = layer.forward(x)
        return x
    def loss(self, x, t):
        # 输出+损失函数计算
        y = self.predict(x)
        return self.lastlayer.forward(y, t)
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1) # 找到最大值对应的索引，即结果
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
    def numerical_gradient(self, x, t):
        # 数值微分法求梯度
        loss_W = lambda W: self.loss(x, t)
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        return grads
    def gradient(self, x, t):
        # forward
        self.loss(x, t)
        # backward
        dout = 1
        dout  = self.lastlayer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse() # 顺序反向
        for layer in layers:
            dout = layer.backward(dout)
        # 设定
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db
        return grads

if __name__=='__main__':
    # 获取数据
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=True)
    # 超参数
    iters_num = 10000
    batch_size = 100
    learning_rate = 0.1
    # 保存过程变量
    train_loss_list = []
    train_acc_list = []
    test_acc_list = []
    iter_per_epoch = max(x_train.shape[0]/batch_size, 1) #  平均每个epoch的重复次数

    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

    for i in range(iters_num):
        # 获取mini-batch
        batch_mask = np.random.choice(x_train.shape[0], batch_size)  # 随机选择batch的索引
        x_train_batch = x_train[batch_mask]
        t_train_batch = t_train[batch_mask]
        # 计算梯度
        grad = network.gradient(x_train_batch, t_train_batch)
        # 更新参数
        for key in ('W1', 'b1', 'W2', 'b2'):
            network.params[key] -= learning_rate*grad[key]
        # 计算损失
        loss = network.loss(x_train_batch, t_train_batch)
        train_loss_list.append(loss)
        # 计算每个epoch的识别精度
        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

    # 绘图展示中间结果
    plt.figure(1)
    markers = {'train': 'o', 'test': 's'}
    x = np.arange(len(train_acc_list))
    plt.plot(x, train_acc_list, label='train acc')
    plt.plot(x, test_acc_list, label='test acc', linestyle='--')
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.ylim(0, 1.0)
    plt.legend(loc='lower right')
    plt.show()

    plt.figure(2)
    plt.plot(np.arange(len(train_loss_list)), train_loss_list)
    plt.xlabel("iter")
    plt.ylabel("loss")
    plt.show()