'''
基于数值微分法梯度下降的简单网络学习
'''
import sys
sys.path.append("F:\study\python\python_programming\Code_python_deeplearning")
from dataset.mnist import load_mnist
import numpy as np
from PIL import Image
from common.functions import sigmoid, softmax


# 显示图像
def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

# ----------均方差损失函数(mini-batch)-------- #
# 输入：预测结果y，实际标记t(需要时one_hot形式)
# 输出：均方差损失函数值
# ------------------------------------------- #
def mean_square_error(y, t):
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)
    batch_size = y.shape[0]  # mini-batch的大小
    return 0.5 * np.sum((y - t)**2) / batch_size

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

# ----------中心差分法算数值微分-------- #
# 输入：函数f和自变量x
# 输出：数值微分(导数)
# ------------------------------------- #
def numerical_diff(f, x):
    h = 1e-4  # 微小偏移
    return (f(x+h) - f(x-h)) / (2*h)

# -----------数值微分求变量梯度-------- #
# 输入：函数f，自变量向量x对应1d，第二种适应多维度
# 输出：变量梯度
# ------------------------------------- #
def numerical_gradient_1d(f, x):
    x = x.astype(np.float64)  # 转为小数
    h = 1e-4  # 微小偏移
    grad = np.zeros_like(x)  # 生成和x形状相同的数组
    
    for ii in range(x.size):
        tmp_val = x[ii]
        # 算f(x+h)
        x[ii] = tmp_val + h
        fxh1 = f(x)
        # 算f(x-h)
        x[ii] = tmp_val - h
        fxh2 = f(x)
        # 算梯度
        grad[ii] = (fxh1 - fxh2) / (2*h)
        x[ii] = tmp_val # 数值还原
    return grad

def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 还原值
        it.iternext()   
        
    return grad

# -----------基于数值微分的梯度下降法-------- #
# 输入：函数f，初始自变量init_x，学习率lr，迭代次数step_num
# 输出：迭代后最优解对应变量x
# ------------------------------------------ #
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    x = x.astype(np.float64)  # 转为小数

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr*grad

    return x

# --------简单的单层网络-------- #
# 输入：测试输入横向量x，实际结果t
# 输出：预测结果predict，交叉熵损失结果loss
# ----------------------------- #
class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3) # 用高斯分布随机初始化权值W

    def predict(self, x):
        x = x.astype(np.float64)  # 转为小数
        return np.dot(x, self.W)

    def loss(self, x, t):
        x = x.astype(np.float64)  # 转为小数
        z = self.predict(x)
        y = softmax(z) # 输出层激活函数softmax
        loss = cross_entropy_error(y, t)
        return loss



if __name__=='__main__':
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=False, flatten=True, one_hot_label=True)
    batch_size = 10  # mini-batch大小
    batch_mask = np.random.choice(x_train.shape[0], batch_size)  # 随机选择batch的索引
    x_train_batch = x_train[batch_mask]
    t_train_batch = t_train[batch_mask]
    