'''
优化网络学习的方法
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

# ----------随机梯度下降法(SGD)做优化求解-------- #
# 输入：学习率lr，待更新参数params，求得的梯度grads
# 输出：完成参数更新
# --------------------------------------------- #
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr
    def update(self, params, grads):
        for key in params.key():
            params[key] -= self.lr * grads[key]

# ----------SGD+momentum做优化求解-------- #
# 输入：学习率lr，摩擦系数momentum，待更新参数params，求得的梯度grads
# 输出：完成参数更新
# --------------------------------------- #
class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None
    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.item():
                self.v[key] = np.zeros_like(val)
        for key in params.key():
            self.v[key] = self.momentum*self.v[key] - self.lr*grads[key]
            params[key] += self.v[key]

# ----------------AdaGrad---------------- #
# 输入：学习率lr，待更新参数params，求得的梯度grads
# 输出：完成参数更新
# --------------------------------------- #
class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None
    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.item():
                self.h[key] = np.zeros_like(val)
        for key in params.key():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)
        
# ----------------Adam------------------- #
# 输入：学习率lr，参数beta1 beta2，待更新参数params，求得的梯度grads
# 输出：完成参数更新
# --------------------------------------- #
class Adam:
    """Adam (http://arxiv.org/abs/1412.6980v8)"""
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None
    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)       
        self.iter += 1
        lr_t  = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)              
        for key in params.keys():
            #self.m[key] = self.beta1*self.m[key] + (1-self.beta1)*grads[key]
            #self.v[key] = self.beta2*self.v[key] + (1-self.beta2)*(grads[key]**2)
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])           
            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)        
            #unbias_m += (1 - self.beta1) * (grads[key] - self.m[key]) # correct bias
            #unbisa_b += (1 - self.beta2) * (grads[key]*grads[key] - self.v[key]) # correct bias
            #params[key] += self.lr * unbias_m / (np.sqrt(unbisa_b) + 1e-7)

# ----------------Dropout------------------- #
# 输入：删除比例dropout_ratio，上游梯度dout
# 输出：对部分输出置0，部分反向传播梯度置0
# --------------------------------------- #
class Dropout:
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None
    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)
    def backward(self, dout):
        return dout * self.mask

if __name__=='__main__':
    # 获取数据
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=True)