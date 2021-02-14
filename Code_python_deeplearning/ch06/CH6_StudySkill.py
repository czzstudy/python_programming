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

# ---------Batch Normalization----------- #
# 输入：动量参数momentum
# 输出：训练参数gamma, beta
# --------------------------------------- #
class BatchNormalization:
    """
    http://arxiv.org/abs/1502.03167
    """
    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None # Conv层的情况下为4维，全连接层的情况下为2维  
        # 测试时使用的平均值和方差
        self.running_mean = running_mean
        self.running_var = running_var    
        # backward时使用的中间数据
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None
    def forward(self, x, train_flg=True):
        self.input_shape = x.shape
        if x.ndim != 2:
            N, C, H, W = x.shape
            x = x.reshape(N, -1)
        out = self.__forward(x, train_flg)
        return out.reshape(*self.input_shape)         
    def __forward(self, x, train_flg):
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)                   
        if train_flg:
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc**2, axis=0)
            std = np.sqrt(var + 10e-7)
            xn = xc / std
            
            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1-self.momentum) * var            
        else:
            xc = x - self.running_mean
            xn = xc / ((np.sqrt(self.running_var + 10e-7)))
        out = self.gamma * xn + self.beta 
        return out
    def backward(self, dout):
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)

        dx = self.__backward(dout)
        dx = dx.reshape(*self.input_shape)
        return dx
    def __backward(self, dout):
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