'''
简单CNN实现
'''
from ast import NodeTransformer
import sys

from numpy.core.defchararray import not_equal
from numpy.core.numeric import outer
sys.path.append("F:\study\python\python_programming\Code_python_deeplearning")
from dataset.mnist import load_mnist
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from common.functions import sigmoid, softmax
from common.gradient import numerical_gradient
from collections import OrderedDict
from common.util import im2col, col2im
from common.layers import *
import pickle

# ---------最大池化层----------- #
# 输入：初始卷积核W及偏置b，卷积核填充pad，步幅stride，四维输入x，上游梯度dout
# 输出：前向传播输出结果out，反向传播梯度结果dout
# ----------------------------- #
class Convolution:
    def __init__(self, W, b, stride=0, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad
        # backward时使用的中间数据
        self.x = None
        self.col = None
        self.col_W = None
        # 权重梯度
        self.dW = None
        self.db = None
    def forward(self, x):
        FN, C, FH, FW = self.W.shape # 卷积核尺寸
        N, C, H, W = x.shape # mini-batch的输入图像尺寸
        # 计算卷积后图像尺寸
        out_h = 1 + int((H + 2*self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2*self.pad - FW) / self.stride)
        # 矩阵变换方便计算卷积
        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T
        out = np.dot(col, col_W) + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
        # 保存中间结果
        self.x = x
        self.col = col
        self.col_W = col_W
        return out
    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)
        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)
        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)
        return dx

# ---------卷积层----------- #
# 输入：池化层滤波器尺寸pool_h pool_w，步幅stride，填充pad
# 输出：前向传播输出结果out，反向传播梯度结果dout
# -------------------------- #
class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
        # 保存中间量
        self.x = None
        self.arg_max = None
    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h*self.pool_w) # 各个通道数据展开拼接
        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)
        # 中间量
        self.x = x
        self.arg_max = arg_max
        return out
    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)
        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,)) 
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        return dx

# ---------简单卷积网络----------- #
# 网络结构：conv - relu - pool - affine - relu - affine - softmax
# input_size : 输入大小（MNIST的情况下为784）
# hidden_size_list : 隐藏层的神经元数量的列表（e.g. [100, 100, 100]）
# output_size : 输出大小（MNIST的情况下为10）
# activation : 'relu' or 'sigmoid'
# weight_init_std : 指定权重的标准差（e.g. 0.01）
# 指定'relu'或'he'的情况下设定“He的初始值”
# 指定'sigmoid'或'xavier'的情况下设定“Xavier的初始值”
# ------------------------------- #
class SimpleConvNet:
    def __init__(self, input_dim=(1,28,28), 
                conv_param={'filter_num':30, 'filter_size':5, 'pad':0, 'stride':1}, 
                hidden_size=100, output_size=10, weight_init_std=0.01):
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        input_size = input_dim[1]
        cov_output_size = (input_size - filter_size + 2*filter_pad) / filter_stride + 1
        pool_output_size = int(filter_num * (cov_output_size/2) * (cov_output_size/2))
        # 权重初始化
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        self.params['b1'] = np.zeros(filter_num)
        self.params['W2'] = weight_init_std * np.random.randn(pool_output_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['W3'] = weight_init_std * np.random.randn(hidden_size, ouput_size)
        self.params['b3'] = np.zeros(output_size)
        # 中间生成层
        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'], 
                                            conv_param['stride'], conv_param['pad'])
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Affine1'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W3'], self.params['b3'])
        # 输出层+损失函数
        self.last_layer = SoftmaxWithLoss()
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x
    def loss(self, x, t):
        y = self.predict(x)
        return self.last_layer.forward(y, t)
    def accuracy(self, x, t, batch_size=100):
        if t.dim != 1:
            t = np.argmax(t, axis=1)
        acc = 0.0
        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt) 
        return acc / x.shape[0]
    def gradient(self, x, t):
        """求梯度（误差反向传播法）
        Parameters
        ----------
        x : 输入数据
        t : 教师标签
        Returns
        -------
        具有各层的梯度的字典变量
            grads['W1']、grads['W2']、...是各层的权重
            grads['b1']、grads['b2']、...是各层的偏置
        """
        # forward
        self.loss(x, t)
        # backward
        dout = 1
        dout = self.last_layer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        # 设定
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Conv1'].dW, self.layers['Conv1'].db
        grads['W2'], grads['b2'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W3'], grads['b3'] = self.layers['Affine2'].dW, self.layers['Affine2'].db
        return grads
    def save_params(self, file_name="params.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)
    def load_params(self, file_name="params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val
        for i, key in enumerate(['Conv1', 'Affine1', 'Affine2']):
            self.layers[key].W = self.params['W' + str(i+1)]
            self.layers[key].b = self.params['b' + str(i+1)]

if __name__=='__main__':
    # 获取数据
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=True)