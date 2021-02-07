'''
基于MNIST数据集的前向传播程序
'''
import sys, os
sys.path.append("F:\study\python\python_programming\Code_python_deeplearning")
from dataset.mnist import load_mnist
import numpy as np
from PIL import Image
import pickle
from common.functions import sigmoid, softmax

# 显示图像
def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

# 获取数据(test)
def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True)
    
    return x_test, t_test

# 获取已经训练好的网络的参数
def init_network():
    with open("F:\study\python\python_programming\Code_python_deeplearning\ch03\sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    
    return network

# 带入网络参数进行预测
def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y

if __name__=='__main__':
    x, t = get_data()
    network = init_network()

    print("------------------单个处理---------------")
    accuracy_cnt = 0
    for i in range(len(x)):
        y = predict(network, x[i])
        p = np.argmax(y)  # 返回最大值的索引
        if p == t[i]:
            accuracy_cnt += 1
    print("Accuracy: " + str(float(accuracy_cnt)/len(x)))
    print("----------------------------------------")

    print("------------------批处理---------------")  # 批处理速度更快一些，减少数据读入的时间
    batch_size = 100 # 批处理大小
    accuracy_cnt = 0
    for i in range(0, len(x), batch_size):
        x_batch = x[i:i+batch_size]
        y_batch = predict(network, x_batch)
        p = np.argmax(y_batch, axis=1)  # 返回最大值的索引
        accuracy_cnt += np.sum(p == t[i:i+batch_size])
    print("Accuracy: " + str(float(accuracy_cnt)/len(x)))
    print("----------------------------------------")