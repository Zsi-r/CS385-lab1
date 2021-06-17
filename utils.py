import numpy as np
import time
import os


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

# def sigmoid(x):
#     if x >= 0:  # 对sigmoid函数的优化，避免了出现极大的数据溢出
#         return 1.0 / (1 + np.exp(-x))
#     else:
#         return np.exp(x) / (1 + np.exp(x))


def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)
