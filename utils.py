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
    sum_list = np.sum(exp_x, axis=1)
    for i in range(x.shape[0]):
        exp_x[i, :] /= sum_list[i]
    return exp_x
