# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt
import _pickle as pickle


# 初始化权重
def weight_variable(shape,name_str):
    initial = tf.truncated_normal(shape, stddev=0.13)
    # 这是一个截断的产生正太分布的函数，
    # 就是说产生正太分布的值如果与均值的差值大于两倍的标准差，那就重新生成
    return tf.Variable(initial,name=name_str)

# 初始化偏置项
def bias_variable(shape,name_str):
    initial = tf.constant(0.13, shape=shape)
    return tf.Variable(initial,name=name_str)

#卷积过程
def conv2d(x,w,strides=[1,2,2,1]):
    return tf.nn.conv2d(x,w,
                        strides,padding='SAME')
def depthwise_conv2d(x,w,strides=[1,2,2,1]):
    return tf.nn.depthwise_conv2d(x,w,
                                  strides,padding='SAME')
#池化过程
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],
                        strides=[1,2,2,1],padding='SAME')


def unpickle(filename):
    '''解压数据'''
    # Python2
    # with open(filename) as f:
    #     d = pickle.load(f)
    #     return d

    # Python3
    with open(filename, 'rb') as f:
        d = pickle.load(f)
        #d = pickle.load(f, encoding='latin1')
        return d


def onehot(labels):
    '''one-hot 编码'''
    n_sample = len(labels)
    n_class = max(labels) + 1
    onehot_labels = np.zeros((n_sample, n_class))
    onehot_labels[np.arange(n_sample), labels] = 1
    return onehot_labels

def one_hot1(x, n):
    """
    convert index representation to one-hot representation
    """
    x = np.array(x)
    assert x.ndim == 10
    return np.eye(n)[x]

