# %matplotlib inline
import math
import time
import numpy as np
import tensorflow as tf
from d2l import tensorflow as d2l

"""
n = 10000
a = tf.ones(n)
b = tf.ones(n)

c = tf.Variable(tf.zeros(n))
timer = d2l.Timer()
for i in range(n):
    c[i].assign(a[i] + b[i])
print(f'{timer.stop():.5f}sec')

timer.start()
d = a + b
print(f'{timer.stop():.5f}sec')
"""


def normal(x, mu, sigma):
    """计算正态分布"""
    p = 1 / math.sqrt(2 * math.pi * sigma ** 2)
    return p * np.exp(-0.5 / sigma ** 2 * (x - mu) ** 2)


def visualize_normal(x, params):
    """可视化正态分布"""
    d2l.plot(x, [normal(x, mu, sigma) for mu, sigma in params],
             xlabel='x', ylabel='p(x)',
             legend=[f'mean {mu}, std {sigma}' for mu, sigma in params])


visualize_normal(np.arange(-7, 7, 0.01), params=[(0, 1), (0, 2), (3, 1)])
