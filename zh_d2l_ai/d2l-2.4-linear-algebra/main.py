import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib_inline import backend_inline
from d2l import tensorflow as d2l


def f(x):
    return 3 * x ** 2 - 4 * x


def numerical_lim(f, x, h):
    return (f(x + h) - f(x)) / h


h = 0.1
for i in range(5):
    print(f'h={h:.5f}, numerical_lim={numerical_lim(f, 1, h):.5f}')
    h *= 0.1

x = np.arange(0, 3, 0.1)
d2l.plot(x, [f(x), 2 * x - 3], 'x', 'f(x)', legend=['f(x)', 'Tangent line (x=1)'])
