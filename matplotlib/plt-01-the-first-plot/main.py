import numpy as np
import math
from matplotlib import pyplot as plt

x = np.arange(0, math.pi * 2, 0.05)
y = np.sin(x)
# plt.plot(x, y)
# plt.xlabel("angle")
# plt.ylabel("sine")
# plt.title("sine wave")
# plt.show()

fig = plt.figure()
axes = fig.add_axes([0, 0, 1, 1])
axes.plot(x, y)
axes.set_title("sine wave")
axes.set_xlabel('angle')
axes.set_ylabel('sine')
fig.show()