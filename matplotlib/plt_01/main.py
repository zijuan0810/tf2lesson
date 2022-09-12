import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


# A simple example
fig, ax = plt.subplots()
ax.plot([1, 2, 3, 4], [1, 4, 2, 3])
ax.set_title('标题')
plt.show()


# Figure
# fig = plt.figure()
# fig, ax = plt.subplot()
# fig, axs = plt.subplots(2, 2)