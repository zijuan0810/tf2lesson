import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


# A simple example
# fig, ax = plt.subplots()
# ax.plot([1, 2, 3, 4], [1, 4, 2, 3])
# ax.set_title('标题')
# fig.show()


# Figure
# fig = plt.figure()
# fig, ax = plt.subplot()
# fig, axs = plt.subplots(2, 2)

# Types of inputs to plotting functions
def types_of_inputs_to_plotting_functions():
    np.random.seed(19680801)
    data = {'a': np.arange(50),
            'c': np.random.randint(0, 50, 50),
            'd': np.random.randn(50)}
    data['b'] = data['a'] + 10 * np.random.randn(50)
    data['d'] = np.abs(data['d']) * 100

    fig, ax = plt.subplots(figsize=(5, 2.7), layout='constrained')
    ax.scatter('a', 'b', c='c', s='d', data=data)
    ax.set_title('Types of inputs to plotting functions')
    ax.set_xlabel('entry a')
    ax.set_ylabel('entry b')
    fig.show()


types_of_inputs_to_plotting_functions()


# use the OO-style
def use_the_OO_style():
    x = np.linspace(0, 2, 100)  # Sample data.

    # Note that even in the OO-style, we use `.pyplot.figure` to create the Figure.
    fig, ax = plt.subplots(figsize=(5, 2.7), layout='constrained')
    ax.plot(x, x, label='linear')  # Plot some data on the axes.
    ax.plot(x, x ** 2, label='quadratic')  # Plot more data on the axes...
    ax.plot(x, x ** 3, label='cubic')  # ... and some more.
    ax.set_xlabel('x轴label')  # Add an x-label to the axes.
    ax.set_ylabel('y轴label')  # Add a y-label to the axes.
    ax.set_title("Simple Plot")  # Add a title to the axes.
    ax.legend()  # Add a legend.
    fig.show()


use_the_OO_style()
