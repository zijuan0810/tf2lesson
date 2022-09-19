import numpy as np
import tensorflow as tf

# 定义一个随机数 （标量）
random_float = tf.random.uniform(shape=())
print("random_float: ", random_float)
# 定义一个有2个元素的零向量
zero_vector = tf.zeros(shape=2)
print("zero_vector: ", zero_vector)
# 定义两个2x2的常量矩阵
A = tf.constant([[1.0, 2.0], [3.0, 4.0]])
B = tf.constant([[5.0, 6.0], [7.0, 8.0]])
# 查看矩阵A的形状、类型和值
print("矩阵A的形状: ", A.shape)
print("矩阵A的类型", A.dtype)
print("矩阵A的值", A.numpy())  # 张量的 numpy() 方法是将张量的值转换为一个 NumPy 数组。
print("A + B: ", tf.add(A, B).numpy())
print("A * B: ", tf.matmul(A, B).numpy())


# ============================================================================================================
# 使用tf.GradientTape() 计算函数 y(x) = x^2 在 x = 3 时的导数
def gradient_tape_square():
    x = tf.Variable(initial_value=3.0)
    with tf.GradientTape() as tape:  # 在 tf.GradientTape() 的上下文内，所有计算步骤都会被记录以用于求导
        y = tf.square(x)
    y_grad = tape.gradient(y, x)  # 计算y关于x的导数
    print(y, y_grad)


gradient_tape_square()


# ============================================================================================================
# 以下代码展示了如何使用 tf.GradientTape() 计算函数 L(w, b) = \|Xw + b - y\|^2 在 w = (1, 2)^T, b = 1 时分别对 w, b 的偏导数。
# 其中 X = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix},  y = \begin{bmatrix} 1 \\ 2\end{bmatrix}。
def gradient_tape_2():
    X = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    y = tf.constant([[1.0], [2.0]])
    w = tf.Variable(initial_value=[[1.0], [2.0]])
    b = tf.Variable(initial_value=1.0)
    with tf.GradientTape() as tape:
        L = tf.reduce_sum(tf.square(tf.matmul(X, w) + b - y))
    w_grad, b_grad = tape.gradient(L, [w, b])  # 计算L(w, b)关于w, b的偏导数
    print(w_grad, b_grad)


gradient_tape_2()


# ============================================================================================================
# 将数组归一化
def normal_array(data_arr):
    data_max = data_arr.max()
    data_min = data_arr.min()
    return (data_arr - data_min) / (data_max - data_min)


def linear_regression():
    x_raw = np.array([2013, 2014, 2015, 2016, 2017], dtype=np.float32)
    y_raw = np.array([12000, 14000, 15000, 16500, 17500], dtype=np.float32)
    X = normal_array(x_raw)  # (x_raw - x_raw.min()) / (x_raw.max() - x_raw.min())
    y = normal_array(y_raw)  # (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min())
    X = tf.constant(X)
    y = tf.constant(y)

    a = tf.Variable(initial_value=0.0)
    b = tf.Variable(initial_value=0.0)
    variables = [a, b]

    num_epoch = 10000
    optimizer = tf.keras.optimizers.SGD(learning_rate=5e-4)
    for e in range(num_epoch):
        # 使用tf.GradientTape()记录损失函数的梯度信息
        with tf.GradientTape() as tape:
            y_pred = a * X + b
            loss = tf.reduce_sum(tf.square(y_pred - y))
        # TensorFlow自动计算损失函数关于自变量（模型参数）的梯度
        grads = tape.gradient(loss, variables)
        # TensorFlow自动根据梯度更新参数
        optimizer.apply_gradients(grads_and_vars=zip(grads, variables))

    print("variables: ", variables)


linear_regression()
