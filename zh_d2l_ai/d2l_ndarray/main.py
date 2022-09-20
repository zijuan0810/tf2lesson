import tensorflow as tf

# x = tf.range(12)
# print(tf.reshape(x, (3, 4))) #转换为3x4的矩阵
# print(tf.reshape(x, (-1, 4))) #转换为3x4的矩阵
# print(tf.reshape(x, (3, -1))) #转换为3x4的矩阵

# 创建一个形状为（2,3,4）的张量，其中所有元素都设置为0
# print(tf.zeros((2, 3, 4)))
# 创建一个形状为（2,3,4）的张量，其中所有元素都设置为1
# print(tf.ones((2, 3, 4)))

#  以下代码创建一个形状为（3,4）的张量。 其中的每个元素都从均值为0、标准差为1的标准高斯分布（正态分布）中随机采样。
print(tf.random.normal(shape=[3, 4]))
