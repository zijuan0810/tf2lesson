# %matplotlib inline
import random
import tensorflow as tf
from d2l import tensorflow as d2l


def synthetic_data(w, b, num_examples):
    """生成y=Xw + b + 噪声"""
    # 生成随机正态分布数据[1000,2]，即1000行2列
    X = tf.random.normal(shape=(num_examples, w.shape[0]))
    print(X)

    # 为了满足矩阵乘法规则，将一维向量shape=(2,)转换为列向量shape=(2, 1)
    w1 = tf.reshape(w, (-1, 1))  # -1表示该维度通过计算得出，等价于tf.reshape(w, (2, 1))
    # 计算y=Xw + b
    y = tf.matmul(X, w1) + b
    # 添加符合正态分布的噪声
    y += tf.random.normal(shape=y.shape, stddev=0.01)
    y = tf.reshape(y, (-1, 1)) #这步可以不需要，因为上面已经生成了(1000, 1)向量
    return X, y


def data_iter(batch_size, features, labels):
    """ 函数接收批量大小、特征矩阵和标签向量作为输入，生成大小为batch_size的小批量。
    每个小批量包含一组特征和标签"""
    num_exaples = len(features)
    indices = list(range(num_exaples))
    # 这些样本是随机读取的，没有特定顺序
    random.shuffle(indices)
    for i in range(0, num_exaples, batch_size):
        j = tf.constant(indices[i: min(i + batch_size, num_exaples)])
        yield tf.gather(features, j), tf.gather(labels, j)


def linreg(X, w, b):
    """线性回归模型"""
    return tf.matmul(X, w) + b


def squared_loss(y_hat, y):
    """定义损失函数：均方损失"""
    return (y_hat - tf.reshape(y, y_hat.shape)) ** 2 / 2


def sgd(params, grads, lr, batch_size):
    """小批量随机梯度下降"""
    for param, grad in zip(params, grads):
        param.assign_sub(lr * grad / batch_size)


true_w = tf.constant([2, -3.4])
true_b = 4.2
features, lables = synthetic_data(true_w, true_b, 1000)
print('features: ', features.shape, '\nlablel:', lables.shape)
# d2l.plt.scatter(features[:, 1].numpy(), lables.numpy(), 1)
# d2l.plt.show()

batch_size = 10
for X, y in data_iter(batch_size, features, lables):
    print(X, '\n', y)
    break

"""
w = tf.Variable(tf.random.normal(shape=(2, 1), mean=0, stddev=0.001), trainable=True)
b = tf.Variable(tf.zeros(1), trainable=True)

# 开始训练
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, lables):
        with tf.GradientTape() as tape:
            l = loss(net(X, w, b), y) # X和y的小批量损失
        # 计算l关于[w,b]的梯度
        dw, db = tape.gradient(l, [w, b])
        # 使用参数的梯度更新参数
        sgd([w, b], [dw, db], lr, batch_size)
    train_l = loss(net(features, w, b), lables)
    print(f'epoch {epoch + 1}, loss {float(tf.reduce_mean(train_l)):f}')

print(f'w的估计误差：{true_w - tf.reshape(w, true_w.shape)}')
print(f'b的估计误差：{true_b - b}')
"""
