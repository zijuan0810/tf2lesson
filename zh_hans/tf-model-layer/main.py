import tensorflow as tf

# from tensorflow.python import keras


X = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
y = tf.constant([[10.0], [20.0]])


class Linear(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # 该层包含权重矩阵 kernel = [input_dim, units]和偏置向量 bias = [units]两个可训练变量，对应于 f(AW + b) 中的 W 和 b。
        self.dense = tf.keras.layers.Dense(
            units=1,  # 输出张量的维度
            # 激活函数，对应于 f(AW + b) 中的 f ，默认为无激活函数（ a(x) = x
            # 常用的激活函数包括 tf.nn.relu 、 tf.nn.tanh 和 tf.nn.sigmoid
            activation=None,
            use_bias=True,  # 是否加入偏置向量 bias ，即 f(AW + b) 中的 b。默认为 True
            # 权重矩阵 kernel 和偏置向量 bias 两个变量的初始化器。默认为 tf.glorot_uniform_initializer 1 。
            # 设置为 tf.zeros_initializer 表示将两个变量均初始化为全 0
            kernel_initializer=tf.zeros_initializer(),
            bias_initializer=tf.zeros_initializer()
        )

    def call(self, inputs):
        outputs = self.dense(inputs)
        return outputs


# 以下代码结构与前节类似
model = Linear()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
for i in range(100):
    with tf.GradientTape() as tape:
        y_pred = model.call(X)
        loss = tf.reduce_mean(tf.square(y_pred - y))
        print("batch %d: loss %f" % (i, loss.numpy()))
    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
print(model.variables)
