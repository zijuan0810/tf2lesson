import tensorflow as tf

x = tf.range(4, dtype=tf.float32)
x = tf.Variable(x)

# 把所有计算记录在磁带上
with tf.GradientTape() as tape:
    y = 2 * tf.tensordot(x, x, axes=1)

print(y)
print(tape.gradient(y, x))
print(tf.reduce_sum(x * x))
