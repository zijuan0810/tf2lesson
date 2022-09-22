import tensorflow as tf

# x, y = tf.constant(3.0), tf.constant(2.0)
# print(x + y, x * y, x / y, x // y, x ** y)

a = tf.constant(tf.range(12, dtype=tf.float32), shape=(3,4))
print(a)
# print(tf.reduce_sum(a, axis=0))
# print(tf.reduce_sum(a, axis=1))
# print(tf.reduce_mean(a), tf.reduce_sum(a) / tf.size(a).numpy())
# print(tf.reduce_mean(a, axis=0, keepdims=True))
# print(tf.reduce_mean(a, axis=1, keepdims=True))
print(tf.cumsum(a, axis=0))
