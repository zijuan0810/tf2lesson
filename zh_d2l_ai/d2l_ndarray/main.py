import tensorflow as tf

a = tf.reshape(tf.range(12, dtype=tf.float32), (3, 4))
print(a)

v = tf.Variable(a, dtype=tf.float32)
# print(v[0, 0].assign(9))
# X_var[0:2, :].assign(tf.ones(X_var[0:2,:].shape, dtype = tf.float32) * 12)
print(tf.ones(v[0:2, :].shape, dtype=tf.float32) * 12)
print(v[0:2, :].assign(tf.ones(v[0:2, :].shape, dtype=tf.float32) * 12))
print(id(v))
