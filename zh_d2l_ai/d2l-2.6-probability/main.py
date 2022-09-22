# %matplotlib inline
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from d2l import tensorflow as d2l


fair_probs = tf.ones(6) / 6
# print(tfp.distributions.Multinomial(1000, fair_probs).sample() / 1000)

# 让我们进行500组实验，每组抽取10个样本。
counts = tfp.distributions.Multinomial(10, fair_probs).sample(500)
cun_counts = tf.cumsum(counts, axis=0)
estimates = cun_counts / tf.reduce_sum(cun_counts, axis=1, keepdims=True)

d2l.set_figsize((6, 4.5))
for i in range(6):
    d2l.plt.plot(estimates[:, i].numpy(), label=("P(die=" + str(i + 1) + ")"))
d2l.plt.axhline(y=0.167, color='black', linestyle='dashed')
d2l.plt.gca().set_xlabel('Groups of experiments')
d2l.plt.gca().set_ylabel('Estimated probability')
d2l.plt.legend()
d2l.plt.show()