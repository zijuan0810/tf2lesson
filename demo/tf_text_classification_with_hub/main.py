import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

# =====================================================================================================================
# 打印版本信息
print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")

# tfds.take(k).cache().repeat()

# =====================================================================================================================
# 下载 IMDB 数据集
# IMDB数据集可以在 Tensorflow 数据集处获取。以下代码将 IMDB 数据集下载至您的机器（或 colab 运行时环境）中：
#
# Split the training set into 60% and 40% to end up with 15,000 examples
# for training, 10,000 examples for validation and 25,000 examples for testing.
train_data, validation_data, test_data = tfds.load(
    name="imdb_reviews",
    split=('train[:60%]', 'train[60%:]', 'test'),
    as_supervised=True)

# =====================================================================================================================
# 探索数据
# 我们花一点时间来了解数据的格式。每个样本都是一个代表电影评论的句子和一个相应的标签。句子未经过任何预处理。
# 标签是一个整数值（0 或 1），其中 0 表示负面评价，而 1 表示正面评价。
# 我们来打印下前十个样本。
train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))
print(train_labels_batch)

# =====================================================================================================================
# 构建模型
# 在本示例中，您使用来自 TensorFlow Hub 的 预训练文本嵌入向量模型，名称为 google/nnlm-en-dim50/2。
embedding = "https://tfhub.dev/google/nnlm-en-dim50/2"
embed = hub.load(embedding)
hub_layer = hub.KerasLayer(embedding, input_shape=[], dtype=tf.string, trainable=True)
# 现在让我们构建完整模型
model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1))
model.summary()  # Prints a string summary of the network.

# =====================================================================================================================
# 损失函数与优化器
# 一个模型需要一个损失函数和一个优化器来训练。由于这是一个二元分类问题，并且模型输出 logit（具有线性激活的单一单元层），
# 因此，我们将使用 binary_crossentropy 损失函数。
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

# =====================================================================================================================
# 训练模型
# 使用包含 512 个样本的 mini-batch 对模型进行 10 个周期的训练，也就是在 x_train 和 y_train 张量中对所有样本进行 10 次迭代。
# 在训练时，监测模型在验证集的 10,000 个样本上的损失和准确率
history = model.fit(train_data.shuffle(10000).batch(512),
                    epochs=10,
                    validation_data=validation_data.batch(512),
                    verbose=1)

# =====================================================================================================================
# 评估模型
# 我们来看下模型的表现如何。将返回两个值。损失值（loss）（一个表示误差的数字，值越低越好）与准确率（accuracy）
results = model.evaluate(test_data.batch(512), verbose=1)
for name, value in zip(model.metrics_names, results):
    print("%s: %.3f" % (name, value))
