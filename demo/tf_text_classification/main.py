import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

# 下载 IMDB 数据集
imdb = keras.datasets.imdb
# 参数 num_words=10000 保留了训练数据中最常出现的 10,000 个单词。为了保持数据规模的可管理性，低频词将被丢弃
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# 探索数据
# 让我们花一点时间来了解数据格式。该数据集是经过预处理的：每个样本都是一个表示影评中词汇的整数数组。
# 每个标签都是一个值为 0 或 1 的整数值，其中 0 代表消极评论，1 代表积极评论
print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))

# 评论文本被转换为整数值，其中每个整数代表词典中的一个单词。首条评论是这样的：
# print(train_data[0])

# ======================================================================================================================
# 将整数转换回单词
# 了解如何将整数转换回文本对您可能是有帮助的。这里我们将创建一个辅助函数来查询一个包含了整数到字符串映射的字典对象：
# 一个映射单词到整数索引的词典
word_index = imdb.get_word_index()

# 保留第一个索引
word_index = {k: (v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


# 现在我们可以使用 decode_review 函数来显示首条评论的文本
# print(decode_review(train_data[0]))
# ======================================================================================================================


# ======================================================================================================================
# 准备数据
# 影评——即整数数组必须在输入神经网络之前转换为张量。
# 我们可以填充数组来保证输入数据具有相同的长度，然后创建一个大小为 max_length * num_reviews 的整型张量。
# 我们可以使用能够处理此形状数据的嵌入层作为网络中的第一层
# 由于电影评论长度必须相同，我们将使用 pad_sequences 函数来使长度标准化
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)
# print(len(train_data[0]), len(test_data[0]))
print(decode_review(train_data[0]))
# ======================================================================================================================


# ======================================================================================================================
# 构建模型
# 神经网络由堆叠的层来构建，这需要从两个主要方面来进行体系结构决策：
#
# 模型里有多少层？
# 每个层里有多少隐层单元（hidden units）？
# 在此样本中，输入数据包含一个单词索引的数组。要预测的标签为 0 或 1。让我们来为该问题构建一个模型
vocad_size = 10000  # 输入形状是用于电影评论的词汇数目（10,000 词）
model = keras.Sequential()
model.add(keras.layers.Embedding(vocad_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))
model.summary()
# 配置模型来使用优化器和损失函数
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# ======================================================================================================================


# ======================================================================================================================
# 创建一个验证集
# 在训练时，我们想要检查模型在未见过的数据上的准确率（accuracy）。通过从原始训练数据中分离 10,000 个样本来创建一个验证集。
# （为什么现在不使用测试集？我们的目标是只使用训练数据来开发和调整模型，然后只使用一次测试数据来评估准确率（accuracy））
x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]
# ======================================================================================================================


# ======================================================================================================================
# 训练模型
# 以 512 个样本的 mini-batch 大小迭代 40 个 epoch 来训练模型。这是指对 x_train 和 y_train 张量中所有样本的的 40 次迭代。
# 在训练过程中，监测来自验证集的 10,000 个样本上的损失值（loss）和准确率（accuracy）
histroy = model.fit(partial_x_train, partial_y_train,
                    epochs=40, batch_size=512, validation_data=(x_val, y_val),
                    verbose=1)
# ======================================================================================================================


# ======================================================================================================================
# 评估模型
# 我们来看一下模型的性能如何。将返回两个值。损失值（loss）（一个表示误差的数字，值越低越好）与准确率（accuracy）
results = model.evaluate(test_data, test_labels, verbose=2)
print(results)
# ======================================================================================================================


# ======================================================================================================================
# 创建一个准确率（accuracy）和损失值（loss）随时间变化的图表
# model.fit() 返回一个 History 对象，该对象包含一个字典，其中包含训练阶段所发生的一切事件
histroy_dict = histroy.history
print(histroy_dict.keys())
acc = histroy_dict['accuracy']
val_acc = histroy_dict['val_accuracy']
loss = histroy_dict['loss']
val_loss = histroy_dict['val_loss']
epochs = range(1, len(acc) + 1)

# Training and validation loss
fig, ax = plt.subplots(figsize=(5, 2.7), layout='constrained')
ax.plot(epochs, loss, 'bo', label='Training loss')  # “bo”代表 "蓝点"
ax.plot(epochs, val_loss, 'b', label='Validation loss')  # b代表“蓝色实线”
ax.set_title('Training and validation loss')
ax.set_xlabel('Epochs')
ax.set_ylabel('Loss')
ax.legend()
fig.show()

# Training and validation accuracy
fig2, ax2 = plt.subplots(figsize=(5, 2.7), layout='constrained')
ax2.plot(epochs, acc, 'bo', label='Training acc')  # “bo”代表 "蓝点"
ax2.plot(epochs, val_acc, 'b', label='Validation acc')  # b代表“蓝色实线”
ax2.set_title('Training and validation loss')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')
ax2.legend()
fig2.show()
# ======================================================================================================================
