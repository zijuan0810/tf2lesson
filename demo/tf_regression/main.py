import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)

# ============================================================================================================
# 获取数据
# 首先下载数据集。
dataset_path = keras.utils.get_file("auto-mpg.data",
                                    "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
print(dataset_path)
# 使用 pandas 导入数据集
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv(dataset_path, names=column_names,
                          na_values="?", comment='\t',
                          sep=" ", skipinitialspace=True)
dataset = raw_dataset.copy()
print(dataset.tail())

# ============================================================================================================
# 数据清洗
# 数据集中包括一些未知值
print(dataset.isna().sum())
# 为了保证这个初始示例的简单性，删除这些行
dataset = dataset.dropna()
# "Origin" 列实际上代表分类，而不仅仅是一个数字。所以把它转换为独热码 （one-hot）
origin = dataset.pop('Origin')
dataset['USA'] = (origin == 1) * 1.0
dataset['Europe'] = (origin == 2) * 1.0
dataset['Japan'] = (origin == 3) * 1.0
print(dataset.tail())

# ============================================================================================================
# 拆分训练数据集和测试数据集
# 现在需要将数据集拆分为一个训练数据集和一个测试数据集。
# 我们最后将使用测试数据集对模型进行评估
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# ============================================================================================================
# 数据检查
# 快速查看训练集中几对列的联合分布
sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")
plt.show()
# 也可以查看总体的数据统计
train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()
print(train_stats)

# ============================================================================================================
# 从标签中分离特征
# 将特征值从目标值或者"标签"中分离。 这个标签是你使用训练模型进行预测的值
train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')


# ============================================================================================================
# 数据规范化
# 再次审视下上面的 train_stats 部分，并注意每个特征的范围有什么不同。
# 使用不同的尺度和范围对特征归一化是好的实践。尽管模型可能 在没有特征归一化的情况下收敛，它会使得模型训练更加复杂，并会造成生成的模型依赖输入所使用的单位选择。
# 注意：尽管我们仅仅从训练集中有意生成这些统计数据，但是这些统计信息也会用于归一化的测试数据集。我们需要这样做，将测试数据集放入到与已经训练过的模型相同的分布中。
def norm(x):
    return (x - train_stats['mean']) / train_stats['std']


normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)


# ============================================================================================================
# 模型
# 构建模型
# 让我们来构建我们自己的模型。这里，我们将会使用一个“顺序”模型，其中包含两个紧密相连的隐藏层，以及返回单个、连续值得输出层。
# 模型的构建步骤包含于一个名叫 'build_model' 的函数中，稍后我们将会创建第二个模型。 两个密集连接的隐藏层
def build_model():
    m = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    m.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    return m


model = build_model()

# ============================================================================================================
# 检查模型
# 使用 .summary 方法来打印该模型的简单描述
model.summary()
