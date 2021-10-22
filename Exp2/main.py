from keras.utils import to_categorical
from keras import models, layers, regularizers
from keras.optimizers import RMSprop
from keras.datasets import mnist # 导入数据集
import matplotlib.pyplot as plt

# 加载数据集
(train_images, train_labels), (test_images, test_labels) = mnist.load_data() # 下载数据集
# print(train_images.shape, test_images.shape) # 打印输出形状
# print(train_images[0])
# print(train_labels[0])
# plt.imshow(train_images[0])
# plt.show()

# 将二维数据铺开成一维
# train_images = train_images.reshape((60000, 28*28)).astype('float') # 784，输入层有784个神经元
# test_images = test_images.reshape((10000, 28*28)).astype('float')
# # 标签值编码
# train_labels = to_categorical(train_labels)
# test_labels = to_categorical(test_labels)
# print(train_labels[0])

# # 搭建神经网络
# # 输入层28*28个神经元，隐藏层15个神经元，输出层10个神经元（0~9）
# network = models.Sequential() # 序列式模型
# network.add(layers.Dense(units=128, activation='relu', input_shape=(28*28, ),
#                          kernel_regularizer=regularizers.l1(0.0001))) # 隐藏层
# # 正则化消除过拟合问题
# # Dense：全连接层
# # units：15个神经元
# # 激活函数：ReLu（sigmoid和tanh会产生梯度弥散现象，自变量很大时，图像很平缓，梯度下降十分缓慢）
# network.add(layers.Dropout(0.01)) # 0.01的概率使神经元丧失功能
# network.add(layers.Dense(units=32, activation='relu',
#                          kernel_regularizer=regularizers.l1(0.0001))) # 第二层隐藏层
# network.add(layers.Dropout(0.01)) # 0.01的概率使神经元丧失功能
# network.add(layers.Dense(units=10, activation='softmax')) # 输出层
# # 激活函数选择softmax，输出为概率值

# 尝试使用卷积神经网络
def Lenet():
    network = models.Sequential()
    network.add(layers.Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
    network.add(layers.AveragePooling2D((2, 2)))
    network.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
    network.add(layers.AveragePooling2D((2, 2)))
    network.add(layers.Conv2D(filters=120, kernel_size=(3, 3), activation='relu'))
    network.add(layers.Flatten())
    network.add(layers.Dense(84, activation='relu'))
    network.add(layers.Dense(10, activation='softmax'))
    return network

network = Lenet()

# 训练神经网络
# 编译：确定优化器（学习率/梯度下降步长）和损失函数等
network.compile(optimizer=RMSprop(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

train_images = train_images.reshape((60000, 28, 28, 1)).astype('float') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# epochs表示训练多少个回合，batch_size表示每次训练给多大的数据
network.fit(train_images, train_labels, epochs=10, batch_size=128, verbose=2)
# 经过20轮训练，训练集准确率达到80%
# print(network.summary())

# 用模型进行预测
# y_pre = network.predict(test_images[:5])
# print(y_pre, test_labels[:5])
test_loss, test_accuracy = network.evaluate(test_images, test_labels)
print("test_loss: ", test_loss, "          test_accuracy: ", test_accuracy)