import numpy as np
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense, Input
import matplotlib.pyplot as plt
from PIL import Image


# 加载数据集 x (60000, 28, 28) y(10000, )
# 自编码不需要标签
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# 数据归一化处理 转 浮点
X_train = X_train.astype('float32')/255.0
X_test = X_test.astype('float32')/255.0

# reshape 数据形状 适用于dense层input 需要
X_train = X_train.reshape(-1, 784)
X_test = X_test.reshape(-1, 784)

# 定义encoding的终极维度
encoding_dim = 2 # 因为输出的时候需要一个坐标 坐标只有两个值

#定义输入层Input 可以接收的数据shape, 类似tensorflow的 placeholder
input_img = Input(shape=(784,))

# 定义编码层 这里是把 数据从大维度降低到小维度  如 28*28 或者 784   -->降低到2个维度
# 特别注意keras 这次使用API函数模式构建网络层 (区别于Sequantial)

# ** 第一层编码 **  输入784, 输出128 
encoded = Dense(units=128, activation='relu')(input_img) # 数据从input_img 来
# ** 第二层编码 **
encoded = Dense(units=64, activation='relu',)(encoded)
# ** 第三层编码 **
encoded = Dense(units=32, activation='relu',)(encoded)
# ** 第四层编码 **
encoded_output = Dense(units=encoding_dim)(encoded)

# 这里可以输出一个结果

# 定义解码层
# ** 第一层解码**
decoded = Dense(units=32, activation='relu')(encoded_output)
# ** 第二层解码**
decoded = Dense(units=64, activation='relu')(decoded)
# ** 第三层解码**
decoded = Dense(units=128, activation='relu')(decoded)
# ** 第四层解码**
decoded = Dense(units=784, activation='tanh')(decoded)

# 构建自动编码模型结构
autoencoder = Model(inputs=input_img, output=decoded)

# 构建编码模型结构
encoder = Model(inputs=input_img, output= encoded_output)

# 编译模型
autoencoder.compile(optimizer='adam', loss='mse')

# 训练
autoencoder.fit(x=X_train, y= X_train, epochs=1000, batch_size=512, shuffle=True, ) # 注意  无标签区别, 编码解码后与 原图对比 无监督学习 .s

# 打印结果
encoded_imgs = encoder.predict(X_test)
plt.scatter(x=encoded_imgs[:,0], y=encoded_imgs[:,1], c=Y_test, s=3) # c是label, s是精度
plt.show()

# 打印三个图对比

decoded_img = autoencoder.predict(x_test[1].reshape(1, 784))
encoded_img = encoder.predict(x_test[1].reshape(1, 784))

plt.figure(1)
plt.imshow(decoded_img[0].reshape(28, 28))
plt.figure(2)
plt.imshow(encoded_img[0].reshape(2, 2))
plt.figure(3)
plt.imshow(x_test[1].reshape(28, 28))
plt.show()

# 额外的

autoencoder.save('xxx.h5')

ex_img1 = Image.open('7-1.jpg')
ex_img2 = Image.open('7-2.jpg')

ex_img1 = np.array(ex_img1)
ex_img2 = np.array(ex_img2)

encoded_img1 = encoder.predict(ex_img1.reshape(1,784))
encoded_img2 = encoder.predict(ex_img2.reshape(1,784))

print(encoded_img1) # 这是两个坐标
print(encoded_img2) # 这是两个坐标


