import numpy as np
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation, Dropout
from keras.utils import np_utils

import matplotlib.pyplot as plt
import matplotlib.image as processimage

#Load mnist RAW dataset 拉取原始数据
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

#Prepare 准备数据
#reshape
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784) 
#-set type into float32 设置成浮点型
X_train = X_train.astype('float32') #as type into xx.
X_test = X_test.astype('float32')
X_train = X_train/255 # 图片颜色有256种, 归一化
X_test /= 255

#Prepare basic setups for network
batch_sizes = 1024 #每次注入多少数据
nb_class = 10 #训练十个类别
nb_epochs = 20 # 训练多少次

#Class vectors [0,0,0,0,0,0,0,1,0,0] 判断是7  标签向量
Y_test = np_utils.to_categorical(Y_test, nb_class) # Label
Y_train = np_utils.to_categorical(Y_train, nb_class)


#设置网络结构
model = Sequential()

#1st layer
model.add(Dense(512,input_shape=(784,))) # 输入维度 input_dim = 784, Dense=512 是输出
model.add(Activation('relu'))
model.add(Dropout(0.2)) # overfit

# 2nd layer
model.add(Dense(256))  # 在同一个网络中,不用指定输入,因为上面已经有Dense=512了,Dense=256是输出
model.add(Activation('relu'))
model.add(Dropout(0.2))

# 3rd layer
model.add(Dense(10))
model.add(Activation('softmax')) # softmax是对十个进行分类

#编译 compile 用上面函数来处理这个网络
model.compile(
        loss = 'categorical_crossentropy', # 损失函数
        optimizer = 'rmsprop', # 优化器,adam, SGD
        metrics = ['accuracy'],
        )

# 启动网络训练
Trainning = model.fit(
        X_train, Y_train,
        batch_size = batch_sizes,
        epochs = nb_epochs,
        validation_data = (X_test, Y_test)
        ) #除此之外, 还有多个参数.如verbos=2,可以设置训练过程中怎么显示,value=0~4

#以上可以运行
Trainning.history # 查看训练历史参数变化过程
Trainning.params # 查看本次训练参数

#拉取test里面的图-----看测试集或训练集图和标签是否一致
testrun = X_test[9999].reshape(1,784)
testlabel = Y_test[9999]
print('lable:', testlabel)
print(testrun.shape)
plt.imshow(testrun.reshape([28,28]))

# 判定输出结果 放入一张图,判断预测是否一致
pred = model.predict(testrun)
print(testrun)
print('label:', testlabel)
print('预测结果', pred)  #pred是一个向量,需要转换, 用一个函数读取向量
print([final.argmax() for final in pred])

#用自己的图预测一下
target_img = processimage.imread(path)
plt.imshow(target_img)
target_img = target_img.reshape(1, 784)
target_img = np.array(target_img)

target_img = target_img.astype('float32')
target_img /= 255

mypred = model.predict(target_img)

