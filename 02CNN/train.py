import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Convolution2D, Activation, MaxPool2D, Flatten, Dense
from keras.optimizers import Adam

nb_class = 10
nb_epoch = 4
batchsize = 128

# Prepare your data mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

#setup data shape  卷积层中要给网络一个矩阵; 01节中的全链接层把照片变成一位数组.
X_train = X_train.reshape(-1, 28, 28, 1) #tensorflow -1 代表 不知道多少照片, 28, 28,是像素, 1是通道. theano通道是在前面(1,28,28)
X_test = X_test.reshape(-1, 28, 28, 1)

X_train = X_train/255.000
X_test = X_test/255.000
# One-hot [0,0,0,0,0,1,0,0,0] = 5 把测试集和训练集label转成onehot

Y_train = np_utils.to_categorical(Y_train, nb_class)
Y_test = np_utils.to_categorical(Y_test, nb_class)

#setup model

model = Sequential()

#1st Conv2D layer 给卷积神经网络 设置第一层卷积层 池化层 
model.add(Convolution2D(
    filters=32, # ??
    kernel_size=[5,5], # ??小筛子
    padding='same', # ?? 5*5过边界的时候
    input_shape=(28,28,1) #输入的图都是这个规格
    ))

model.add(Activation('relu'))

model.add(MaxPool2D(
    pool_size = (2,2), # 池化大小 2*2
    strides = (2,2), # 池化的时候空两个抓两个
    padding='same',
    ))

#2nd Conv2D layer
model.add(Convolution2D(
    filters = 64, # 往上加???为什么
    kernel_size = (5,5),
    padding='same',
    ))

model.add(Activation('relu'))

model.add(MaxPool2D(
    pool_size = (2,2),
    strides=(2,2),
    padding = 'same',
    ))

# 神经网络层数可以一直叠加

# 1st Fully connected Dense 接下来全连接层

model.add(Flatten()) # 打成一维

model.add(Dense(1024)) #输入在卷积层,在model里面的话,不用考虑层数, Dense=1024是输出
model.add(Activation('relu'))

# extra layer
model.add(Dense(256))
model.add(Activation('relu'))  # 多一层relu 是最好的.



# 2nd Fully connected Dense
model.add(Dense(10))
model.add(Activation('softmax'))


# Define Optimizer and setup Param
adam = Adam(lr = 1e-4)  # 把adam实例化

# compile model
model.compile(
        optimizer=adam, 
        loss = 'categorical_crossentropy',
        metrics=['accuracy'],
        )

# Run


model.fit(
        x=X_train, # 原始数据
        y=Y_train, # 原始数据label
        epochs=nb_epoch,
        batch_size=batchsize,
        verbose=1, #显示模式
        validation_data=(X_test, Y_test), # 每一次训练都会把测试集放进去
        )

model.save('model_name.h5')
# evaluation = model.evaluate(X_test, Y_test) 
# print(evaluation)

