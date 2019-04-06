import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation, Dropout
from keras.utils import np_utils
import keras
from keras.optimizers import Adam

'''
第一步 准备数据
'''
# matlab文件名 准备数据
file_name = u'G:/GANCode/CSWU/12k drive end vps/trainset/D/D_dataset.mat'
original_data = sio.loadmat(file_name)
X_train = original_data['x_train']
Y_train = original_data['y_train']
X_test = original_data['x_test']
Y_test = original_data['y_test']

X_train = X_train.astype('float32')  # astype SET AS TYPE INTO
X_test = X_test.astype('float32')

# X_train = (X_train+1)/2
# X_test = (X_test+1)/2

'''
第二步 准备神经网络基本参数
'''
batch_sizes = 1024
nb_class = 10
nb_epochs = 100

# 写一个LossHistory类，保存loss和acc
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.savefig("acc_loss_history.png")
        plt.show()

'''
第三步 设置标签 one-hot
'''
Y_test = np_utils.to_categorical(Y_test, nb_class)  # Label
Y_train = np_utils.to_categorical(Y_train, nb_class)

'''
第四步 设置神经网络
'''
model = Sequential()  # 顺序搭建层
# 1st layer
model.add(Dense(512, input_shape=(2048,)))  # Dense是输出给下一层, input_dim = 784 [X*784]
model.add(Activation('relu'))  # tanh
model.add(Dropout(0.2))  # overfitting

# 2nd layer
model.add(Dense(256))  # 256是因为上一层已经输出512了，所以不用标注输入
model.add(Activation('relu'))
model.add(Dropout(0.2))

# 3rd layer
model.add(Dense(128))  # 256是因为上一层已经输出512了，所以不用标注输入
model.add(Activation('tanh'))
model.add(Dropout(0.2))

# last layer
model.add(Dense(10))
model.add(Activation('softmax'))  # 根据10层输出，softmax做分类

'''
第五步 编译compile 并运行
'''

adam = Adam(lr=0.0001)  # Adam实例化

model.compile(
    loss='categorical_crossentropy',
    optimizer='rmsprop',  # rmsprop
    metrics=['accuracy']
)

# 创建一个实例history
history = LossHistory()

# 启动网络训练 Fire up
Training = model.fit(
    X_train,
    Y_train,
    batch_size=batch_sizes,
    shuffle=True,
    epochs=nb_epochs,
    validation_data=(X_test, Y_test),
    callbacks=[history]
)


'''
第六步 检查工作
'''
# 模型评估
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

# 绘制acc-loss曲线
history.loss_plot('epoch')








yangchaoyue = 666
