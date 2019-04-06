# 原文：https://blog.csdn.net/gjq246/article/details/75118751 
#(1) 生成器generator_model
def generator_model():

	model = Sequential()

	model.add(Dense(input_dim=100, output_dim=1024))

	model.add(Activation('relu'))

	model.add(Dense(128*7*7)) #对上一层的神经元进行全部连接，实现特征的非线性组合，对特征进行综合考虑

	model.add(BatchNormalization())

	model.add(Activation('relu'))

	model.add(Reshape((7, 7, 128), input_shape=(128*7*7,)))

	model.add(UpSampling2D(size=(2, 2))) #第一个2表示行重复2次，第二个2表示列重复2次，可以用于复原图像等

	model.add(Convolution2D(64, 5, 5, padding='same')) #卷积提取特征，卷积核64个，核大小为5*5，就是用64个5*5大小的卷积核去提取不同的特征，步长（就是卷积核每次移动的大小）默认为1，不为1时会改变输出矩阵的大小，例如步长strides为2则会缩小一半

	model.add(Activation('relu'))

	model.add(UpSampling2D(size=(2, 2)))

	model.add(Convolution2D(1, 5, 5, padding='same'))

	model.add(Activation('tanh'))

	return model

#(2)判别器discriminator

def discriminator_model():

	model = Sequential()

	model.add(Convolution2D(

	64, 5, 5,

	border_mode='same',

	input_shape=(28, 28, 1))) #输入一张图片，，卷积核64个，核大小为5*5，就是用64个5*5大小的卷积核去提取不同的特征，步长（就是卷积核每次移动的大小）默认为1

	model.add(Activation('tanh'))

	model.add(MaxPooling2D(pool_size=(2, 2))) #空间池化，定义了空间上的邻域（2x2的窗）并且从纠正特征映射中取出窗里最大的元素，逐步减少输入特征的空间尺寸

	model.add(Convolution2D(128, 5, 5))

	model.add(Activation('tanh'))

	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Flatten()) #把多维的输入一维化，常用在从卷积层到全连接层的过渡

	model.add(Dense(1024)) #对上一层的神经元进行全部连接，实现特征的非线性组合，对特征进行综合考虑

	model.add(Activation('tanh'))

	model.add(Dense(1))

	model.add(Activation('sigmoid'))

	return model

#(3)对抗模型

def generator_containing_discriminator(generator, discriminator):

	model = Sequential()

	model.add(generator)

	discriminator.trainable = False

	model.add(discriminator)

	return model

#（4）训练
def train(BATCH_SIZE):
	(X_train, y_train), (X_test, y_test) = mnist.load_data()
	X_train = (X_train.astype(np.float32) - 127.5)/127.5  #范围调整为[-1,1]
	X_train = X_train.reshape((X_train.shape[0], 1) + X_train.shape[1:]) #需要上机调试看看\
	discriminator = discriminator_model()
	generator = generator_model()
	discriminator_on_generator = \
	generator_containing_discriminator(generator, discriminator)

    d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)

    g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)

    generator.compile(loss='binary_crossentropy', optimizer="SGD") #生成器，损失函数为（亦称作对数损失，logloss），SGD-随机梯度下降算法

    discriminator_on_generator.compile(

        loss='binary_crossentropy', optimizer=g_optim) #对抗模型

    discriminator.trainable = True

    discriminator.compile(loss='binary_crossentropy', optimizer=d_optim) #判别器

    noise = np.zeros((BATCH_SIZE, 100)) #每次BATCH_SIZE个样本

    for epoch in range(100): #所有样本重复100次

        print("Epoch is", epoch)

        print("Number of batches", int(X_train.shape[0]/BATCH_SIZE)) #总共有多少批

        for index in range(int(X_train.shape[0]/BATCH_SIZE)): #所有样本按照批大小训练一次

            for i in range(BATCH_SIZE):

                noise[i, :] = np.random.uniform(-1, 1, 100) #产生BATCH_SIZE个噪点

            image_batch = X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE] #获取对应的BATCH_SIZE个训练样本

            image_batch = image_batch.transpose((0,2,3,1)) #？？？改变下标

            generated_images = generator.predict(noise, verbose=0) #产生BATCH_SIZE个图片

            if index % 20 == 0:

                generated_images_tosave = generated_images.transpose((0,3,1,2)) #？？？改变下标

                image = combine_images(generated_images_tosave)#？？？

                image = image*127.5+127.5 #重新变回图像数据

                Image.fromarray(image.astype(np.uint8)).save(

                    str(epoch)+"_"+str(index)+".png") #保存一部分图片，下标能被20整除的

            print(image_batch.shape, generated_images.shape)

            X = np.concatenate((image_batch, generated_images)) #训练和产生的图片合并在一起

            y = [1] * BATCH_SIZE + [0] * BATCH_SIZE #y为输出，？？？一部分为1，另一部分为0

            d_loss = discriminator.train_on_batch(X, y)

            print("batch %d d_loss : %f" % (index, d_loss))

            for i in range(BATCH_SIZE):

                noise[i, :] = np.random.uniform(-1, 1, 100)

            discriminator.trainable = False

            g_loss = discriminator_on_generator.train_on_batch(

                noise, [1] * BATCH_SIZE) #对抗模型，只是把生成器和判别器联合在一起

            discriminator.trainable = True

            print("batch %d g_loss : %f" % (index, g_loss)) #损失函数，应该是检验生成的图片是否能够判别出来？直到无法判别真假，损失函数与输出（0-1之间的值）之间的关系？？？

            if index % 10 == 9:

                generator.save_weights('generator', True) #生成器模型

                discriminator.save_weights('discriminator', True) #判别器模型，而对抗模型是他们的联合不需要保存