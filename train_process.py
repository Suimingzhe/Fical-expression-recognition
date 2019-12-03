"""
模型训练
"""

import numpy as np
from keras import layers
from keras import models
from keras import optimizers
import time
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt


#==================================================================================================================
batch_siz = 128 #表示train_generator等生成器中的batch_size
img_size = 48 #处理后的图片48*48
num_classes = 7 #表情分类，7类
nb_epoch = 60 #fit中的循环次数
train_dir = r'E:\python_work\Fical expression recognition\FER2013\train'
val_dir = r'E:\python_work\Fical expression recognition\FER2013\val'
test_dir = r'E:\python_work\Fical expression recognition\FER2013\test'
save_path = r'E:\python_work\Fical expression recognition'


#==================================================================================================================
#整体框架如下
class Model:
    """创建一个类，各个处理模块"""

    def __init__(self):
        #只有一个属性，这样None定义在调用之后必须赋值
        self.model = None

    def build_model(self):
        #建立模型
        self.model = models.Sequential() #这里只需要一个即可，后面的def也不需要
        self.model.add(layers.Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu', input_shape=(img_size, img_size, 1)))
        self.model.add(layers.Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='relu'))
        self.model.add(layers.MaxPooling2D(pool_size=(2,2), strides=2))

        self.model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu'))
        self.model.add(layers.MaxPooling2D(pool_size=(2,2), strides=2))

        self.model.add(layers.Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu'))
        self.model.add(layers.MaxPooling2D(pool_size=(2,2), strides=2))

        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(units=2048, activation='relu'))
        self.model.add(layers.Dropout(0.5))
        self.model.add(layers.Dense(units=1024, activation='relu'))
        self.model.add(layers.Dropout(0.5))
        self.model.add(layers.Dense(units=num_classes, activation='softmax'))

        self.model.summary()

    def train_model(self): #参考之前的顺序，model.add之后，即compile，然后才是generator及fit
        """开始数据的预处理过程"""
        sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True) #SGD随机梯度下降，lr学习率，momentum动量参数，decay每次更新后的学习率衰减值
        self.model.compile(loss='categorical_crossentropy', #多元分类损失函数
                           optimizer=sgd,
                           metrics=['accuracy'])

        #利用生成器进行图像预处理，数据增强
        train_datagen = ImageDataGenerator(rescale=1./255, #像素归一化
                                           rotation_range=40, #角度值0-40
                                           width_shift_range=0.2,
                                           height_shift_range=0.2,
                                           shear_range=0.2, #随机错切变换的角度
                                           zoom_range=0.2, #随机缩放的角度
                                           horizontal_flip=True)#随机将一半的图像水平翻转

        #注意，验证集合测试集数据不能增强！可以进行归一化数据
        val_datagen = ImageDataGenerator(rescale=1./255)
        test_datagen = ImageDataGenerator(rescale = 1./255)



        #flow_from_directory（dictory）方法,该文件夹都要包含一个子文件夹.子文件夹中任何JPG、PNG、BNP、PPM的图片都会被生成器使用
        #以文件分类名划分label
        train_generator = train_datagen.flow_from_directory(train_dir,
                                                            target_size=(img_size,img_size), #这一部不需要，因为train_dir里面已经是resize过的了
                                                            color_mode='grayscale', #颜色模式，graycsale（默认rgb）
                                                            batch_size=batch_siz, #batch_size = 128
                                                            class_mode='categorical') #返回的标签形式,输出打印：显示有7类

        val_generator = val_datagen.flow_from_directory(val_dir,
                                                        target_size=(img_size,img_size),
                                                        color_mode='grayscale',
                                                        batch_size=batch_siz,
                                                        class_mode='categorical')

        test_generator = test_datagen.flow_from_directory(test_dir,
                                                          target_size=(img_size,img_size),
                                                          color_mode='grayscale',
                                                          batch_size=batch_siz,
                                                          class_mode='categorical')

        #监测功能(好像用到)
        early_stopping = EarlyStopping(monitor='loss', patience=3) #检测loss，当发现loss在当前epoch没有变化时候，等待3个循环后停下来

        # 开始训练！
        tic = time.time()
        history = self.model.fit_generator(train_generator, #训练生成器，一次128张图片，一共是28709张图片
                                           steps_per_epoch=450, #28709/128 =224.23，但是使用了数据增强，所以训练集远远大于28709
                                           epochs=nb_epoch,
                                           verbose=2,
                                           validation_data=val_generator,
                                           validation_steps=30, #3589/128=28.04，同样验证集不要大，因为验证集没有数据增强
                                           #callbacks=[early_stopping]
                                           )
        toc = time.time()
        print("Time：" + (str(float((toc - tic)/3600))) + ' hours')

        #从生成器上预测结果
        predictions = self.model.predict_generator(test_generator, #测试集生成器
                                                   steps=30) #3589/128=28.04，与验证集一样即可
        print(predictions)
        print(predictions.shape)
        test_generator.reset()

        test_loss, test_acc = self.model.evaluate_generator(test_generator,
                                                   steps=30)
        print("test_acc :" + str(test_acc))

        #保存训练日志
        with open(r'E:\python_work\Fical expression recognition\model_fit_log', 'w') as f:
            f.write(str(history.history))
        with open(r'E:\python_work\Fical expression recognition\model_predict_log', 'w') as f:
            f.write(str(predictions))

        #画图
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(acc) + 1)
        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()  # 添加plot中的label

        plt.figure()  # 与MATLAB中figure(1),即创建了一个新的画图窗口
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.show()

    def save_model(self):
        model_json = self.model.to_json()
        with open(r'E:\python_work\Fical expression recognition\model_json.json', "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights(r'E:\python_work\Fical expression recognition\model_weight.h5')
        self.model.save(r'E:\python_work\Fical expression recognition\model.h5')


if __name__=='__main__':
    model=Model() #先加载类
    model.build_model() #加载build_model属性
    print('model built')
    model.train_model()
    print('model trained')
    model.save_model()
    print('model saved')

