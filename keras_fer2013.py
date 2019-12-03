"""
fer2013数据集的处理（.csv文件）
"""

import os
import numpy as np
import pandas as pd #处理csv文件
from PIL import Image

#这一段没用，要生成0-6的文件夹，号处理
emotions = {
    '0' : 'anger', #生气
    '1' : 'digust', #厌恶
    '2' : 'fear', #恐惧
    '3' : 'happy', #开心
    '4' : 'sad', #伤心
    '5' : 'surprised', #惊讶
    '6' : 'normal', #中性，字典的最后一个键值对后面“，”可有可无
}

#print(emotions)

#创建文件夹
def createDir(dir):
    if os.path.exists(dir) is False: #检测有没有该文件夹，因为放在for循环里所以必须要检测
        os.makedirs(dir)
    return dir

FER2013 = createDir(r'E:\python_work\Fical expression recognition\FER2013') #NICE!按自己的方式保存路径


def saveImageFromFer2013(file):
    #读取csv文件

    faces_data = pd.read_csv(file) #注意csv文件中第一行标注不算
    #print(len(faces_data)) #35887*3, len = 35887

    #逐行处理！

    for index in range(len(faces_data)):
        emotion_data = faces_data.loc[index][0] #loc加载csv文件的行和列，可以看成矩阵处理，第一列代表标签,1个值
        image_data = faces_data.loc[index][1] #第二列代表像素，2304个值
        usage_data = faces_data.loc[index][2] #第三列表示用途，包括三类training，piblictest，privatetest，1个值

        data_array = list(map(float, image_data.split())) #处理为浮点型数据，.split()表示以空格作为分隔符 map函数（function,对象）
        #print(data_array) #第一行的
        data_array = np.asarray(data_array) #asarray不会占用内存，没有array的copy过程
        image = data_array.reshape(48, 48) #全部处理成48*48像素

        #进行分类，并创建文件名
        if usage_data == 'Training':
            usage_data = 'train'
        elif usage_data == 'PublicTest':
            usage_data = 'val'
        else:
            usage_data = 'test'

        dirName = os.path.join(FER2013, usage_data) #首先对应的三个数据集
        #emotionName = emotions[str(emotion_data)] #每一个数据集中具体对应的表情分类名称

        # 图片要保存的文件夹
        imagePath = os.path.join(dirName, str(emotion_data))

        # 创建“用途文件夹”和“表情”文件夹，这一段是来一个图像如果其对应的文件夹已经建立了那么就跳过该部分
        createDir(dirName)
        createDir(imagePath)

        # 图片文件名
        imageName = os.path.join(imagePath, str(index) + '.jpg')

        #sm.toimage(image).save(imageName) 这个功能已经删除了,采用pillow中的image
        img = Image.fromarray(image)
        img.convert('L').save(imageName) #这里一定要加convert否则会报错，其次采用L模式
        imageCount = index
    print('总共有' + str(imageCount) + '张图片')

if __name__ == '__main__':
    saveImageFromFer2013(r'E:\python_work\Fical expression recognition\origin_data\fer2013\fer2013.csv')
















