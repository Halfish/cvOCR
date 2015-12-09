#coding:utf-8

'''
    GPU run command:
        THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cnn.py
    CPU run command:
        python cnn.py
'''
#导入各种用到的模块组件
from __future__ import absolute_import
from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from six.moves import range
import random
import cPickle
import time

def load_data():
    start = time.time()
    print('loading data...')
    #加载数据
    data, label = cPickle.load(open('./data.pkl', 'rb'))
    #打乱数据
    index = [i for i in range(len(data))]
    random.shuffle(index)
    data = data[index]
    label = label[index]
    #label为0~9共10个类别，keras要求格式为binary class matrices,转化一下，直接调用keras提供的这个函数
    label = np_utils.to_categorical(label, 116)

    t = int(time.time() - start)
    print('\tcost ' + str(t / 60) + ' mins, ' + str(t % 60) + ' seconds')
    print(data.shape[0], ' samples')

    return data, label


def build_model_eng():
    #生成一个model
    model = Sequential()

    #第一个卷积层，4个卷积核，每个卷积核大小5*5。
    #border_mode可以是valid或者full，具体看这里说明：http://deeplearning.net/software/theano/library/tensor/nnet/conv.html#theano.tensor.nnet.conv.conv2d
    #激活函数用tanh
    #你还可以在model.add(Activation('tanh'))后加上dropout的技巧: model.add(Dropout(0.5))
    model.add(Convolution2D(8, 3, 3, border_mode='valid', input_shape=(1, 48, 48)))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))

    #第二个卷积层，8个卷积核，每个卷积核大小3*3。
    #激活函数用tanh
    #采用maxpooling，pool_size为(2,2)
    model.add(Convolution2D(16, 3, 3, border_mode='valid'))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #第三个卷积层，16个卷积核，每个卷积核大小3*3
    #激活函数用tanh
    #采用maxpooling，pool_size为(2,2)
    model.add(Convolution2D(32, 3, 3, border_mode='valid'))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #全连接层，先将前一层输出的二维特征图flatten为一维的。
    #全连接有128个神经元节点,初始化方式为normal
    model.add(Flatten())
    model.add(Dense(128, init='normal'))
    model.add(Activation('tanh'))

    #Softmax分类，输出是10类别
    model.add(Dense(116, init='normal'))
    model.add(Activation('softmax'))

    #使用SGD + momentum
    #model.compile里的参数loss就是损失函数(目标函数)
    sgd = SGD(l2=0.0,lr=0.01, decay=5e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd,class_mode="categorical")

    return model

def build_model_chi():
    #生成一个model
    model = Sequential()

    #第一个卷积层，4个卷积核，每个卷积核大小5*5。
    #border_mode可以是valid或者full，具体看这里说明：http://deeplearning.net/software/theano/library/tensor/nnet/conv.html#theano.tensor.nnet.conv.conv2d
    #激活函数用tanh
    #你还可以在model.add(Activation('tanh'))后加上dropout的技巧: model.add(Dropout(0.5))
    model.add(Convolution2D(16, 3, 3, border_mode='valid', input_shape=(1, 48, 48)))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))

    #第二个卷积层，8个卷积核，每个卷积核大小3*3。
    #激活函数用tanh
    #采用maxpooling，pool_size为(2,2)
    model.add(Convolution2D(32, 3, 3, border_mode='valid'))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #第三个卷积层，16个卷积核，每个卷积核大小3*3
    #激活函数用tanh
    #采用maxpooling，pool_size为(2,2)
    model.add(Convolution2D(64, 3, 3, border_mode='valid'))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #全连接层，先将前一层输出的二维特征图flatten为一维的。
    #全连接有128个神经元节点,初始化方式为normal
    model.add(Flatten())
    model.add(Dense(1024, init='normal'))
    model.add(Activation('tanh'))

    #Softmax分类，输出是10类别
    model.add(Dense(3230, init='normal'))
    model.add(Activation('softmax'))

    #使用SGD + momentum
    #model.compile里的参数loss就是损失函数(目标函数)
    sgd = SGD(l2=0.0,lr=0.01, decay=5e-5, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd,class_mode="categorical")

    return model

def training():
    data, label = load_data()
    #调用fit方法，就是一个训练过程. 训练的epoch数设为10，batch_size为100．
    #数据经过随机打乱shuffle=True。verbose=1，show_accuracy=True，训练时每一个epoch都输出accuracy。
    #validation_split=0.2，将20%的数据作为验证集。
    model = build_model_eng()
    model.fit(data, label, batch_size=100, nb_epoch=100,shuffle=True,verbose=1,show_accuracy=True,validation_split=0.2)
    model.save_weights('model.h5', True)

if __name__ == '__main__':
    '''
    运行这个程序就是要重新训练的，只有预测才会需要load model.pkl
    '''
    training()
