#!/usr/bin/env python
# coding=utf-8
import os
import cv2
import numpy as np
import cPickle
import cnn

'''
程序运行的前提是要有 model.pkl 和 decoder.pkl

predict.py 根据已经训练好的模型，读入图片并开始预测单字
模型存在 model.pkl 中
读取的单字图片在 ../results/ 中
返回的data_x.shape为(7, 1, 48, 48)
'''

# 根据文件夹名读取下面所有的小图片，生成测试集
def load_data(filename):
    data_x = []
    for f in os.listdir(filename):
        fullname = os.path.join(filename, f)
        img = cv2.imread(fullname, 0)
        img = cv2.resize(img, (48, 48))
        name, ext = os.path.splitext(f)
        data_x.append([img, int(name)])
    data_x.sort(lambda x,y:cmp(x[1], y[1]))
    x = []
    for a, b in data_x:
        x.append(a)
    data_x = np.array(x)
    data_x = data_x.reshape(len(data_x), 1, 48, 48)
    return data_x

'''
根据model和data_x，找出概率最大的结果并打印出来
'''
def recognize(model, decoder, data_x):
    print('\nrecognizing...')
    for x in data_x:
        x = x.reshape(1, 1, 48, 48)
        r = model.predict(x)
        index = np.argmax(r)
        print(decoder[index] + '\t-->\t'  + str(r.max()))

if __name__ == '__main__':
    print('loading model...')
    model = cnn.build_model()
    model.load_weights('model.h5')
    decoder = cPickle.load(open('./decoder.pkl', 'rb'))
    print('loading model finished')

    recognize(model, decoder, load_data('./results/0'))
    recognize(model, decoder, load_data('./results/1'))
    recognize(model, decoder, load_data('./results/2'))
    pass
