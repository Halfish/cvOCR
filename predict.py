#!/usr/bin/env python
# coding=utf-8
import os
import cv2
import numpy as np
import cPickle

# 读取小图片，生成训练集
def load_data(filename):
    data_x = []
    for f in os.listdir(filename):
        fullname = os.path.join(filename, f)
        print fullname
        img = cv2.imread(fullname, 0)
        img = cv2.resize(img, (48, 48))
        data_x.append(img)
    img = cv2.resize(img, (48, 48))
    data_x.append(img)
    data_x = np.array(data_x)
    data_x = data_x.reshape(len(data_x), 1, 48, 48)
    print(data_x.shape)
    return data_x

def recognize(data_x):
    model = cPickle.load(open('./model.pkl', 'rb'))
    for x in data_x:
        x = x.reshape(1, 1, 48, 48)
        index = np.argmax(model.predict(x))
        print(index)

if __name__ == '__main__':
    recognize(load_data('../results/1'))
    pass
