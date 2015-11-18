#!/usr/bin/python
# -*- coding:utf-8 -*-
#########################################################################
# File Name: a.py
# Author: Bruce Zhang
# mail: zhangxb.sysu@gmail.com
# Created Time: 2015年11月16日 星期一 18时10分50秒
#########################################################################

import cv2
import numpy as np
import cnn
import cPickle

'''
读取merge下的所有textline，并根据region.txt中的位置信息，
逐个识别单字，给出识别结果
'''

print('loading model...')
model = cnn.build_model()
model.load_weights('model.h5')
decoder = cPickle.load(open('./decoder.pkl', 'rb'))
print('loading model finished')

def normalize(img):
    '''
    将图片标准化
        1. 加1/4的margin
        2. resize到(48, 48)
    '''
    h, w = img.shape
    size = max(w, h)
    size = size + size / 4
    normal = 255 * np.ones((size,size), np.uint8)
    normal[(size - h) / 2 : (size + h) / 2, (size - w) / 2 : (size + w) / 2] = img
    normal = cv2.resize(normal, (48, 48))
    return normal


def predict(normal):
    '''
    调用分类器，给出识别结果
    '''
    x = normal.reshape(1, 1, 48, 48)
    r = model.predict(x)
    index = np.argmax(r)
    print(decoder[index] + '\t-->\t'  + str(r.max()))


def recognize(filename, box):
    '''
    识别整个textline
    '''
    print "reconizing ", filename
    img = cv2.imread('./textLine/' + filename, 0)
    for b in box:
        if b[4] == '1':
            # 若是汉字，则提取，让分类器识别
            word = img[int(b[1]):int(b[3]), int(b[0]):int(b[2])]
            normal = normalize(word)
            predict(normal)
        else:
            # 不是汉字，就直接给出类型就可以了
            print b[4], ' '


if __name__ == '__main__':
    filename = ''
    for line in open('region.txt', 'r'):
        line = line.strip().split(' ')
        if len(line) == 1:
            if line[0] != '':
                # textline的开始
                filename = line[0] + '.png'
                box = []
            else:
                # textline结束，开始识别这张图片
                recognize(filename, box)
        else:
            if len(line) == 5:
                # textline的其中一个字
                box.append(line)
    print 'recognization finished!'
