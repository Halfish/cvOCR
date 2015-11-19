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
import os
import cnn
import cPickle

'''
读取所有textline，并根据region.txt中的位置信息，
逐个识别单字，给出识别结果
'''

print('loading model...')
model = cnn.build_model_chi()
model.load_weights('model_cv.h5')
decoder = cPickle.load(open('./decoder_cv.pkl', 'rb'))
print('loading model finished')
results = []

def normalize(img, meanHeight, mode):
    '''
    将图片标准化
        1. 加1/4的margin
        2. resize到(48, 48)
    '''
    h, w = img.shape
    size = meanHeight
    if mode == 1:
        size = size + size / 2
    else:
        if mode == 2:
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
    conf = r.max()
    #print(decoder[index] + '\t-->\t'  + str(conf))
    return decoder[index], conf

def recognizeCHI(filename, box, meanHeight):
    '''
    识别整个textline
    '''
    #print "reconizing ", filename
    textline = []
    img = cv2.imread('./textLine/' + filename, 0)
    for b in box:
        if b[4] == '1':
            # 若是汉字，则提取，让分类器识别
            word = img[int(b[1]):int(b[3]), int(b[0]):int(b[2])]
            normal = normalize(word, meanHeight, 2)
            r, c = predict(normal)
            textline.append((r, c))
        else:
            textline.append(('', 0))
    results.append(textline)

def recognizeENG(filename, box, meanHeight, index):
    '''
    识别整个textline
    '''
    count = 0
    #print "reconizing ", filename
    img = cv2.imread('./textLine/' + filename, 0)
    for b in box:
        if b[4] == '0':
            # 若是英文，则提取，让分类器识别
            word = img[int(b[1]):int(b[3]), int(b[0]):int(b[2])]
            normal = normalize(word, meanHeight, 1)
            r, c = predict(normal)
            results[index][count] = (r, c)
        count = count + 1

def recognizeTESS(filename, box, index):
    '''
    识别整个textline
    '''
    count = 0
    img = cv2.imread('./textLine/' + filename, 0)
    for b in box:
        if b[4] == '2':
            #若是类型2，说明要用Tesseract识别
            word = img[int(b[1]):int(b[3]), int(b[0]):int(b[2])]
            cv2.imwrite('word.png', word)
            output = os.popen('tesseract word.png a -l eng 2> /dev/null && cat a.txt')
            r = output.read()
            r = r.strip()
            results[index][count] = (r, 0.999)
        count = count + 1

def run(language):
    filename = ''
    index = 0
    meanHeight = 0
    box = []
    for line in open('./region.txt', 'r'):
        line = line.strip().split(' ')
        if len(line) == 2:
            # textline的开始
            box = []
            filename = line[0] + '.png'
            index = int(line[0])
            meanHeight = int(line[1])
        else:
            if len(line) == 5:
                # textline的其中一个字
                box.append(line)
            if len(line) == 1:
                # textline结束，开始识别这张图片
                if language == 'chi':
                    recognizeCHI(filename, box, meanHeight)
                elif language == 'eng':
                    recognizeENG(filename, box, meanHeight, index)
                elif language == 'tess':
                    recognizeTESS(filename, box, index)

    print 'recognization finished!'

if __name__ == '__main__':
    run('chi')
    print('loading eng model...')
    model = cnn.build_model_eng()
    model.load_weights('model.h5')
    decoder = cPickle.load(open('./decoder.pkl', 'rb'))
    print('loading eng model finished')
    run('eng')
    print('running tesseract...')
    run('tess')
    for t in results:
        print ''.join([r for r, c in t])
        print
