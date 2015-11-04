#!/usr/bin/env python
# coding=utf-8

'''
python split.py
将 ./complex/ 文件夹下的验证码，根据色调分离出来
只将像素值排名前6的色调存成图片
    
    PS: 色调，Hue，指的是HSV编码中的H
'''

import cv2
import os.path
import os
import numpy as np

def run(fullname):
    dirname, filename = os.path.split(fullname)
    file, ext = os.path.splitext(filename)
    if os.path.exists('./samples/' + file) == False:
        os.mkdir('./samples/' + file)

    img = cv2.imread(fullname)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    hist = cv2.calcHist([h], [0], None, [180], [0, 180])
    arg = np.argsort(-hist, axis=0)
    arg = arg.reshape(arg.shape[0])
    for i in range(6):
        newImage = 255 * np.ones(h.shape, np.uint8)
        for j in range(newImage.shape[0]):
            for k in range(newImage.shape[1]):
                if (h[j][k] == arg[i]):
                    newImage[j][k] = 0
        name = './samples/%s/%d.png' % (file, i)
        cv2.imwrite(name, newImage)
    pass


if __name__ == '__main__':
    if (os.path.exists('./samples/') == False):
        os.mkdir('samples')
    for f in os.listdir('./complex/'):
        run('./complex/' + f)
    pass
