#!/usr/bin/env python
# coding=utf-8
import cv2
import numpy as np
import cPickle

'''
读取大图片和文字位置信息，提取出文字小图片到文件夹里
根据不同的字体归类到不同的文件夹中
'''

def extractPatches(filename):
    '''
    从box文件中读取位置信息，把图片和是否valid的信息放到small中
    box 文件格式 （汉字，左下角x坐标，y坐标，右上角x坐标，y坐标，0）
    '''
    [picname, boxname] = filename
    img = cv2.imread(picname, 0)
    with open(boxname, 'r') as f:
        patches = [] 
        for line in f.readlines():
            s = line.split(' ')
            s = map(int, s[1:5])
            y1 = img.shape[0] - s[1]
            y2 = img.shape[0] - s[3]
            [x1, x2] = [s[0], s[2]]
            patch = img[y2:y1, x1:x2]
            patch = cv2.adaptiveThreshold(patch, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            pixes = ((255 - patch) / 255).sum()
            if (pixes > 10):
                patches.append((patch, True, s[0]))
            else:
                patches.append((patch, False, s[0]))
    return patches 

def normalizePatches(patches, saveddir):
    '''
    将每张图片都放到一个大小相同的背景中，然后resize到（48,48）大小
    存到saveddir文件夹下面，并生成data.pkl训练集
    patches 格式
        1. patch 图片 
        2. isValid是否是一个汉字 
        3. ch汉字
    '''
    index = 0
    # calc maxsize
    maxsize = 0
    for p in patches:
        if (p[0].shape[0] > maxsize) :
            maxsize= p[0].shape[0]
        if (p[0].shape[1] > maxsize) :
            maxsize = p[0].shape[1]
    print "maxsize is " + str(maxsize)

    train_set = []
    for p in patches:
        if (p[1] == False):
            continue
        norpic = 255 * np.ones([maxsize, maxsize], np.uint8)
        x1 = (maxsize - p[0].shape[1]) / 2        
        x2 = x1 + p[0].shape[1] 
        y1 = (maxsize - p[0].shape[0]) / 2        
        y2 = y1 + p[0].shape[0] 
        norpic[y1:y2, x1:x2] = p[0]
        final = cv2.resize(norpic, (48, 48))
        cv2.imwrite(saveddir + str(index) + '.png', final)
        index = index + 1
        train_set.append([final, p[2]])
    f = open("./data.pkl", "wb")
    cPickle.dump(train_set, f)
    f.close()

if __name__== '__main__':
    patches = extractPatches(["./pictures/cv.msyh.exp0.tif", "./pictures/cv.msyh.exp0.box"]);
    normalizePatches(patches, "./msyh/")

