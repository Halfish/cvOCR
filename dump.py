#!/usr/bin/env python
# coding=utf-8
import cv2
import numpy as np
import cPickle

'''
读取大图片和文字位置信息，提取出文字小图片到文件夹里
根据不同的字体归类到不同的文件夹中
'''

train_set_x = []
train_set_y = []
label = []

def extractPatches(filename):
    '''
    从box文件中读取位置信息，把图片和是否valid的信息放到small中
    box 文件格式 （汉字，左下角x坐标，y坐标，右上角x坐标，y坐标，0）
    '''
    [picname, boxname] = filename
    img = cv2.imread(picname, 0)
    with open(boxname, 'r') as f:
        patches = [] 
        lines = f.readlines()
        print '\nread ' + str(len(lines)) + ' lines'
        for line in lines:
            s = line.split(' ')
            ch = s[0]
            s = map(int, s[1:5])
            y1 = img.shape[0] - s[1]
            y2 = img.shape[0] - s[3]
            [x1, x2] = [s[0], s[2]]
            patch = img[y2:y1, x1:x2]
            #patch = cv2.adaptiveThreshold(patch, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            if (ch == '\t'):
                patches.append((patch, False, ch))
            else:
                patches.append((patch, True, ch))
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
    if (maxsize > 50):
        print "maxsize is " + str(maxsize)

    count = 0
    for p in patches:
        if (p[1] == True):
            count = count + 1
    print 'only ' + str(count) + ' valid'

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

        train_set_x.append(final)
        train_set_y.append(p[2])
        label.append(index)

        index = index + 1

if __name__== '__main__':
    fonts = ['msyh', 'simsun', 'simkai']
    for f in fonts:
        for i in ['0', '1']:
            for s in ['48']:
                picname = './pictures/cv.' + f + '.exp' + i + '.size' + s + '.tif'
                boxname = './pictures/cv.' + f + '.exp' + i + '.size' + s + '.box'
                patches = extractPatches([picname, boxname]);
                normalizePatches(patches, './samples/' + f + i + '/' + s + '/')
    x = np.array(train_set_x)
    x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
    y = np.array(train_set_y) 
    label = np.array(label)

    f = open("./data.pkl", "wb")
    train_set = [x, label, y]
    print 'dumping data...'
    cPickle.dump(train_set, f)
    f.close()

