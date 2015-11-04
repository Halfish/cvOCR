#!/usr/bin/env python
# coding=utf-8
import cv2
import numpy as np
import cPickle
import os
import time

'''
1.创建文件夹 eg ./samples/simsun44dilation51/
2.从./bigpic/下读取tif和box文件，每张图片分出3000+个汉字，放到文件夹中
3.归一化，打包成data.pkl
'''

train_set_x = []
train_set_y = []
label = []

def extractPatches(filename):
    '''
    从box文件中读取位置信息，把单字的图片存到patches中
        box 文件格式 
        （汉字，左下角x坐标，左下角y坐标，右上角x坐标，右上角y坐标，0）

    每行末尾都有一个汉字是空的，用'\t'标示
    '''
    [picname, boxname] = filename
    img = cv2.imread(picname, 0)
    with open(boxname, 'r') as f:
        patches = [] 
        lines = f.readlines()
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
    归一化，将每张图片都放到一个大小相同的背景中，然后resize到（48,48）大小
    存到saveddir文件夹下面，并生成data.pkl训练集
    patches 格式
        1. patch 图片 
        2. isValid是否是一个汉字 
        3. ch汉字
    '''
    if (os.path.exists(saveddir) == False):
        os.mkdir(saveddir)

    index = 0
    # calc maxsize
    maxsize = 0
    for p in patches:
        if (p[0].shape[0] > maxsize) :
            maxsize= p[0].shape[0]
        if (p[0].shape[1] > maxsize) :
            maxsize = p[0].shape[1]
    if (maxsize > 50):
        pass
        #print "maxsize is " + str(maxsize)
    count = 0
    for p in patches:
        if (p[1] == True):
            count = count + 1
    #print 'only ' + str(count) + ' valid'
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
        cv2.imwrite(os.path.join(saveddir, str(index) + '.png'), final)

        train_set_x.append(final)
        train_set_y.append(p[2])
        label.append(index)
        index = index + 1

def generateSamples():
    '''
    读取 ./bigpic/ 下的所有图片
        每一张图片都是3230个汉字，都调用extractPatches和normalizePatches

    把归一化后的单字图片，放到 ./samples/ 下面
    '''
    if (os.path.exists('./samples/') == False):
        os.mkdir('./samples')
    '''
    fonts = ['simkai']
    size = ['36']
    morph = ['']
    '''
    fonts = ['simkai', 'simsun', 'simhei', 'simfang']
    #size = ['36', '42']
    size = ['36']
    #morph = ['', 'close33', 'dilate13', 'dilate31', 'dilate33', 'erode13', 'erode31', 'gblur11', 'gblur13', 'gblur15', 'gblur17']
    morph = ['']
    count = 0
    total = str(len(fonts) * len(size))
    for f in fonts:
        for s in size:
            boxname = './bigpic/cv.' + f + s + '.box'
            count = count + 1
            print('reading ' + str(count) + '/' + total + '\t-->\t' + boxname)
            for m in morph:
                picname = './bigpic/cv.' + f + s + '.' + m + '.tif'
                picname = picname.replace('..', '.')
                dirname = './samples/' + f + s + m + '/'
                patches = extractPatches([picname, boxname]);
                normalizePatches(patches, dirname)
    pass

def dumpData():
    '''
    生成数据集 data.pkl
    生成解码器 decoder.pkl
    '''
    print 'dumping data...'
    x = np.array(train_set_x)
    print(x.shape)
    x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
    l = np.array(label)

    f = open('./data.pkl', 'wb')
    train_set = [x, l]
    print 'saving data.pkl'
    cPickle.dump(train_set, f)
    f.close()

    f = open('./decoder.pkl', 'wb')
    y = train_set_y 
    y = y[0:3230]
    print 'dumping decoder...'
    cPickle.dump(y, f)
    f.close()

    os.system('ls -lh *.pkl')

if __name__== '__main__':
    s = time.time()
    generateSamples()
    e = time.time()
    print('\n cost ' + str(int((e - s) / 60)) + ' mins ' + 
          str(float('%0.3f'%((e - s) % 60))) + ' seconds to generate samples \n\n')

    s = time.time()
    dumpData()
    e = time.time()
    print('\n cost ' + str(int((e - s) / 60)) + ' mins ' + 
          str(float('%0.3f'%((e - s) % 60))) + ' seconds to dump data \n\n')

