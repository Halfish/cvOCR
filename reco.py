#!/usr/bin/python
# -*- coding:utf-8 -*-
#########################################################################
# File Name: reco.py
# Author: Bruce Zhang
# mail: zhangxb.sysu@gmail.com
# Created Time: 2015年11月16日 星期一 18时10分50秒
#########################################################################

import cv2
import numpy as np
import os
import sys
import cnn
import cPickle

'''
读取所有textline，并根据region.txt中的位置信息，
逐个识别单字，给出识别结果，通过 python reco.py ./pic/4.jpg 调用

要有文件：
1. model_cv.h5
2. decoder_cv.pkl
3. model_eng.h5
4. decoder_eng.pkl
5. main
6. region.txt（main会生成)
'''

print('启动中文分类器...')
model_chi = cnn.build_model_chi()
model_chi.load_weights('./etc/model_cv.h5')
decoder_chi = cPickle.load(open('./etc/decoder_cv.pkl', 'rb'))

print('加载英文分类器...')
model_eng = cnn.build_model_eng()
model_eng.load_weights('./etc/model_eng.h5')
decoder_eng = cPickle.load(open('./etc/decoder_eng.pkl', 'rb'))
print('加载完毕...')

results = []

def normalize(img, meanHeight, mode):
    '''
    将图片标准化
        1. 加1/4的margin
        2. resize到(48, 48)
    '''
    h, w = img.shape
    size = meanHeight
    if (size < max(w, h)):
        size = max(w, h)

    if mode == 1:
        size = size + size / 2
    if mode == 2:
        size = size + size / 4
    normal = 255 * np.ones((size,size), np.uint8)
    normal[(size - h) / 2 : (size + h) / 2, (size - w) / 2 : (size + w) / 2] = img
    normal = cv2.resize(normal, (48, 48))

    return normal

def topFiveResults(r, language):
    '''
    给出识别的结果，返回准确率最高的5个结果
    '''
    r_pair = []
    for i in range(len(r)):
        r_pair.append([i, r[i]])
    r_pair.sort(lambda x, y: cmp(x[1], y[1]), reverse=True)
    r_pair = r_pair[0:3]
    if language == 'chi':
        for i in range(len(r_pair)):
            r_pair[i][0] = decoder_chi[r_pair[i][0]]
    if language == 'eng':
        for i in range(len(r_pair)):
            r_pair[i][0] = decoder_eng[r_pair[i][0]]
    return r_pair

def predict(normal, language):
    '''
    调用分类器，给出识别结果
    '''
    x = normal.reshape(1, 1, 48, 48)
    r = []
    if language == 'chi':
        r = model_chi.predict(x)
    if language == 'eng':
        r = model_eng.predict(x)
    return topFiveResults(r[0], language)


def recognizeCHI(filename, box, meanHeight):
    #识别整个textline
    textline = []
    img = cv2.imread('./tempFiles/textLine/' + filename, 0)
    for b in box:
        if b[4] == '1':
            # 若是汉字，则提取，让分类器识别
            word = img[int(b[1]):int(b[3]), int(b[0]):int(b[2])]
            normal = normalize(word, meanHeight, 2)
            r = predict(normal, 'chi')
            textline.append(r)
        else:
            textline.append([])
    results.append(textline)

def recognizeENG(filename, box, meanHeight, index):
    #识别整个textline
    count = 0
    img = cv2.imread('./tempFiles/textLine/' + filename, 0)
    for b in box:
        if b[4] == '0':
            # 若是英文，则提取，让分类器识别
            word = img[int(b[1]):int(b[3]), int(b[0]):int(b[2])]
            normal = normalize(word, meanHeight, 1)
            r = predict(normal, 'eng')
            results[index][count] = r
        count = count + 1

def recognizeTESS(filename, box, index):
    #识别整个textline
    count = 0
    img = cv2.imread('./tempFiles/textLine/' + filename, 0)
    for b in box:
        if b[4] == '2':
            #若是类型2，说明要用Tesseract识别
            word = img[int(b[1]):int(b[3]), int(b[0]):int(b[2])]
            cv2.imwrite('word.png', word)
            output = os.popen('tesseract word.png a -l eng 2> /dev/null && cat a.txt')
            r = output.read()
            r = r.strip()
            results[index][count] = [[r, 0.999]]
        count = count + 1

def run(language, region):
    for index in range(len(region)):
        r = region[index]
        filename = r[0][0]
        meanHeight = r[0][1]
        box = r[2]
        if language == 'chi':
            recognizeCHI(filename, box, meanHeight)
        elif language == 'eng':
            recognizeENG(filename, box, meanHeight, index)
        elif language == 'tess':
            recognizeTESS(filename, box, index)

def mergePatch(p1, p2):
    patch = []
    patch.append(str(min(int(p1[0]), int(p2[0]))))
    patch.append(str(min(int(p1[1]), int(p2[1]))))
    patch.append(str(max(int(p1[2]), int(p2[2]))))
    patch.append(str(max(int(p1[3]), int(p2[3]))))
    patch.append(str(max(int(p1[4]), int(p2[4]))))
    return patch

def refreshRegion(region):
    '''
    识别中文后，把识别率低且和类型2（会让Tesseract识别的整个单词）相邻的中文，
    归类到一起
    '''
    global results
    leni = len(region)
    for i in range(leni):
        lenj = len(region[i][2])
        for j in range(lenj):
            if j > 0 and region[i][2][j][4] == '1' and region[i][2][j-1][4] == '2':
                if (results[i][j][0][1] < 0.4):
                    region[i][2][j-1] = mergePatch(region[i][2][j], region[i][2][j-1])
                    region[i][2][j] = ['0', '0', '0', '0', '0']
            if j < lenj - 1 and region[i][2][j][4] == '1' and region[i][2][j+1][4] == '2':
                if (results[i][j][0][1] < 0.4):
                    region[i][2][j] = mergePatch(region[i][2][j], region[i][2][j+1])
                    region[i][2][j+1] = ['0', '0', '0', '0', '0']

    tmp_results = []
    reg = []
    leni = len(region)
    for i in range(leni):
        patch = []
        r = []
        lenj = len(region[i][2])
        for j in range(lenj):
            if not region[i][2][j] == ['0', '0', '0', '0', '0']:
                patch.append(region[i][2][j])
                r.append(results[i][j])
        reg.append((region[i][0], region[i][1], patch))
        tmp_results.append(r)
    results = tmp_results
    return reg

def loadRegion():
    # 把region.txt文件读取成变量
    region_txt = open('./region.txt', 'r')
    region = []
    filename = ''
    meanHeight = 0
    box = []
    rowIndex = []
    colIndex = []
    for line in region_txt:
        line = line.strip().split(' ')
        if len(line) == 4:
            box = []
            filename = line[0] + '.png'
            meanHeight = int(line[1])
            rowIndex = int(line[2])
            colIndex = int(line[3])
        if len(line) == 5:
            box.append(line)
        if len(line) == 1:
            r = ((filename, meanHeight), (rowIndex, colIndex), box)
            region.append(r)
    return region

def saveResultsToFile(filename):
    with open(filename, 'w') as f:
        for textline in results:
            f.write('textline\n\n')
            for word in textline:
                f.write('word\n')
                for candidate in word:
                    f.write(candidate[0] + ' --> ' + str(candidate[1]) + '\n')

def doReco(filepath):
    global results
    results = []
    os.system('mkdir -p ./tempFiles/textLine')
    os.system("./etc/main " + filepath)
    os.system("mv rotatedRects.png ./static/images/")
    region = loadRegion()
    run('chi', region)
    saveResultsToFile('results.txt')
    region = refreshRegion(region)
    run('eng', region)

    print('运行Tesseract...')
    run('tess', region)
    print '识别完毕!\n'
    os.system('rm a.txt')
    os.system('rm word.png')

    ret = []
    for textline in results:
        w = ''
        for word in textline:
            w = w + word[0][0]
        ret.append(w)

    final = []
    line = ret[0]
    for i in range(len(region)-1):
        (rowIndexi, colIndexi) = region[i][1]
        (rowIndexj, colIndexj) = region[i+1][1]
        if rowIndexi == rowIndexj:
            line = line + '\t' + ret[i+1]
        else:
            final.append(line)
            line = ret[i+1]
    l = len(region)
    if region[l-2][1][0] != region[l-1][1][0]:
        final.append(line)
    return final

if __name__ == '__main__':
    doReco(sys.argv[1])
