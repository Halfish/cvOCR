#!/usr/bin/env python
# coding=utf-8
import os
import cv2

'''
moreSample.py
读取 ./bigpic/下的 tif 和 box 文件，加以数学形态学操作，生成更多的图片
    1. 闭合操作(close) 让字体间笔画断开
    2. 膨胀（dilation）让笔画变细
    3. 腐蚀（erode）让笔画变粗
    4. 模糊（blur） 让笔画模糊，二值化以后，产生笔画断连等
共 1 + 3 + 2 + 4，再加原图，数量扩大了11倍 `ls ./bigpic/*.tif | wc -l`
'''

def getPrefixNames():
    names = []
    for f in os.listdir('./bigpic/'):
        fullname = os.path.splitext(f)
        if (fullname[1] == '.box'):
            names.append(fullname[0])
    return names

def run():
    names = getPrefixNames()
    for n in names:
        print('processing ' + n + '....')
        n = './bigpic/' + n
        img = cv2.imread(n + '.tif', 0)

        kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))

        dilation = cv2.dilate(img, kernel1, iterations = 1)
        blur = cv2.GaussianBlur(dilation, (13, 13), 0)
        adaptive = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        cv2.imwrite(n + ".dilate13.tif", adaptive)

        dilation = cv2.dilate(img, kernel2, iterations = 1)
        blur = cv2.GaussianBlur(dilation, (13, 13), 0)
        adaptive = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        cv2.imwrite(n + ".dilate31.tif", adaptive)

        erosion = cv2.erode(img, kernel1, iterations = 1)
        blur = cv2.GaussianBlur(erosion, (13, 13), 0)
        adaptive = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        cv2.imwrite(n + ".erode13.tif", adaptive)

        erosion = cv2.erode(img, kernel2, iterations = 1)
        blur = cv2.GaussianBlur(erosion, (13, 13), 0)
        adaptive = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        cv2.imwrite(n + ".erode31.tif", adaptive)

        blur = cv2.GaussianBlur(img, (11, 11), 0)
        adaptive = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        cv2.imwrite(n + ".gblur11.tif", adaptive)
        blur = cv2.GaussianBlur(img, (13, 13), 0)
        adaptive = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        cv2.imwrite(n + ".gblur13.tif", adaptive)
        blur = cv2.GaussianBlur(img, (15, 15), 0)
        adaptive = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        cv2.imwrite(n + ".gblur15.tif", adaptive)

if __name__ == '__main__':
    run()
