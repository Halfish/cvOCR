#!/usr/bin/env python
# coding=utf-8
import os
import cv2

'''
moreSample.py
读取 ./bigpic/下的 tif 和 box 文件，加以数学形态学操作，生成更多的图片
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
        kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        kernel4 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
        kernel5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))

        closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel3)
        cv2.imwrite(n + ".close33.tif", closing)

        opening = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel1)
        cv2.imwrite(n + ".open13.tif", opening)
        opening = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel2)
        cv2.imwrite(n + ".open31.tif", opening)
        opening = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel3)
        cv2.imwrite(n + ".open33.tif", opening)

        dilation = cv2.dilate(img, kernel3, iterations = 1)
        cv2.imwrite(n + ".dilate33.tif", dilation)
        dilation = cv2.dilate(img, kernel4, iterations = 1)
        cv2.imwrite(n + ".dilate15.tif", dilation)
        dilation = cv2.dilate(img, kernel5, iterations = 1)
        cv2.imwrite(n + ".dilate51.tif", dilation)

        erosion = cv2.erode(img, kernel1, iterations = 1)
        cv2.imwrite(n + ".erode13.tif", erosion)
        erosion = cv2.erode(img, kernel2, iterations = 1)
        cv2.imwrite(n + ".erode31.tif", erosion)

if __name__ == '__main__':
    run()
