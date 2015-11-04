#!/usr/bin/env python
# coding=utf-8

'''
extract.py
从图片中提取文字区域，并存储到 ./patches/ 下
每个单字对应一张图片，后面的分类器会读取单字来给出识别结果
'''

import cv2
import numpy as np
import math
import os.path

'''
输入一张灰度图，输出查找到的单字的矩形位置
'''
def findRegions(img, gray):
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    # close 去掉噪声
    closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel2)

    # 让字的笔画连在一起，好识别
    erosion = cv2.erode(closing, kernel1, iterations=2)

    # 有时候字的笔画接触了边缘，无法正确识别出轮廓，因此要加margin
    newImage = 255 * np.ones((gray.shape[0]+4, gray.shape[1]+4), np.uint8)
    newImage[2:-2, 2:-2] = erosion

    contours, hierarchy = cv2.findContours(newImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    region = []
    m = gray.shape[0] * gray.shape[1] * 4 / 5
    for i in range(len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        if area < 100 or area > m:
            continue
        region.append(cnt)

    '''
    show = 255 * np.ones((img.shape[0]+4, img.shape[1]+4, img.shape[2]), np.uint8)
    show[2:-2, 2:-2] = img 
    cv2.drawContours(show, [region[0]], 0, (0, 255, 0), 2)
    '''
    return region

def dist(p1, p2):
    ret = (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2
    return math.sqrt(ret)

'''
根据region和img把单字截取，存下来
'''
def savePatches(gray, region):
    #  
    mask = 255 * np.ones((gray.shape[0]+4, gray.shape[1]+4), np.uint8)
    mask[2:-2, 2:-2] = gray 
    if len(region) != 1:
        print "box number error..."
        return  
    cnt = region[0]
    rect = cv2.minAreaRect(cnt)
    x, y, w, h = cv2.boundingRect(cnt)
    box = cv2.cv.BoxPoints(rect)
    rectImg = mask.copy()
    cv2.rectangle(rectImg, (x, y), (x+w, y+h), (0, 255, 0), 2)
    m = mask[y:y+h, x:x+w]
    mm = 255 * np.ones((m.shape[0] + 10, m.shape[1] + 10), np.uint8)
    mm[5:-5, 5:-5] = m
    center = (w / 2 + 5, h / 2 + 5)
    angle = rect[2]
    if (angle < -45):
        angle = angle + 90
    M = cv2.getRotationMatrix2D(center, angle, 1)
    dst = cv2.warpAffine(mm, M, mm.shape)
    width = int(dist(box[0], box[1]))
    height= int(dist(box[1], box[2]))
    crop = cv2.getRectSubPix(dst, (width, height), center) 
    blur = cv2.GaussianBlur(crop, (1, 1), 0)
    ret, thres = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)

    return thres

def run(fullname):
    img = cv2.imread(fullname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    region = findRegions(img, gray)
    word = savePatches(gray, region)

    dirname, filename = os.path.split(fullname)
    cv2.imwrite(dirname + '/m' + filename, word)


if __name__ == "__main__":
    for f in os.listdir('./samples/'):
        for ff in os.listdir('./samples/' + f):
            run('./samples/' + f + '/' + ff)

