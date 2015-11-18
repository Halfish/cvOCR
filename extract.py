# coding: utf-8
#!/usr/bin/env python

'''
extract.py
从图片中提取文字区域，并存储到 ./patches/ 下
每个单字对应一张图片，后面的分类器会读取单字来给出识别结果
'''

import cv2
import numpy as np
import sys
import math

def normalize(img):
    '''
    归一化操作，先把图像放到max(w, h)大小的正方向中，再resize到32*32像素
    '''
    h = img.shape[0]
    w = img.shape[1]
    size = max(w, h)
    normal = 255 * np.ones((size, size), np.uint8)
    normal[(size - h) / 2: (size + h) / 2, (size - w) / 2: (size + w) / 2] = img
    normal = cv2.resize(normal, (36, 36))
    cv2.imwrite('normal.png', normal)
    return normal

def dist(p1, p2):
    '''
    计算两点之间的欧几里得距离
    ''' 
    ret = (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2
    return math.sqrt(ret)

def savePatches(gray, region):
    '''
    根据region和img把单字进行截取，倾斜矫正，裁剪
    返回归一化以后的图片和文字中心位置的横坐标
    '''
    # 排除不是汉字的情况
    if len(region) != 1:
        return None, None 
    #  还原带有margin的图片
    mask = 255 * np.ones((gray.shape[0]+4, gray.shape[1]+4), np.uint8)
    mask[2:-2, 2:-2] = gray 
    
    cnt = region[0]
    rect = cv2.minAreaRect(cnt)
    box = cv2.cv.BoxPoints(rect)
    x, y, w, h = cv2.boundingRect(cnt)
    
    # img1是大矩形框住的文字图片，因为旋转后会有黑色阴影，所以加10像素的边缘，得到img2
    img1 = mask[y:y+h, x:x+w]
    img2 = 255 * np.ones((img1.shape[0] + 10, img1.shape[1] + 10), np.uint8)
    img2[5:-5, 5:-5] = img1
    
    # 汉字的中心和旋转角度
    center = (w / 2 + 5, h / 2 + 5)
    angle = rect[2]
    if (angle < -45):
        angle = angle + 90
    
    # M为仿射变换的矩阵，得到旋转后的图片img3
    M = cv2.getRotationMatrix2D(center, angle, 1)
    img3 = cv2.warpAffine(img2, M, img2.shape)
    
    # 宽度和高度要计算，因为原矩形是带有倾斜角度
    width = int(dist(box[0], box[1]))
    height= int(dist(box[1], box[2]))
    
    # 裁剪，高斯模糊，二值化
    crop = cv2.getRectSubPix(img3, (width, height), center) 
    blur = cv2.GaussianBlur(crop, (1, 1), 0)
    ret, thres = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)

    # 返回的x坐标是在mask中的位置
    return normalize(thres), x + w / 2


def findRegions(gray):
    '''
    输入一张灰度图，输出查找到的单字的矩形位置
    '''
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    # close 去掉噪声
    closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel2)
    # 让字的笔画连在一起，好识别
    erosion = cv2.erode(closing, kernel1, iterations=2)
    # 有时候字的笔画接触了边缘，无法正确识别出轮廓，因此要加margin
    mask = 255 * np.ones((gray.shape[0]+4, gray.shape[1]+4), np.uint8)
    mask[2:-2, 2:-2] = erosion

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    region = []
    # m表示图片的面积 * 0.8
    m = gray.shape[0] * gray.shape[1] * 4 / 5 
    # 轮廓过滤
    for i in range(len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        if area < 100 or area > m:
            continue
        region.append(cnt)
    return region


def getData(pictures):
    '''
    给定pictures下的图片集，生成标准的测试集
    '''
    test_data = []
    for gray in pictures:
        pic, x = savePatches(gray, findRegions(gray))
        if pic != None:
            test_data.append([pic, x])
    # 根据横坐标排序
    test_data = [p for p, x in sorted(test_data, key = lambda x : x[1])]
    return test_data


def split(fullname):
    '''
    将验证码图片，根据色调分离出来，只将像素值排名前6的色调存成图片
    PS: 色调，Hue，指的是HSV编码中的H
    '''
    pictures = []
    # 读取原始验证码图片
    img = cv2.imread(fullname)
    # 转化成 HSV 空间
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 统计 Hue 通道的直方图
    hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
    # 按像素值多的排名，即同一颜色的统计像素值越多的，排名越靠前，
    # 所以arg[0]是肯定背景的像素值
    arg = np.argsort(-hist, axis=0)
    arg = arg.reshape(arg.shape[0])
    
    # 分离出前六名
    for i in range(6):
        newImage = 255 * np.ones((hsv.shape[0], hsv.shape[1]), np.uint8)
        for j in range(newImage.shape[0]):
            for k in range(newImage.shape[1]):
                if (hsv[j][k][0] == arg[i]):
                    newImage[j][k] = 0
        pictures.append(newImage)
    return pictures


if __name__ == "__main__":
    pictures = split(sys.argv[1])
    test_data = getData(pictures)
    print len(test_data), ' characters detected!'
    print test_data[0].shape

    for i in range(len(test_data)):
        cv2.imwrite('test' + str(i) + '.png', test_data[i])
