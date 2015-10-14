/*************************************************************************
	> File Name: preprocess.h
	> Author: Bruce Zhang
	> Mail: zhangxb.sysu@gmail.com 
	> Created Time: 2015年10月08日 星期四 09时39分55秒
 ************************************************************************/

#ifndef PRE_PROCESS_H
#define PRE_PROCESS_H

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include<iostream>
using namespace std;

const int MAX_AREA = 2000;

/*
 * 1. 调整偏转角度（因为有些可能是负值，长宽需要对调）
 * 2. 增加矩形的边缘margin
 * 3. 筛选长宽比不正常的矩形
 */
vector<cv::RotatedRect> rotatedRectsFilter(const vector<cv::RotatedRect> rotatedRects) {
	vector<cv::RotatedRect> ret;

	int len = rotatedRects.size();
	for (int i = 0; i < len; ++ i) {
		cv::RotatedRect rRect = rotatedRects[i];
		cv::Size size = rRect.size;
		float angle = rRect.angle;

		if(angle < -45) {
			angle += 90;
			swap(size.width, size.height);
		}
		//size.height += 15;

		rRect.angle = angle;
		rRect.size = size;

		if(size.width > size.height) {
			ret.push_back(rRect);			
		}
	}

	return ret;
}


/* description:
 * 从灰度图中利用数学形态学的方法查找到文字行，返回一个带倾斜角的矩形
 *
 * input:	cv::Mat 灰度图
 * output:	vector<cv::RotatedRect> 倾斜的矩形 
 *
 * RotatedRect {
 *	cv::Size size; 
 *	float angle; 
 *	cv::Point2f center;
 * }
 */
vector<cv::RotatedRect> findRotatedRects(cv::Mat gray) {
	cv::Mat sobel, blur, binary, dilation, erosion, closing;
	cv::Mat element1 = getStructuringElement(cv::MORPH_RECT, cv::Size(20, 1));
	cv::Mat element2 = getStructuringElement(cv::MORPH_RECT, cv::Size(28, 3));
	cv::Mat kernel = getStructuringElement(cv::MORPH_RECT, cv::Size(10, 1));

	cv::Sobel(gray, sobel, CV_8U, 1, 0, 1, 1, 0);
	cv::GaussianBlur(sobel, blur, cv::Size(5, 5), 0, 0);
	cv::threshold(blur, binary, 0, 255, CV_THRESH_OTSU+CV_THRESH_BINARY);

	cv::dilate(binary, dilation, element2, cv::Point(-1, -1), 1);
	cv::erode(dilation, erosion, element1, cv::Point(-1, -1), 2);
	cv::dilate(erosion, dilation, element2, cv::Point(-1, -1), 3);
	cv::morphologyEx(dilation, closing, cv::MORPH_CLOSE, kernel);

	vector<vector<cv::Point> > contours;
	cv::findContours(dilation, contours, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
	
	vector<cv::RotatedRect> rotatedRects;
	int len = contours.size();
	for(int i = 0; i < len; ++ i) {
		cv::RotatedRect rRect = minAreaRect(cv::Mat(contours[i]));
		if (cv::contourArea(contours[i]) > MAX_AREA) {
			rotatedRects.push_back(rRect);
		}
	}

	return rotatedRectsFilter(rotatedRects);
}


/*
 * description: 从倾斜的矩形中，提炼出文本行，并进行倾斜矫正和背景去噪
 *
 * input: cv::Mat gray 原始图片
 * input: const vector<cv::RotatedRect> rotatedRects 倾斜的矩形
 * output: vector<cv::Mat> textLines 文本行的小图片
 */
vector<cv::Mat> extractTextLine(cv::Mat gray, const vector<cv::RotatedRect> rotatedRects) {
	vector<cv::Mat> textLines;
	int len = rotatedRects.size();
	for (int i = 0; i < len; ++ i) {
		cv::RotatedRect rotate = rotatedRects[i];
		cv::Rect rect = rotate.boundingRect();

		cv::Point2f center(rect.width / 2, rect.height / 2);
		float angle = rotate.angle;

		cv::Mat roi = gray(rect);
		cv::Mat matrix = cv::getRotationMatrix2D(center, angle, 1.0);
		cv::Mat warp, crop, blur, adaptive;

		cv::warpAffine(roi, warp, matrix, roi.size(), CV_INTER_CUBIC);
		cv::getRectSubPix(warp, rotate.size, center, crop);
		cv::GaussianBlur(crop, blur, cv::Size(7, 7), 0);
		cv::adaptiveThreshold(blur, adaptive, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C,
				CV_THRESH_BINARY, 11, 2);

		textLines.push_back(adaptive);
		char name[16];
		sprintf(name, "./textLine/%d.png", i);
		cv::imwrite(name, adaptive);
	}

	return textLines;
}

/*
 * description: 把诸多小图片，根据大致位置信息，贴到一张背景是白色的大图片中 
 *
 * input: cv::Mat gray 原始图片，只用到了其大小信息
 * input: vector<cv::RotatedRect> rotatedRects 文本行位置信息
 * input: vector<cv::Mat> textLines 文本行
 *
 * output: vector<cv::Rect> rects 新图片文本行位置信息
 * output: cv::Mat newImage 新的图片
 */
pair<vector<cv::Rect>, cv::Mat> generateCleanImage(cv::Mat gray, 
		vector<cv::RotatedRect> rotatedRects, vector<cv::Mat> textLines) {
	vector<cv::Rect> rects;
	cv::Mat newImage = 255 * cv::Mat::ones(gray.rows, gray.cols, CV_8UC1);

	int len = rotatedRects.size();
	for (int i = 0; i < len; ++ i) {
		cv::RotatedRect rotate = rotatedRects[i];
		cv::Rect rect(rotate.center.x - rotate.size.width / 2.0,
				rotate.center.y - rotate.size.height / 2.0,
				rotate.size.width, rotate.size.height);
		rects.push_back(rect);

		cv::Mat roi = newImage(rect);
		textLines[i].copyTo(roi);
	}

	return make_pair(rects, newImage);
}

/*
 * description: 在原图上画出找到的带倾斜角度的矩形框，并把图片存下来
 */
void drawRectangle(cv::Mat src, vector<cv::RotatedRect> rotatedRects) {
	cv::Mat img = src.clone();
	int len = rotatedRects.size();
	for (int i = 0; i < len; ++ i) {
		cv::Point2f vertices[4];
		rotatedRects[i].points(vertices);

		for (int j = 0; j < 4; ++ j) {
			line(img, vertices[j], vertices[(j+1) % 4], cv::Scalar(0, 255, 0), 3, 8);
		}
	}
	cv::imwrite("region.png", img);
}

/*
 * description: 在原图上画出找到的矩形框，并把图片存下来
 */
void drawRectangle(cv::Mat src, vector<cv::Rect> rects) {
	cv::Mat img = src.clone();
	int len = rects.size();
	for (int i = 0; i < len; ++ i) {
		char text[16];
		sprintf(text, "%d", len - i - 1);
		putText(img, text, cv::Point(rects[i].x - 50, rects[i].y + 40),
				CV_FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 255, 0));

		rectangle(img, rects[i].tl(), rects[i].br(), cv::Scalar(0, 255, 0), 2, 8, 0);
	}
	cv::imwrite("region.png", img);
}

#endif
