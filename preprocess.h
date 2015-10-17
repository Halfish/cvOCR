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

class PreImageProcessor {

public:
	PreImageProcessor(cv::Mat mGrayImage);
	~PreImageProcessor();
	void init();
    vector<cv::RotatedRect> getRotatedRects();
    vector<cv::Mat> getTextLines();
    int getMeanImageHeight();
    cv::Mat getGrayImage();
    cv::Mat getCleanImage();
	void drawRectangles(cv::Mat, const vector<cv::RotatedRect> &);
	void drawRectangles(cv::Mat, const vector<cv::Rect> &);
	void generateCleanImage();

private:
	cv::Mat morphologyProcess(const cv::Mat &);
	cv::Mat morphologyProcess2(const cv::Mat &);
	cv::Mat getROI(cv::Mat, cv::RotatedRect); 
	vector<cv::RotatedRect> findRotatedRects(cv::Mat, int);
	void rotatedRectsFilter(vector<cv::RotatedRect> &);
	void reFindRotatedRects();
	void extractTextLines();
	void calcMeanImageHeight();
	void translateRotatedRect(vector<cv::RotatedRect> &, cv::RotatedRect); 

private:
	static const int MAX_AREA = 2000;
	static const int MODE_SHORT = 1;
	static const int MODE_LONG = 2;

	vector<cv::RotatedRect> mRotatedRects;	
	vector<cv::Mat> mTextLines;
	int mMeanImageHeight;
	cv::Mat mGrayImage;
	cv::Mat mCleanImage;
};


/*
 * description: 构造函数，初始图片必须是灰度图
 */
PreImageProcessor::PreImageProcessor(cv::Mat gray) {
	this->mGrayImage = gray;
}


/*
 * 全部预处理步骤
 */
void PreImageProcessor::init() {
	mRotatedRects = findRotatedRects(mGrayImage, MODE_LONG);
	calcMeanImageHeight();
	reFindRotatedRects();
	extractTextLines();
	generateCleanImage();
//	drawRectangles(mGrayImage, mRotatedRects);
}


/*
 * rotatedRects过滤器:
 *		1. 调整偏转角度（因为有些可能是负值，长宽需要对调）
 *		2. 增加矩形的边缘margin
 *		3. 筛选长宽比不正常的矩形
 */
void PreImageProcessor::rotatedRectsFilter(vector<cv::RotatedRect> &origin) {
	vector<cv::RotatedRect> v;

	int len = origin.size();
	for (int i = 0; i < len; ++ i) {
		cv::RotatedRect rRect = origin[i];

		if(rRect.angle < -45) {
			rRect.angle += 90;
			swap(rRect.size.width, rRect.size.height);
		}
		if(rRect.size.width > rRect.size.height) {
			v.push_back(rRect);			
		}
	}
	v.swap(origin);
}


/*
 * description: 形态学处理，返回便于查找轮廓的图片
 *
 * input cv::Mat mGrayImage 原始灰度图
 * output cv::Mat closing 形态学处理后，便于查找轮廓的图片 
 */
cv::Mat PreImageProcessor::morphologyProcess(const cv::Mat &gray) {
	cv::Mat sobel, blur, binary, dilation, erosion, closing;
	cv::Mat element1, element2, kernel;

	element1 = getStructuringElement(cv::MORPH_RECT, cv::Size(20, 1));
	element2 = getStructuringElement(cv::MORPH_RECT, cv::Size(28, 3));
	kernel = getStructuringElement(cv::MORPH_RECT, cv::Size(12, 1));

	cv::Sobel(gray, sobel, CV_8U, 1, 0, 1, 1, 0);
	cv::GaussianBlur(sobel, blur, cv::Size(5, 5), 0, 0);
	cv::threshold(blur, binary, 0, 255, CV_THRESH_OTSU+CV_THRESH_BINARY);

	cv::dilate(binary, dilation, element2, cv::Point(-1, -1), 1);
	cv::erode(dilation, erosion, element1, cv::Point(-1, -1), 2);
	cv::dilate(erosion, dilation, element2, cv::Point(-1, -1), 3);
	cv::morphologyEx(dilation, closing, cv::MORPH_CLOSE, kernel);

	return closing;
}

/*
 * description: 形态学处理2，返回便于查找轮廓的图片	
 *			腐蚀力度小一点，用于小图片二次切分
 *
 * input cv::Mat mGrayImage 原始灰度图
 * output cv::Mat closing 形态学处理后，便于查找轮廓的图片 
 */
cv::Mat PreImageProcessor::morphologyProcess2(const cv::Mat &gray) {
	cv::Mat sobel, blur, binary, dilation, erosion, closing;
	cv::Mat element1, element2, kernel;

	element1 = getStructuringElement(cv::MORPH_RECT, cv::Size(20, 1));
	element2 = getStructuringElement(cv::MORPH_RECT, cv::Size(28, 3));
	kernel = getStructuringElement(cv::MORPH_RECT, cv::Size(9, 1));

	cv::Sobel(gray, sobel, CV_8U, 1, 0, 1, 1, 0);
	cv::GaussianBlur(sobel, blur, cv::Size(5, 5), 0, 0);
	cv::threshold(blur, binary, 0, 255, CV_THRESH_OTSU+CV_THRESH_BINARY);

	cv::dilate(binary, dilation, element2, cv::Point(-1, -1), 1);
	cv::erode(dilation, erosion, element1, cv::Point(-1, -1), 2);
	cv::dilate(erosion, dilation, element2, cv::Point(-1, -1), 2);
	cv::morphologyEx(dilation, closing, cv::MORPH_CLOSE, kernel);

	return closing;
}


/* description:	从灰度图中利用数学形态学的方法查找到文字行，
 *			返回一个带倾斜角的矩形
 *
 * input:	cv::Mat mGrayImage 灰度图
 * output:	vector<cv::RotatedRect> mRotatedRects 倾斜的矩形 
 */
vector<cv::RotatedRect> PreImageProcessor::findRotatedRects(cv::Mat img, int mode) {
	cv::Mat closing;
	switch (mode) {
		case MODE_LONG:
			closing = morphologyProcess(img);	break;
		case MODE_SHORT:
			closing = morphologyProcess2(img);	break;
		default:	exit(1);
	}

	vector<cv::RotatedRect> rotatedRects;
	vector<vector<cv::Point> > contours;
	cv::findContours(closing, contours, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
	
	int len = contours.size();
	for(int i = 0; i < len; ++ i) {
		cv::RotatedRect rRect = minAreaRect(cv::Mat(contours[i]));
		if (cv::contourArea(contours[i]) > MAX_AREA) {
			rotatedRects.push_back(rRect);
		}
	}
	rotatedRectsFilter(rotatedRects);

	return rotatedRects;
}


/*
 * description: 计算所有小图片中的平均高度
 */
void PreImageProcessor::calcMeanImageHeight() {
	int sum = 0;

	int len = mRotatedRects.size();
	for (int i = 0; i < len; ++ i) {
		sum += mRotatedRects[i].size.height;
	}

	mMeanImageHeight = sum / len;
}


void PreImageProcessor::translateRotatedRect(vector<cv::RotatedRect> &v, cv::RotatedRect rotate) {
	cv::Rect rect = rotate.boundingRect();
	int len = v.size();
	for (int i = 0; i < len; ++ i) {
		v[i].center.x += rect.x;
		v[i].center.y += rect.y;
	}
}

cv::Mat PreImageProcessor::getROI(cv::Mat gray, cv::RotatedRect rotate) {
	cv::Mat src = gray.clone();
	cv::Mat roi;
	cv::Rect rect = rotate.boundingRect();
	roi = src(rect);
	cv::Mat mask(rect.height, rect.width, CV_8U, cv::Scalar(0));
	cv::Point2f vertices[1][4];
	rotate.points(vertices[0]);
	cv::Point pt[4];
	for (int i = 0; i < 4; ++ i) {
		pt[i].x = vertices[0][i].x - rect.x;
		pt[i].y = vertices[0][i].y - rect.y;
	}
	const cv::Point *ppt[1] = {pt};
	int npt[] = {4};
	cv::fillPoly(mask, ppt, npt, 1, cv::Scalar(255));
	for (int i = 0; i < mask.rows; ++ i) {
		for (int j = 0; j < mask.cols; ++ j) {
			if (mask.at<uchar>(i, j) == 0) {
				roi.at<uchar>(i, j) = 255;
			}
		}
	}

	cv::imwrite("roi.png", roi);
	return roi;
}

void PreImageProcessor::reFindRotatedRects() {
	vector<cv::RotatedRect> newRotatedRects;
	int len = mRotatedRects.size();
	for (int i = 0; i < len; ++ i) {
		cv::RotatedRect rotate = mRotatedRects[i];
		if (rotate.size.height > mMeanImageHeight * 3 / 2) {
			cv::Mat roi = getROI(mGrayImage, rotate);
			vector<cv::RotatedRect> v = findRotatedRects(roi, MODE_SHORT);
			translateRotatedRect(v, rotate);
			newRotatedRects.insert(newRotatedRects.end(), v.begin(), v.end());	
		} else {
			newRotatedRects.push_back(rotate);
		}
	}
	rotatedRectsFilter(newRotatedRects);
	newRotatedRects.swap(mRotatedRects);
}


/*
 * description: 从倾斜的矩形中，提炼出文本行，并进行倾斜矫正和背景去噪
 *
 * input: cv::Mat mGrayImage 原始图片
 * input: vector<cv::RotatedRect> mRotatedRects 倾斜的矩形
 * output: vector<cv::Mat> mTextLines; 文本行小图片
 */
void PreImageProcessor::extractTextLines() {
	int len = mRotatedRects.size();
	for (int i = 0; i < len; ++ i) {
		cv::RotatedRect rotate = mRotatedRects[i];
		cv::Rect rect = rotate.boundingRect();

		cv::Point2f center(rect.width / 2, rect.height / 2);
		float angle = rotate.angle;

		cv::Mat roi = mGrayImage(rect);
		cv::Mat matrix = cv::getRotationMatrix2D(center, angle, 1.0);
		cv::Mat warp, crop, blur, adaptive;

		cv::warpAffine(roi, warp, matrix, roi.size(), CV_INTER_CUBIC);
		cv::getRectSubPix(warp, rotate.size, center, crop);
		cv::GaussianBlur(crop, blur, cv::Size(7, 7), 0);
		cv::adaptiveThreshold(blur, adaptive, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C,
				CV_THRESH_BINARY, 11, 2);

		mTextLines.push_back(adaptive);

		char name[16];
		sprintf(name, "./textLine/%d.png", i);
		cv::imwrite(name, adaptive);
	}
}


/*
 * description: 把诸多小图片，根据大致位置信息，贴到一张背景是白色的大图片中 
 *
 * input: cv::Mat mGrayImage 原始图片，只用到了其大小信息
 * input: vector<cv::RotatedRect> mRotatedRects 文本行位置信息
 * input: vector<cv::Mat> mTextLines 文本行
 *
 * output: vector<cv::Rect> rects 新图片文本行位置信息
 * output: cv::Mat newImage 新的图片
 */
void PreImageProcessor::generateCleanImage() {
	vector<cv::Rect> rects;
	mCleanImage = 255 * cv::Mat::ones(mGrayImage.rows, mGrayImage.cols, CV_8UC1);

	int len = mRotatedRects.size();
	for (int i = 0; i < len; ++ i) {
		cv::RotatedRect rotate = mRotatedRects[i];
		cv::Rect rect(rotate.center.x - rotate.size.width / 2.0,
				rotate.center.y - rotate.size.height / 2.0,
				rotate.size.width, rotate.size.height);
		rects.push_back(rect);

		cv::Mat roi = mCleanImage(rect);
		mTextLines[i].copyTo(roi);
	}

    cv::imwrite("newImage.png", mCleanImage);
}


/*
 * description: 在原图上画出找到的带倾斜角度的矩形框，并把图片存下来
 */
void PreImageProcessor::drawRectangles(cv::Mat src, const vector<cv::RotatedRect> &rotatedRects) {
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
void PreImageProcessor::drawRectangles(cv::Mat src, const vector<cv::Rect> &rects) {
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

vector<cv::RotatedRect> PreImageProcessor::getRotatedRects() {
    return mRotatedRects;
}

    vector<cv::Mat> PreImageProcessor::getTextLines() {
    return mTextLines;
}

int PreImageProcessor::getMeanImageHeight() {
    return mMeanImageHeight;
}

cv::Mat PreImageProcessor::getGrayImage() {
    return mGrayImage;
}

cv::Mat PreImageProcessor::getCleanImage() {
    return mCleanImage;
}

#endif
