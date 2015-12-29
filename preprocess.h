/*************************************************************************
	> File Name: preprocess.h
	> Author: Bruce Zhang
	> Mail: zhangxb.sysu@gmail.com 
	> Created Time: 2015年10月08日 星期四 09时39分55秒
 ************************************************************************/

#ifndef PRE_PROCESS_H
#define PRE_PROCESS_H

#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/features2d/features2d.hpp>
#include<iostream>
#include<cstdio>
#include<cmath>
using namespace std;

/*
 * 图片预处理类
 */
class PreImageProcessor {

public:
	PreImageProcessor(cv::Mat);
	~PreImageProcessor();
	void init();
    vector<cv::RotatedRect> getRotatedRects();
    vector<cv::Mat> getTextLines();
    int getMeanImageHeight();
    cv::Mat getGrayImage();
    cv::Mat getCleanImage();
    vector<pair<int, int> > getTextLineIndex();

	void drawRectangles(cv::Mat, const vector<cv::RotatedRect> &);
	void drawRectangles(cv::Mat, const vector<cv::Rect> &);
	void generateCleanImage();

private:
	cv::Mat morphologyProcess(const cv::Mat &);
	cv::Mat morphologyProcess2(const cv::Mat &);
	cv::Mat getROI(const cv::Mat &, cv::RotatedRect); 
	vector<cv::RotatedRect> findRotatedRects(cv::Mat, int);
    vector<cv::RotatedRect> findRotatedRectsWithMSER(cv::Mat); 
    cv::Mat eliminateVerLine(cv::Mat); 
	void rotatedRectsFilter(vector<cv::RotatedRect> &);
	void reFindRotatedRects();
    void reArrangeRotatedRects();
	void extractTextLines();
	void calcMeanImageHeight();
	void translateRotatedRect(vector<cv::RotatedRect> &, cv::RotatedRect); 

private:
    static const int MIN_AREA = 1500;
	static const int MODE_SHORT = 1;
	static const int MODE_LONG = 2;

	vector<cv::RotatedRect> mRotatedRects;	
	vector<cv::Mat> mTextLines;
    vector<pair<int, int> > mTLIndex; // textLineIndex; (rowIndex, colIndex)
	int mMeanImageHeight;
    cv::Mat mImage;
	cv::Mat mGrayImage;
	cv::Mat mCleanImage;
};


/*
 * description: 构造函数
 */
PreImageProcessor::PreImageProcessor(cv::Mat img) {
    this->mImage = img;
    cv::cvtColor(this->mImage, this->mGrayImage, CV_BGR2GRAY);
}


/*
 * 全部预处理步骤
 */
void PreImageProcessor::init() {
	//mRotatedRects = findRotatedRects(mGrayImage, MODE_LONG);
    cv::Mat gray;
    cv::cvtColor(eliminateVerLine(mImage), gray, CV_BGR2GRAY);
    mRotatedRects = findRotatedRectsWithMSER(gray);
	calcMeanImageHeight();
	//reFindRotatedRects();
    reArrangeRotatedRects();
	extractTextLines();
	generateCleanImage();
}

/*
 * 去掉竖直的表格线，形态学方法
 * input: 原始图片
 * output: 返回去掉了直线的图片
 */
 cv::Mat PreImageProcessor::eliminateVerLine(cv::Mat img) {
    cv::Mat gray, opening, closing, blur, adaptive, erosion, closing2, opening2;
    cv::Mat element1, element2, element3, element4;
	element1 = getStructuringElement(cv::MORPH_RECT, cv::Size(1, 15));
	element2 = getStructuringElement(cv::MORPH_RECT, cv::Size(1, 100));
	element3 = getStructuringElement(cv::MORPH_RECT, cv::Size(3, 1));
	element4 = getStructuringElement(cv::MORPH_RECT, cv::Size(1, 30));

    // 形态学操作，筛选出直线
    cv::cvtColor(img, gray, CV_BGR2GRAY);
    cv::morphologyEx(gray, opening, cv::MORPH_OPEN, element1);
    cv::morphologyEx(opening, closing, cv::MORPH_CLOSE, element2);
	cv::GaussianBlur(closing, blur, cv::Size(5, 5), 0, 0);
    cv::adaptiveThreshold(blur, adaptive, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY, 11, 2);
    cv::erode(adaptive, erosion, element3, cv::Point(-1, -1), 2);
    cv::morphologyEx(erosion, closing2, cv::MORPH_CLOSE, element4);
    cv::morphologyEx(closing2, opening2, cv::MORPH_OPEN, element4);

    int height = gray.rows;
    int width = gray.cols;
    cv::Mat newImage(height, width, CV_8UC1, cv::Scalar(255));
    cv::Rect rect(5, 5, width - 10, height - 10);
    cv::Mat roiNewImage = newImage(rect);
    cv::Mat roiClosing = opening2(rect);
    roiClosing.copyTo(roiNewImage);

    // 按照面积和形状，过滤掉不符合标准的直线
    int totalArea = gray.cols * gray.rows;
	vector<vector<cv::Point> > contours;
	vector<vector<cv::Point> > contours2;
	cv::findContours(newImage, contours, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
	int len = contours.size();
	for(int i = 0; i < len; ++ i) {
		if (cv::contourArea(contours[i]) > totalArea / 5) {
            continue;
		}
		cv::RotatedRect rRect = minAreaRect(cv::Mat(contours[i]));
        if (rRect.angle < -45) {
            int temp = rRect.size.height;
            rRect.size.height = rRect.size.width;
            rRect.size.width = temp;
        }
        if (rRect.size.width > 35 | rRect.size.height < 300) {
            continue;
        }
        contours2.push_back(contours[i]);
	}
    cv::Mat mask(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
    len = contours2.size();
    for (int i = 0; i < len; ++ i) {
        cv::drawContours(mask, contours2, i, cv::Scalar(255, 255, 255), CV_FILLED);
    }

    // 生成去掉了直线的图片
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(30, 2));
    cv::Mat closing3, img2, img3, img4;

    img2 = img & ~mask;
    cv::morphologyEx(img2, closing3, cv::MORPH_CLOSE, kernel);
    img3 = closing3 & mask;
    img4 = img2 | img3;

    return img4;
}


/*
 * rotatedRects过滤器:
 *		1. 调整偏转角度（因为有些可能是负值，长宽需要对调）
 *		2. 增加矩形的边缘margin
 *		3. 由于倾斜，可能会造成找到的矩形边框超过原图，
 *		    所以有四个if来检测
 *		4. 筛选长宽比不正常的矩形
 */
void PreImageProcessor::rotatedRectsFilter(vector<cv::RotatedRect> &origin) {
	vector<cv::RotatedRect> v;

	int len = origin.size();
    cout << len << " rotatedRects before filter" << endl;
	for (int i = 0; i < len; ++ i) {
		cv::RotatedRect rRect = origin[i];

        if (rRect.center.x < 0 || rRect.center.y < 0) {
            continue;
        }

		if(rRect.angle < -45) {
			rRect.angle += 90;
			swap(rRect.size.width, rRect.size.height);
		}
        cv::Rect rect = rRect.boundingRect();
        if (rect.x < 0) {
            rRect.size.width = (rRect.center.x - rRect.size.height / 2 * sin(rRect.angle / 360)) / cos(rRect.angle / 360) - 15;
        }
        if ((rect.x + rect.width) >= mGrayImage.cols) {
            rRect.size.width = (mGrayImage.cols - rRect.center.x - rRect.size.height / 2 * sin(rRect.angle / 360)) / cos(rRect.angle / 360) - 15;
        }
        if (rect.y < 0) {
            rRect.size.height = (rRect.center.y - rRect.size.width / 2 * sin(rRect.angle / 360)) / cos(rRect.angle / 360) - 15;
        }
        if ((rect.y + rect.height) >= mGrayImage.rows) {
            rRect.size.height = (mGrayImage.rows - rRect.center.y - rRect.size.width / 2 * sin(rRect.angle / 360)) / cos(rRect.angle / 360) - 15;
        }
        if (rRect.size.width<= 0) {
            rRect.size.width = 1;
        }
        if (rRect.size.height <= 0) {
            rRect.size.height = 1;
        }
		if(rRect.size.width * 2 > rRect.size.height) {
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
	cv::Mat element1, element2, element3, kernel;

	element1 = getStructuringElement(cv::MORPH_RECT, cv::Size(28, 3));
	element2 = getStructuringElement(cv::MORPH_RECT, cv::Size(20, 1));
	element3 = getStructuringElement(cv::MORPH_RECT, cv::Size(28, 3));
	kernel = getStructuringElement(cv::MORPH_RECT, cv::Size(12, 1));

	cv::Sobel(gray, sobel, CV_8U, 1, 0, 1, 1, 0);
	cv::GaussianBlur(sobel, blur, cv::Size(5, 5), 0, 0);
	cv::threshold(blur, binary, 0, 255, CV_THRESH_OTSU+CV_THRESH_BINARY);
	cv::dilate(binary, dilation, element1, cv::Point(-1, -1), 1);
	cv::erode(dilation, erosion, element2, cv::Point(-1, -1), 2);
	cv::dilate(erosion, dilation, element3, cv::Point(-1, -1), 3);
	cv::morphologyEx(dilation, closing, cv::MORPH_CLOSE, kernel);

    //cv::imwrite("sobel.png", sobel);
    //cv::imwrite("blur.png", blur);
    //cv::imwrite("dilation1.png", dilation);
    //cv::imwrite("binary.png", binary);
    //cv::imwrite("erosion.png", erosion);
    //cv::imwrite("dilation2.png", dilation);
    //cv::imwrite("close.png", closing);

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
 * input:	cv::Mat gray 灰度图
 * output:	vector<cv::RotatedRect> mRotatedRects 倾斜的矩形 
 */
vector<cv::RotatedRect> PreImageProcessor::findRotatedRects(cv::Mat gray, int mode) {
	cv::Mat closing;
	switch (mode) {
		case MODE_LONG:
			closing = morphologyProcess(gray);	break;
		case MODE_SHORT:
			closing = morphologyProcess2(gray);	break;
		default:	exit(1);
	}

	vector<cv::RotatedRect> rotatedRects;
	vector<vector<cv::Point> > contours;
	cv::findContours(closing, contours, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
	
	int len = contours.size();
	for(int i = 0; i < len; ++ i) {
		cv::RotatedRect rRect = minAreaRect(cv::Mat(contours[i]));
		if (cv::contourArea(contours[i]) > MIN_AREA) {
			rotatedRects.push_back(rRect);
		}
	}
	rotatedRectsFilter(rotatedRects);
	return rotatedRects;
}


/*
 * description: 用 MSER(Maximally Stable Extremal Region)和 Morphology 结合的方法，提取文本行
 * input:	cv::Mat gray 灰度图
 * output:	vector<cv::RotatedRect> mRotatedRects 倾斜的矩形 
 */
vector<cv::RotatedRect> PreImageProcessor::findRotatedRectsWithMSER(cv::Mat gray) {
    vector<cv::RotatedRect> rotatedRects;
    vector<vector<cv::Point> > regions;
    vector<vector<cv::Point> > regions2;
    cv::Mat mask(gray.rows, gray.cols, CV_8UC1, cv::Scalar(0));  
    vector<cv::Rect> rects;

    cv::Ptr<cv::MSER> mser = cv::MSER::create(1, 20);
    mser->detectRegions(gray, regions, rects);
    cout << "find " << regions.size() << " contours!" << endl;

    regions2.reserve(regions.size());
    for (int i = 0; i < regions.size(); ++ i) {
        float w = rects[i].width;
        float h = rects[i].height;
        //float ratio = w > h ? w / h : h / w;
        if (w < 500 & h < 500) {
            regions2.push_back(regions[i]);            
        }
    }
    cout << "only " << regions2.size() << " contours left!" << endl;

    cout << "drawing contours" << endl;

    for (int i = 0; i < regions2.size(); ++ i) {
        //cv::drawContours(mask, regions2, i, cv::Scalar(255), CV_FILLED, 8);
        for (int j = 0; j < regions2[i].size(); ++ j) {
            mask.at<uchar>(regions2[i][j]) = 255;
        }
    }
    cv::imwrite("mask.jpg", mask);

    cv::Mat kernel1 = getStructuringElement(cv::MORPH_RECT, cv::Size(5, 1));
    cv::Mat kernel2 = getStructuringElement(cv::MORPH_RECT, cv::Size(50, 1));
    cv::Mat dilation, closing;
    cv::dilate(mask, dilation, kernel1, cv::Point(-1, -1), 2);
    cv::morphologyEx(mask, closing, cv::MORPH_CLOSE, kernel2);

    cv::imwrite("dilation.png", dilation);
    cv::imwrite("closing.png", closing);
    
    vector<vector<cv::Point> > contours;
    cv::findContours(closing, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

    cout << "find " << contours.size() << " textlines" << endl;

    // filter
    int totalArea = gray.cols * gray.rows;
	for(int i = 0; i < contours.size(); ++ i) {
		cv::RotatedRect rRect = minAreaRect(cv::Mat(contours[i]));
        int area = cv::contourArea(contours[i]);
		if (area > MIN_AREA && area < totalArea / 4) {
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

cv::Mat PreImageProcessor::getROI(const cv::Mat &gray, cv::RotatedRect rotate) {
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
	return roi;
}

/*
 * description: 重新查找倾斜矩阵，主要是为了处理两行文字误合并到了一起的情况；
 */
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


bool cmp(cv::RotatedRect rect1, cv::RotatedRect rect2) {
    return rect1.center.y < rect2.center.y;
}

/*
 * description: 根据位置重新排列，以及合并rotatedRects
 */
void PreImageProcessor::reArrangeRotatedRects() {
    int len = mRotatedRects.size();
    sort(mRotatedRects.begin(), mRotatedRects.end(), cmp);

    int rowIndex = 0, colIndex = 0;
    mTLIndex.push_back(make_pair(rowIndex, colIndex));
    for (int i = 1; i < len; ++ i) {
        cv::RotatedRect rRect1 = mRotatedRects[i-1];
        cv::RotatedRect rRect2 = mRotatedRects[i];
        if ((rRect2.center.y - rRect1.center.y) < (rRect1.size.height + rRect2.size.height) / 6) {
            colIndex ++;
        } else {
            rowIndex ++;
            colIndex = 0;
        }
        mTLIndex.push_back(make_pair(rowIndex, colIndex));
    }

    // 排序
    for (int i = 0; i < len; ++ i) {
        int rowIndexi = mTLIndex[i].first;
        int colIndexi = mTLIndex[i].second;
        for (int j = i + 1; j < len; ++ j) {
            int rowIndexj = mTLIndex[j].first;
            int colIndexj = mTLIndex[j].second;
            if (rowIndexi != rowIndexj) {
                break;
            }
            if (mRotatedRects[i].center.x > mRotatedRects[j].center.x) {
                cv::RotatedRect tmp = mRotatedRects[i];
                mRotatedRects[i] = mRotatedRects[j];
                mRotatedRects[j] = tmp;
            }
        }
    }
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
    //cout << "len = " << len << endl;
	for (int i = 0; i < len; ++ i) {
        //cout << " i = " << i << endl;
		cv::RotatedRect rotate = mRotatedRects[i];
		cv::Rect rect = rotate.boundingRect();

        /*
        if (i == 80) {
            cout << "rows and cols = " << endl;
            cout << mGrayImage.rows << endl;
            cout << mGrayImage.cols << endl;
            cout << "rotate x, y, width, height" << endl;
            cout << rotate.center.x << "\t" << rotate.center.y << "\t" << rotate.size.width << "\t" << rotate.size.height << endl;
            cout << "rect" << endl;
            cout << rect.x << "\t" << rect.y << "\t" << rect.width << "\t" << rect.height << endl;
            cout << "angle = " << rotate.angle << endl;
        }
        */

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

		char name[128];
		sprintf(name, "./tempFiles/textLine/%d.png", i);
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
 * description: 在原图上画出找到的"带倾斜角度"的矩形框，并把图片存下来
 */
void PreImageProcessor::drawRectangles(cv::Mat src, const vector<cv::RotatedRect> &rotatedRects) {
	cv::Mat img = src.clone();
	int len = rotatedRects.size();
	for (int i = 0; i < len; ++ i) {
		cv::Point2f vertices[4];
		rotatedRects[i].points(vertices);

		for (int j = 0; j < 4; ++ j) {
			line(img, vertices[j], vertices[(j+1) % 4], cv::Scalar(0, 255, 0), 2, 8);
		}
	}
	cv::imwrite("rotatedRects.png", img);
}


/*
 * description: 在原图上画出找到的矩形框，并把图片存下来
 */
void PreImageProcessor::drawRectangles(cv::Mat src, const vector<cv::Rect> &rects) {
	cv::Mat img = src.clone();
	int len = rects.size();
	for (int i = 0; i < len; ++ i) {
		char text[128];
		sprintf(text, "%d", len - i - 1);
		putText(img, text, cv::Point(rects[i].x - 50, rects[i].y + 40),
				CV_FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 255, 0));

		rectangle(img, rects[i].tl(), rects[i].br(), cv::Scalar(0, 255, 0), 2, 8, 0);
	}
	cv::imwrite("rects.png", img);
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

vector<pair<int, int> > PreImageProcessor::getTextLineIndex() {
    return mTLIndex;
}

#endif
