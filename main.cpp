/*************************************************************************
	> File Name: main.cpp
	> Author: Bruce Zhang
	> Mail: zhangxb.sysu@gmail.com 
	> Created Time: 2015年10月08日 星期四 15时14分01秒
 ************************************************************************/

#include "tesstool.h"
#include "preprocess.h"
#include <iostream>
#include <vector>
using namespace std;

pair<vector<cv::Rect>, cv::Mat> mypair;

void preprocessImage(char *filename) {
	cv::Mat img, gray, newImage;
	img = cv::imread(filename);
	cvtColor(img, gray, CV_BGR2GRAY);

	vector<cv::RotatedRect> rotatedRects = findRotatedRects(gray);
//	drawRectangle(img, rotatedRects);

	vector<cv::Mat> textLines;
	textLines = extractTextLine(gray, rotatedRects); 

	mypair = generateCleanImage(gray, rotatedRects, textLines);
	drawRectangle(mypair.second, mypair.first);
}

void runTesseract() {
	vector<cv::Rect> rects = mypair.first;
	cv::Mat newImage = mypair.second;
	
	const char *lang = "cv";
	printResults(recognizeByLine(lang, newImage, rects));
}

int main(int argc, char** argv) {
	if (argc != 2) {
		cout << "Please specify the input image!" << endl;
		return -1;
	}

	preprocessImage(argv[1]);
//	runTesseract();
	
	return 0;
}
