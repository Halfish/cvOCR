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

void preprocessImage(char *filename) {
	cv::Mat img = cv::imread(filename);
	cv::Mat gray;
	cvtColor(img, gray, CV_BGR2GRAY);

	PreImageProcessor *pip = new PreImageProcessor(gray);
	pip->init();
}

void runTesseract() {
	
	const char *lang = "cv";
//	printResults(recognizeByLine(lang, newImage, rects));
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
