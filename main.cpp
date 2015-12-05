/*************************************************************************
	> File Name: main.cpp
	> Author: Bruce Zhang
	> Mail: zhangxb.sysu@gmail.com 
	> Created Time: 2015年10月08日 星期四 15时14分01秒
 ************************************************************************/

#include "preprocess.h"
#include "cut.h"
#include <iostream>
#include <vector>
#include <fstream>
using namespace std;


/*
 * 读取图片简历，分析存储文本行（textLine），切分（cut），
 * 重切分（recut），合并（merge）
 */

void preprocessImage(char *filename) {
	cv::Mat img = cv::imread(filename);
	cv::Mat gray;
	cvtColor(img, gray, CV_BGR2GRAY);

	PreImageProcessor *pip = new PreImageProcessor(gray);
	pip->init();
    
    vector<cv::Mat> textLines = pip->getTextLines();
    vector<cv::RotatedRect> rotatedRects = pip->getRotatedRects();
    pip->drawRectangles(img, pip->getRotatedRects());

    // 创建和清空文本
    fstream out("region.txt", ios::out);
    out.close();
    int len = textLines.size();
    char myfile[16];
    //cout << "序号\t汉字数\t英文数\t高度\t类型" << endl;
    for (int i = 0; i < len; ++ i) {
        Region region = cut(textLines[i]);
        drawCutLine(region, i, "./tempFiles/cut");
        reCut(region);
        drawCutLine(region, i, "./tempFiles/recut");
        merge(region);
        drawCutLine(region, i, "./tempFiles/merge");
        findTextlineType(region, i);
        findPatchType(region, i);
        findEnglishText(region, i);
        saveRegionToFile(region, i, "region.txt");
    }

    pip->generateCleanImage();
}

int main(int argc, char** argv) {
	if (argc != 2) {
		cout << "Please specify the input image!" << endl;
		return -1;
	}

	preprocessImage(argv[1]);
	
	return 0;
}
