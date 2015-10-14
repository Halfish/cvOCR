/*************************************************************************
	> File Name: tesstool.h
	> Author: Bruce Zhang
	> Mail: zhangxb.sysu@gmail.com 
	> Created Time: 2015年10月08日 星期四 12时20分23秒
 ************************************************************************/

#ifndef TESS_TOOLS_H
#define TESS_TOOLS_H

#include <tesseract/baseapi.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <vector>
using namespace std;

pair<vector<char *>, vector<int> > recognizeByLine(const char *lang,
		cv::Mat img, vector<cv::Rect> rects) {
	tesseract::TessBaseAPI api;
	api.Init(NULL, lang, tesseract::OEM_DEFAULT);

	vector<char *> results;
	vector<int> confidences;

	api.SetImage((uchar *)img.data, img.cols, img.rows, 1, img.cols);

	int len = rects.size();
	for (int i = 0; i < len; ++ i) {
		cv::Rect rect = rects[i];
		api.SetRectangle(rect.x, rect.y, rect.width, rect.height);
		char *result = api.GetUTF8Text();
		int conf = api.MeanTextConf();

		results.push_back(result);
		confidences.push_back(conf);
	}

	api.Clear();
	api.End();

	return make_pair(results, confidences);
}

void printResults(pair<vector<char *>, vector<int> > p) {
	vector<char *> results = p.first;
	vector<int> confidence = p.second;

	int len = results.size();
	for (int i = 0; i < len; ++ i) {
		cout << "****************************" << endl << endl;
		cout << "The result of " << i << endl;
		cout << results[len - i - 1] << endl;

		cout << "The confidence of " << i << endl;
		cout << confidence[len - i - 1] << endl << endl;
		cout << "****************************" << endl;
	}
}

#endif
