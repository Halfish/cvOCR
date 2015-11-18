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
#include "cut.h"
using namespace std;

/*
 * tesseract调用
 */ 

struct RecoResult {
    vector<int> confs;
    vector<char *> results;
};

/*
 * description: 识别整行的文字，单个文字识别
 */
RecoResult recognizeByTextLine(const char *lang, cv::Mat img, Region region) {
	tesseract::TessBaseAPI api;
    RecoResult recoResult;

	api.Init(NULL, lang, tesseract::OEM_DEFAULT);
	api.SetImage((uchar *)img.data, img.cols, img.rows, 1, img.cols);
    api.SetPageSegMode(tesseract::PSM_SINGLE_CHAR);

	int len = region.patches.size();
	for (int i = 0; i < len; ++ i) {
        Patch patch = region.patches[i];
        api.SetRectangle(patch.start, patch.top, 
                         patch.end - patch.start, patch.bottom - patch.top);
        char *result = api.GetUTF8Text();
        int conf = api.MeanTextConf();
        recoResult.results.push_back(result);
        recoResult.confs.push_back(conf);
	}

	api.Clear(); api.End();
	return recoResult;
}

void printResults(RecoResult recoResult) {

	int len = recoResult.results.size();
	for (int i = len - 1; i >= 0 ; -- i) {
//		cout << "****************************" << endl << endl;
//		cout << "The result of " << i << endl;
		cout << recoResult.results[len - i - 1] << " " ;

//		cout << "The confidence of " << i << endl;
//		cout << recoResult.confs[len - i - 1] << endl << endl;
//		cout << "****************************" << endl;
	}
}

#endif
