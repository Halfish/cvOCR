/*************************************************************************
	> File Name: ergroup.cpp
	> Author: Bruce Zhang
	> Mail: zhangxb.sysu@gmail.com
	> Created Time: 2016年01月08日 星期五 16时27分13秒
 ************************************************************************/

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
using namespace std;

// 两个轮廓之间的最小距离低于这个阈值，则合并成一个
const int MIN_DIST = 5;

// 两个轮廓的初始距离，定义为不相邻
const int MAX_DIST = 1000000;

int abs(int a) {
    return (a > 0) ? a : -a;
}

int min(int a, int b) {
    return (a < b) ? a : b;
}

int min(int a, int b, int c, int d) {
    return min(min(a, b), min(c, d));
}

int max(int a, int b) {
    return (a > b) ? a : b;
}

int max(int a, int b, int c) {
    return max(max(a, b), c);
}

int max(int a, int b, int c, int d) {
    return max(max(a, b), max(c, d));
}

/*
 * 并查集算法，查找根节点编号
 *      这里没有用到复杂的路径压缩，因为一般路径不是很长（合并的轮廓数不是很多）
 */
int findFather(int x, vector<int> &father) {
    int f = x;
    while(father[f] != f) {
        f = father[f];
    }
    return f;
}

/*
 * 并查集算法，合并两个节点
 */
void unionXY(int x, int y, vector<int> &father) {
    int fx = findFather(x, father);
    int fy = findFather(y, father);
    if (fx != fy) {
        father[x] = y;
    }
}

/*
 * 计算两个轮廓之间的距离
 *      cnti, cntj      表示两个轮廓，
 *      recti, rectj    表示两个轮廓的矩形框，传进来而不是直接计算，是为了减少计算的复杂度
 */ 
int findDistance(vector<cv::Point> &cnti, vector<cv::Point> &cntj, cv::Rect &recti, cv::Rect &rectj) {
    int distance = MAX_DIST;

    int min_x_i = recti.x;
    int min_y_i = recti.y;
    int max_x_i = recti.x + recti.width;
    int max_y_i = recti.y + recti.height;

    int min_x_j = rectj.x;
    int min_y_j = rectj.y;
    int max_x_j = rectj.x + rectj.width;
    int max_y_j = rectj.y + rectj.height;

    int big_rect_width = max(max_x_i, max_x_j) - min(min_x_i, min_x_j);
    int big_rect_height = max(max_y_i, max_y_j) - min(min_y_i, min_y_j);

    if ((big_rect_width == recti.width && big_rect_height == recti.height) || 
        (big_rect_width == rectj.width && big_rect_height == rectj.height)) {
        // 1. 包含关系（注意这个if判断要放在相交关系的判断之前）
        distance = min(abs(min_x_i - min_x_j), abs(max_x_j - max_x_i), 
                       abs(min_y_i - min_y_j), abs(max_y_j - max_y_i));
    } else {
    if (big_rect_width < (recti.width + rectj.width) && 
        big_rect_height < (recti.height + rectj.height)) {
        // 2. 相交关系
        distance = 0;
    } else 
        // 3. 相离关系
        distance = max(min_x_i - max_x_j, min_x_j - max_x_i, min_y_i - max_y_j, min_y_j - max_y_i);
    }

    return distance;
}

void drawColorfulContours(cv::Mat img, vector<vector<cv::Point> > contours, string filename) {
    cout << "drawing " << filename << endl;
    vector<cv::Scalar> colors;
    colors.push_back(cv::Scalar(255, 0, 0));
    colors.push_back(cv::Scalar(0, 255, 0));
    colors.push_back(cv::Scalar(0, 0, 255));
    colors.push_back(cv::Scalar(0, 255, 255));
    colors.push_back(cv::Scalar(255, 0, 255));
    colors.push_back(cv::Scalar(255, 255, 0));
    cv::Mat cannyImage(img.rows, img.cols, CV_8UC3, cv::Scalar(0, 0, 0));
    for (int i = 0; i < contours.size(); ++ i) {
        cv::drawContours(cannyImage, contours, i, colors[i % 6], 1, 8);
    }
    cv::imwrite(filename, cannyImage);
}

/*
 * 筛选掉那些面积过大的轮廓
 */
void contoursFilter(vector<vector<cv::Point> > &contours) {
    vector<vector<cv::Point> > newContours;
    const int max_contour_area = 300 * 300;
    for (int i = 0; i < contours.size(); ++ i) {
        cv::Rect rect = cv::boundingRect(contours[i]);
        if (rect.width * rect.height < max_contour_area) {
            newContours.push_back(contours[i]);
        }
    }
    newContours.swap(contours);
}

void findCanny(string filepath) {
    cv::Mat img, gray, blur, canny;
    img = cv::imread(filepath);
    cv::cvtColor(img, gray, CV_BGR2GRAY);
    cv::GaussianBlur(gray, blur, cv::Size(5, 5), 0, 0);
    cv::Canny(blur, canny, 100, 200);
    cv::imwrite("canny.jpg", canny);

    // 画出contours
    vector<vector<cv::Point> > contours;
    cv::findContours(canny, contours, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);
    cout << "find " << contours.size() << " contours" << endl;
    drawColorfulContours(img, contours, "cannyImage.jpg");

    // 计算contours的一些基本位置信息
    vector<cv::Rect> rects;
    rects.reserve(contours.size());
    for (int i = 0; i < contours.size(); ++ i) {
        rects.push_back(cv::boundingRect(contours[i]));
    }

    // 用并查集合并临近的轮廓
    vector<int> father;
    father.reserve(contours.size());
    for (int i = 0; i < contours.size(); ++ i) {
        father.push_back(i);
    }
    for (int i = 0; i < contours.size(); ++ i) {
        for (int j = 0; j < contours.size(); ++ j) {
            if (i == j) continue;
            int dist = findDistance(contours[i], contours[j], rects[i], rects[j]); 
            if (dist < MIN_DIST) {
                unionXY(i, j, father);
            }
        }
    }
    cout << "end of find_union" << endl;
    vector<vector<cv::Point> > candidates;
    candidates.reserve(contours.size());
    for (int i = 0; i < contours.size(); ++ i) {
        vector<cv::Point> v;
        candidates.push_back(v);
    }
    for (int i = 0; i < father.size(); ++ i) {
        int f = findFather(i, father); 
        candidates[f].insert(candidates[f].begin(), contours[i].begin(), contours[i].end());
    }
    drawColorfulContours(img, candidates, "candidates.jpg");
    contoursFilter(candidates);
    drawColorfulContours(img, candidates, "contours.jpg");
}

int main(int argc, char **argv) {
    string filepath = "./pic/short.png";
    if (argc == 2) {
        filepath = argv[1];
    }

    findCanny(filepath); 
}
