/*************************************************************************
	> File Name: cut.cpp
	> Author: Bruce Zhang
	> Mail: zhangxb.sysu@gmail.com 
	> Created Time: 2015年10月09日 星期五 09时59分34秒
 ************************************************************************/

#ifndef CUT_H
#define CUT_H

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <cstdio>
#include <cmath>
#include <sys/stat.h>
using namespace std;

/*
 * 把大图片切分成小图片
 */

// 最小的Patch之间的距离，用于中文偏旁的合并
const int MIN_MARGIN = 6;

// 生成的单字距离边缘的空白的长度
const int PIC_PADDING = 7;

// 最小的Patch长度和高度，用于判断标点和特殊符号
const int MIN_PATCH_WIDTH = 15;
const int MIN_PATCH_HEIGHT= 15;

// 最小的Patch高和宽之间的相似度
const float MIN_SIMILIRITY = 0.8;

// 最小的若连通点的像素值，用于重切分
const int MIN_CUT_PIXES = 4;

// 平均误差H的上阈值
const int MAX_H = 50;

/*
 * 每一个Patch的类型定义
 */
enum PatchType {
    P_NOTYPE = 0,     // 未定义，会用自己训练的分类器识别
    P_HANZI = 1,      // 汉字
    P_ENG = 2,        // 英文字母（大小写），数字，大标点，会被Tesseract识别
    P_PUNC = 3,       // 小标点
    P_NOISE = 4       // 噪音
};

/*
 * description: 单个块的信息，包括
 *	-- start	起点列数 
 *	-- end		末点列数
 *	-- top		上面最高点的行数
 *	-- bottom	下面最低点的行数
 *	-- ptype    类型 
 */
struct Patch {
    Patch() {
        start = 0;
        end = 0;
        top = 0;
        bottom = 0;
        ptype = P_NOTYPE;
    }
	Patch(int s, int e, PatchType pt) {
		start = s;
		end = e;
		ptype = pt;
	}
	Patch(int s, int e, int t, int b, PatchType pt) {
		start = s;
		end = e;
		top = t;
		bottom = b;
		ptype = pt;
	}
	int start, end, top, bottom;
	PatchType ptype;
};

/*
 * 每一个Region的类型定义
 */
enum RegionType {
    R_NOTYPE = 0,       // 未定义
    R_HANZI = 1,        // 这行文本包含了很多汉字 
    R_ENG = 2,          // 这行文本是纯英文 
    R_NOISE = 3         // 这行文本是噪声
};

/*
 * description: 单张图片(单行文字)的信息
 *	-- img		原始图片
 *	-- patches	每个块信息
 */
struct Region {
	cv::Mat img;
	int meanHeight;
    int meanWidth;
	vector<Patch> patches;
    RegionType rtype;

    Region() {
        meanHeight = 1;
        meanWidth = 1;
        rtype = R_NOTYPE;
    }
};

/*
 * description: 为每个块找到界定高度的坐标，从上面往下找top，从下面往上面找bottom
 */
void findHeightForPatch(cv::Mat &img, Patch &patch) {
	int totalHeight = img.rows;
	int start = patch.start;
	int end = patch.end;
	// bool isAvailable = patch.isAvailable;
	
	for (int i = 0; i < totalHeight; ++ i) {
		int whiteCount = 0;
		for (int j = start; j < end; ++ j) {
			whiteCount += (1 - img.at<uchar>(i, j) / 255); 
		}
		if (whiteCount >= 1) {
			patch.top = i; 
			break;
		}
	}
		
	for (int i = totalHeight - 1; i >= 0; -- i) {
		int whiteCount = 0;
		for (int j = start; j < end; ++ j) {
			whiteCount += (1 - img.at<uchar>(i, j) / 255);
		}
		if (whiteCount >= 1) {
			patch.bottom = i; 
			break;
		}
	}

	// make sure the height can not be zero
	if (patch.bottom <= patch.top) {
		patch.bottom += 1;
	}
}

/*
 * description: 计算单行的所有Patch的平均高度，标点和特殊符号的Patch不算在内
 *	高度低于整个图片高度一半的，可能是小字母，可不算在内
 * output: int meanHeight 平均高度
 */
void findMeanHeightWidthForRegion(Region &region) {
	int len = region.patches.size();
	int count = 0;
	int sum = 0;
    int sumW = 0;
	for (int i = 0; i < len; ++ i) {
		int width = region.patches[i].end - region.patches[i].start;
		int height = region.patches[i].bottom - region.patches[i].top;
		if (width <= MIN_PATCH_WIDTH && height <= MIN_PATCH_HEIGHT // punctuation
				|| width <= MIN_PATCH_WIDTH && height >= 0.9 * region.img.rows	// symbol like "|"
				|| width <= (region.img.rows * 3 / 5) ) {
			continue;
		}

		sum += height;
        sumW += width;
		count ++;
	}

	// 0 check
	if (count == 0) {
		region.meanHeight = region.img.rows - 4;
        region.meanWidth = region.img.rows;
	} else {
		region.meanHeight = sum / count;
        region.meanWidth = sumW / count;
	}
}

/*
 * description: 为每行文字查找每个patch的高度，和单region的平均高度
 */
void findHeights(Region &region) {
	const int totalHeight = region.img.cols;
	int len = region.patches.size();
	for(int i = 0; i < len; ++ i) {
		findHeightForPatch(region.img, region.patches[i]);
	}
	findMeanHeightWidthForRegion(region);
}

/*
 * description: 在分割点上划线，分割文字
 * input: const Region &region
 */
void drawCutLine(const Region &region, int index, const char *dirname) {
	int len = region.patches.size();
	cv::Mat img  = region.img.clone();
	for (int i = 0; i < len; ++ i) {
		cv::Point startPoint = cv::Point(region.patches[i].start, region.patches[i].bottom);
		cv::Point endPoint = cv::Point(region.patches[i].end, region.patches[i].top);
		rectangle(img, startPoint, endPoint, cv::Scalar(0, 255, 0), 1, 8, 0);
	}
	char filename[256];
	sprintf(filename, "./%s/%d.png", dirname, index);
	cv::imwrite(filename, img);
}

/*
 * description: 读取图片，用垂直投影法切割单字
 * input: const cv::Mat gray 灰度图 
 * output: Region region 单张图片和图片分割的信息 
 */
 Region cut(const cv::Mat gray) {
	Region region;
	region.img = gray;

	int whiteCount[region.img.cols] = {0};
	for(int i = 0; i < region.img.cols; ++ i) {
		for (int j = 0; j < region.img.rows; ++ j) {
			whiteCount[i] += (1 - region.img.at<uchar>(j, i) / 255); // float and int ??
		}
	}

	int start = 0;
	int end = 0;
	for (int i = 0; i < region.img.cols - 1; ++ i) {
		if(whiteCount[i] == 0 && whiteCount[i+1] > 0) {
			start = i+1;
		}
		if(whiteCount[i] > 0 && whiteCount[i+1] == 0) {
			end = i+1;
			if ((end - start) > 3) {
				// make sure the width can not be zero
				region.patches.push_back(Patch(start, end, P_NOTYPE));
			}
		}
	}
	findHeights(region);

	return region;
}

/*
 * description 重切分Patch
 *      根据minCutPixes的大小，判定要切分的连通度强弱
 *      若切分后仍然存在宽度过长的情况，递归调用自己，并minCutPixes加一
 */
vector<Patch> doReCut(const Region &region, const Patch patch, const int minCutPixes) {
	cv::Mat img = region.img;
	vector<Patch> v;

	if (minCutPixes >= 10) {
		v.push_back(patch);
		return v;
	}
	
	int width = patch.end - patch.start;
	int height = patch.bottom - patch.top;
	int whiteCount[width + 5] = {0};

	for (int i = patch.start; i < patch.end; ++ i) {
		for (int j = 0; j < img.rows; ++ j) {
			whiteCount[i - patch.start] += (1 - img.at<uchar>(j, i) / 255);
		}
	}

	int s = 0, e;
	for (int i = 0; i < width; ++ i) {
		if (whiteCount[i] < minCutPixes && whiteCount[i+1] >= minCutPixes) {
			s = i;
		} 
		if (whiteCount[i] >= minCutPixes && whiteCount[i+1] < minCutPixes) {
			e = i+1;
			if ((e - s) > region.meanHeight * 5 / 4) {
				vector<Patch> tmp = doReCut(region, 
						Patch(s + patch.start, e + patch.start, P_NOTYPE), minCutPixes + 1);
				v.insert(v.end(), tmp.begin(), tmp.end());
			} else if ((e - s) > 2) {
				v.push_back(Patch(s + patch.start, e + patch.start, P_NOTYPE));
			}
		}
	}

	return v;
}

/*
 * description: 针对粘连情况严重的Patch进行重切分，通过宽度大于平均高度
 *	（即汉字平均宽度）* 1.5 的标准来判定是否需要重切分。
 *	input Region & 要更新的Region
 */
void reCut(Region &region) {
	vector<Patch> newPatches;
	int meanHeight = region.meanHeight;
	int len = region.patches.size();
	for (int i = 0; i < len; ++ i) {
		Patch patch = region.patches[i];
		if ((patch.end - patch.start) > region.meanHeight * 4 / 3) {
			vector<Patch> v = doReCut(region, patch, 1);
			newPatches.insert(newPatches.end(), v.begin(), v.end());	
		} else {
			newPatches.push_back(patch);
		}
	}
	newPatches.swap(region.patches);
	findHeights(region);
}


/*
 * description: 判断是不是一个合格的中文汉字区域，必须同时满足下面三个条件
 * 1. 满足 长宽比大于等于0.8
 * 2. 宽度和平均汉字高度比大于等于0.8
 * 3. 长度和平均汉字长度比大于等于0.8
 */
bool validChinesePatch(Patch patch, int standardH, int standardW) {
	float width = patch.end - patch.start;
	float height = patch.bottom - patch.top;
	
	float ratio = (width < height) ? width / height : height / width;
	float ratiow = (width < standardW) ? width / standardW : standardW / width;
	float ratioh = (height < standardH) ? height / standardH : standardH / height;
	
	if (ratio >= 0.83 && ratioh >= 0.85 && ratiow >= 0.85) {
        /*
		cout << endl << "valid" << endl;
		cout << "patch width = " << width << " height = " << height << endl;
		cout << "standardH is " << standardH << endl;
		cout << "standardW is " << standardW << endl;
		cout << "width / height is " << ratio << endl;
		cout << "width / standardW is " << ratiow << endl;
		cout << "height / standardH is " << ratioh << endl;
		cout << "valid" << endl << endl;
        */
		return true;
	}
    /*
	cout << endl << "not valid" << endl;
    cout << "standardH is " << standardH << endl;
    cout << "standardW is " << standardW << endl;
    cout << "width / height is " << ratio << endl;
    cout << "width / standardW is " << ratiow << endl;
    cout << "height / standardH is " << ratioh << endl;
	cout << "not valid" << endl << endl;
    */

	return false;
}


/*
 * description: 计算两个Patch的相似度，用于防止"2010"这种数字的合并问题
 * 必须满足一下两个条件才能给出相似的结论
 * 1. 长宽相差不超过 MIN_SIMILIRIT
 * 2. 高度起始点不超过 MIN_MARGIN 个像素
 */
bool isSimilar(Patch patch1, Patch patch2) {
	float width1 = patch1.end - patch1.start;
	float width2 = patch2.end - patch2.start;
	float height1 = patch1.bottom - patch1.top;
	float height2 = patch2.bottom - patch2.top;

	float ratio1 = (width1 < width2 ? width1 / width2 : width2 / width1);
	float ratio2 = (height1 < height2 ? height1 / height2: height2/ height1);
		
	if (ratio1 < MIN_SIMILIRITY || ratio2 < MIN_SIMILIRITY) {
		return false;
	}

	if(abs(patch1.top - patch2.top) > MIN_MARGIN
			|| abs(patch1.bottom - patch2.bottom) > MIN_MARGIN) {
		return false;
	}

	return true;
}

/*
 * description: 合并分离的中文 
 * 考虑下面的情况：
 *      1. 需要合并
 *          a. 三个Patch连在一起，合并后符合汉字的形状 
 *          b. 四个Patch连在一起，合并后符合汉字的形状 
 *      2. 不需要合并
 *          a. 连续的两个Patch距离过大，大于MIN_MARGIN 
 *          b. 当前Patch已经符合汉字的形状了
 *          c. 合并后的Patch不符合汉字的形状
 *          d. 两个将要合并的Patch非常相似，且高度小于平均高度，
 *              用来排除连续数字或者字母的情况
 *          e. 下一个要合并的Patch不是标点（根据长宽，位置，距离再下一个Patch的距离） 
 */
void merge(Region &region) {
	vector<Patch> newPatches;
	int len = region.patches.size();
	int i;
	for (i = 0; i < len - 1; ++ i) {
		bool canMerge = true;

		Patch patch1 = region.patches[i];
		Patch patch2 = region.patches[i + 1];

		Patch tmpPatch = Patch(patch1.start, patch2.end, 
				min(patch1.top, patch2.top), 
				max(patch1.bottom, patch2.bottom), P_HANZI);

		if (i + 2 < len) {
			Patch patch3 = region.patches[i + 2];
			Patch bigPatch = Patch(patch1.start, patch3.end, 
					min(tmpPatch.top, patch3.top),
					max(tmpPatch.bottom, patch3.bottom), P_HANZI);
			if (validChinesePatch(bigPatch, region.meanHeight, region.meanWidth)) {
				newPatches.push_back(bigPatch);
				i = i + 2;
				continue;
			}

            // 如果i-1和i+2都是数字，那么我们可以是误将数字合并了
            if (i > 0) {
                Patch patch0 = region.patches[i-1];
                if (!validChinesePatch(patch0, region.meanHeight, region.meanWidth) && 
                    !validChinesePatch(patch3, region.meanHeight, region.meanWidth)) {
                    canMerge = false;
                }
            }
		}
		
		if (i + 3 < len) {
			Patch patch4 = region.patches[i + 3];
			Patch bigPatch = Patch(patch1.start, patch4.end, 
					min(tmpPatch.top, patch4.top),
					max(tmpPatch.bottom, patch4.bottom), P_HANZI);
			if (validChinesePatch(bigPatch, region.meanHeight, region.meanWidth)) {
				newPatches.push_back(bigPatch);
				i = i + 3;
				continue;
			}
		}
        if (i + 4 < len) {
            Patch patch5 = region.patches[i + 4];
            Patch bigPatch = Patch(patch1.start, patch5.end, 
                    min(tmpPatch.top, patch5.top),
                    max(tmpPatch.bottom, patch5.bottom), P_HANZI);
            if (validChinesePatch(bigPatch, region.meanHeight, region.meanWidth)) {
                newPatches.push_back(bigPatch);
                i = i + 4;
                continue;
            }
        }

		if ((patch2.start - patch1.end) >= MIN_MARGIN){
			canMerge = false;
		}
		if (validChinesePatch(patch1, region.meanHeight, region.meanWidth)) {
            patch1.ptype = P_HANZI;
			canMerge = false;
		}
		if (!validChinesePatch(tmpPatch, region.meanHeight, region.meanWidth)) {
			canMerge = false;
		} 
		if (isSimilar(patch1, patch2) 
				&& (patch1.bottom - patch1.top) < region.meanHeight * 9 / 10 
				&& (patch2.bottom - patch2.top) < region.meanHeight * 9 / 10) {
			canMerge = false;
		}
		if ((patch2.end - patch2.start) < MIN_PATCH_WIDTH
				&& (patch2.bottom - patch2.top) < MIN_PATCH_HEIGHT
				&& (patch2.top > region.img.rows / 2)
				&& (i + 2) != len 
				&& (region.patches[i+2].start - patch2.end) > region.meanHeight / 3) {
			canMerge = false;
		}

		if (!canMerge) {
			newPatches.push_back(patch1);
		} else {
			newPatches.push_back(tmpPatch);
			++ i;
		}
	}
	if (i == len - 1) {
        Patch p = region.patches[i];
        if (validChinesePatch(p, region.meanHeight, region.meanWidth)) {
            p.ptype = P_HANZI;
        }
		newPatches.push_back(p);
	}
	// clear vector;
	newPatches.swap(region.patches);
}


/*
 * 从Region中找到每个Patch的中心坐标
 */
vector<int> getCentersForRegion(const Region &region) {
    vector<int> centers;
    int len = region.patches.size();
    for (int i = 0; i < len; ++ i) {
        Patch patch = region.patches[i];
        centers.push_back((patch.end + patch.start) / 2);
    }

    cout << "centers size = " << centers.size() << endl;
    /*
    for (int i = 0; i < centers.size(); ++ i) {
        cout << centers[i] << " ";
    }
    cout << endl << endl;
    */

    return centers;
}

/*
 * 根据Patch的中心坐标，找到相邻Patch之间的间距
 * 但是滤除了标点和与两边近似不等距的情况
 */
vector<int> getMarginsForRegion(const Region &region, vector<int> &centers) {
    vector<int> margins;
    int len = centers.size();
    for (int i = 1; i < len-1; ++ i) {
        Patch patch = region.patches[i];
        // 去掉由小标点符号产生的数据
        if ((patch.end - patch.start) < MIN_PATCH_WIDTH 
            || (patch.bottom - patch.top) < MIN_PATCH_HEIGHT) {
                continue;
            }
        // 不近似地等间距则略过
        if (abs(centers[i] * 2 - centers[i-1] - centers[i+1]) > 4) {
            continue;
        }
        // margin[i] 是 i 和 i-1 之间的距离
        margins.push_back(centers[i] - centers[i-1]);
    }

    cout << "margins\tsize = " << margins.size() << endl;
    for (int i = 0; i < margins.size(); ++ i) {
        cout << margins[i] << " ";
    }
    cout << endl << endl;

    return margins;
}


/*
 * 根据旧的b的值，把margins分成两部分，返回两个b
 * b1 >= b2 恒成立
 */
pair<int, int> divideMarginsForRegion(vector<int> &margins, int b) {
    int sum1 = 0, sum2 = 0, count1 = 0, count2 = 0;
    int len = margins.size();
    for (int i = 0; i < len; ++ i) {
        if (margins[i] > b) {
            sum1 += margins[i];
            count1 ++;
        } else {
            sum2 += margins[i];
            count2 ++;
        }
    }
    int b1 = 0, b2 = 0;
    if (count1 != 0) { b1 = sum1 / count1; }
    if (count1 != 0) { b2 = sum2 / count2; }

    return make_pair(b1, b2);
}

int reCalcBforRegion(vector<int> &margins, int b) {
    // 重新计算b
    int len = margins.size();
    int sum = 0, count = 0;
    for (int i = 0; i < len; ++ i) {
        int variance = (b - margins[i]) * (b - margins[i]); 
        if (variance < MAX_H) {
            sum += margins[i];
            count ++;
        }
    }
    int retb = b;
    if (count != 0) {
        retb = sum / count;
    }
    return retb;
}

/*
 * 返回：
 * 1. margins的平均值，即为b的值，即为我们拟合的文字平均宽度
 * 2. 第二个b的值，若不存在，返回0
 */
pair<int, int> calcBforRegion(vector<int> &margins) {
    // calc mean value (aka b) for margins
    int sum = 0;
    int len = margins.size();
    if (len == 0) {
        return make_pair(0, 0);
    }
    for (int i = 0; i < len; ++ i) {
        sum += margins[i];
    }
    int b = sum / len;

    // calc variance for margins
    int variance = 0;
    for (int i = 0; i < len; ++ i) {
        variance += (b - margins[i]) * (b - margins[i]); 
    }
    variance = variance / len; 
    
    int b1 = b, b2 = 0;
    if (variance > MAX_H) {
        // 方差太大，数字波动太大，要有两条直线来拟合
        pair<int, int> p = divideMarginsForRegion(margins, b);
        b1 = p.first; b2 = p.second;
    } else {
        // 只有一个b
        int b1 = reCalcBforRegion(margins, b);
    }

    return make_pair(b1, b2);
}

void printActuallyMargins(vector<int> centers) {
    int len = centers.size();
    cout << endl << "actually margins\tsize = " << len-1 << endl;
    for (int i = 0; i < len-1; ++ i) {
        cout << centers[i+1] - centers[i] << " ";
    }
    cout << endl << endl;
}

vector<int> doDivideLangRegion(Region &region, vector<int> centers, int b1, int b2) {
    vector<int> regionType;
    int len = region.patches.size();
    cout << "len of region is " << len << endl;
    for (int i = 0; i < len; ++ i) {
        cout << "i = " << i << " \t-->\t";
        if ( ((i != 0) && (abs(centers[i] - centers[i-1] - b1) <= 4))
            || ((i != (len-1)) && (abs(centers[i+1] - centers[i] - b1) <= 4))) {
                cout << "CHI" << endl;
            }
        else {
            cout << "NO CHI" << endl;
        }
    }
    return regionType;
}

/*
 * 根据汉字的等间距性，进行中英文区域划分的算法
 *      vector<int> centers 表示Patch的中心点的x坐标
 *      vector<int> margins 表示相邻Patch之间的距离
 */
void divideLangRegion(Region &region, int index) {
    cout << endl << "----------------------------" << endl;
    cout << "index  = " << index << endl << endl;
    vector<int> centers = getCentersForRegion(region);
    vector<int> margins = getMarginsForRegion(region, centers);
    pair<int, int> p = calcBforRegion(margins);
    int b1 = p.first;
    int b2 = p.second;
    cout << "b1 = " << b1 << " b2 = " << b2 << endl;
    printActuallyMargins(centers);
    vector<int> regionType = doDivideLangRegion(region, centers, b1, b2);
}


/*
 * 查找整个文本行的属性，全是英文的要拿去Tesseract中识别，噪音要去掉
 */
void findTextlineType(Region &region, int index) {
    int count_chi = 0;
    int count_eng = 0;
    int len = region.patches.size();
    for (int i = 0; i < len; ++ i) {
        if (region.patches[i].ptype == P_HANZI) {
            count_chi ++;
        } else {
            count_eng ++;
        }
    }
    if (count_chi <= 1 && count_eng >= 2) {
        region.rtype = R_ENG;
    } else if (count_chi >= 1) {
        region.rtype = R_HANZI;
    } else {
        region.rtype = R_NOISE;
    }

    /*
    cout << index;
    cout << "\t" << count_chi;
    cout << "\t" << count_eng;
    cout << "\t" << region.meanHeight;
    cout << "\t" << region.rtype << endl;
    */

    int count_thin = 0;
    // 文字的高度
    int standard = (region.img.rows + region.meanHeight) / 2;
    for (int i = 0; i < len; ++ i) {
        Patch patch = region.patches[i];
        int width = patch.end - patch.start;
        float height = patch.bottom - patch.top;
        if ((width / height < 0.8) && (width * 2 > standard)) {
            count_thin ++;
        }
    }
    if ((count_thin > count_chi) && (count_thin >= 3)) {
        region.rtype = R_ENG;
        Patch patch(0, region.img.cols, 0, region.img.rows, P_ENG);
        vector<Patch> newPatches;
        newPatches.push_back(patch);
        newPatches.swap(region.patches);
    }
}


/*
 * 1. 把Patch归类
 * 2. 把细长型，如“目”，扁的，如“一”，这种字挑出来，表示为汉字
 *      但是必须左边或者右边至少有一个是中文才行，同时考虑开头和结尾的特殊情况
 */
 void findPatchType(Region &region, int index) {
    int standardH = region.meanHeight;
    int standardW = region.meanWidth;
    int len = region.patches.size();
    int count = 0;      //一行不能执行三次这样的操作
    for (int i = 0; i < len; ++ i) {
        Patch patch1, patch2, patch3;
        if (i != 0) {
            patch1 = region.patches[i-1];
        }
        if (i != len -1){
            patch3 = region.patches[i+1];
        }
        patch2 = region.patches[i];
        if (patch2.ptype == P_HANZI) {
            continue;
        }
        float height = patch2.bottom - patch2.top;
        float width = patch2.end - patch2.start;
	    float ratiow = (width < standardW) ? width / standardW : standardW / width;
	    float ratioh = (height < standardH) ? height / standardH : standardH / height;
        bool valid = false;
        if (i == 0 && patch3.ptype == P_HANZI) {
            valid = true;
        } 
        if (i == (len - 1) && patch1.ptype == P_HANZI) {
            valid = true;
        }
        if (i != 0 && i != len - 1 && (patch1.ptype == P_HANZI || patch3.ptype == P_HANZI)) {
            valid = true;
        }
        if (count < 3 && valid && ((ratiow > 0.85 && ratioh < 0.5) || (ratioh > 0.85 && ratiow > 0.6))) {
            region.patches[i].ptype = P_HANZI;
            count ++;
        } 
    }
}


/*
 * 把第i行切分好的图片，存成dirname下的一个个小图片
 *
 * input: region 切分信息
 * input: index 第 i 行文字
 * input: dirname 要存放的单字图片的文件夹
 */
void saveTextLines(Region &region, int index, const char dirname[]) {
    // create a File
    // dirname/index/
    char path[32];
    sprintf(path, "%s/%d", dirname, index);
    mkdir(path, 0755);

    cv::Mat img = region.img;
    int len = region.patches.size();
    int count = 0;
    for (int i = 0; i < len; ++ i) {
        Patch patch = region.patches[i];
        if (!validChinesePatch(patch, region.meanHeight, region.meanWidth)) {
           continue; 
        }
        cv::Rect rect(patch.start, patch.top, 
                      patch.end - patch.start, patch.bottom - patch.top);
        cv::Mat roi = img(rect);
        cv::Mat pic = 255 * cv::Mat::ones(roi.rows + PIC_PADDING, roi.cols + PIC_PADDING, CV_8UC1);
        cv::Rect r((pic.cols - roi.cols) / 2,(pic.rows - roi.rows) / 2, roi.cols, roi.rows);
        roi.copyTo(pic(r));
        
        char filename[32]; 
        sprintf(filename, "%s/%d.png", path, count);
        count ++;
        cv::imwrite(filename, pic);
    }
}

/*
 * 由于英文分类器做的不好，所以把长的英文提取出来，让tessearact识别
 * 注意，这里的逻辑是，假设ptype只有0和1两种，即NOTYPE和CHI两种的情况
 */
void findEnglishText(Region &region, int index) {
    vector<Patch> newPatches;
    int len = region.patches.size();
    if (len <= 0) {
        return;
    }
    int engCount = 0;
    for (int i = 0; i < len; ++ i) {
        // 若连续的0出现超过三次，认为是长英文文本 
        if (region.patches[i].ptype == P_NOTYPE) {
            engCount ++;
        } else {
            if (engCount >= 3) {
                for (int j = 0; j < engCount; ++ j) {
                    region.patches[i-j-1].ptype = P_ENG;
                }
                Patch tmpPatch = Patch(region.patches[i-engCount].start, 
                                       region.patches[i-1].end, 0, 100, P_ENG);
                newPatches.push_back(tmpPatch);
                newPatches.push_back(region.patches[i]);
            } else {
                for (int j = 0; j < engCount; ++ j) {
                    newPatches.push_back(region.patches[i-engCount+j]);
                }
                newPatches.push_back(region.patches[i]);
            }
            engCount = 0;
        }
    }
    // 处理最后一个字符
    if (region.patches[len-1].ptype == P_NOTYPE) {
        if (engCount >= 3) {
            for (int j = 0; j < engCount; ++ j) {
                region.patches[len-j-1].ptype = P_ENG;
            }
            int start = region.patches[len-engCount].start;
            int end = region.patches[len-1].end;
            Patch tmpPatch = Patch(start, end, 0, 100, P_ENG);
            newPatches.push_back(tmpPatch);
        } else {
            for (int j = 0; j < engCount; ++ j) {
                newPatches.push_back(region.patches[len-engCount+j]);
            }
        }
    } 

    newPatches.swap(region.patches);
}


/*
 * 将Region的信息存到一个文件里去，方便让Python读取并识别
 * region.txt的格式如下：
 * 序列号 textline的高度
 * patch1 的start, top, end, bottom坐标，类型
 * patchn 的start...
 *  类型有，0表示未分类，会用英文分类器
 *          1表示为汉字，会用中文分类器
 *          2表示英文，会用Tesseract
 */
void saveRegionToFile(Region &region, int index, const char filename[], int rowIndex, int colIndex) {
    ofstream out;
    out.open(filename, ios::app);
    out << index << " " << region.meanHeight;
    out << " " << rowIndex << " " << colIndex << endl; 
    int len = region.patches.size();
    for (int i = 0; i < len; ++ i) {
        Patch patch = region.patches[i]; 
        out << patch.start << " " << patch.top << " ";
        out << patch.end << " " << patch.bottom << " ";
        /*
        float width = patch.end - patch.start;
        float height = patch.bottom - patch.top;
        out << "width = " << patch.end - patch.start << " ";
        out << "height = " << patch.bottom - patch.top << " ";
        out << "ratio = " << width / height << " ";
        */
        out << patch.ptype<< endl;
    }
    out << endl;
    out.close();
}

#endif
