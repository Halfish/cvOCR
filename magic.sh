#!/bin/bash

# magic.sh
# 用于从文本和字体文件中生成特定的图片，切分成单字
# 1. 创建 samples 文件夹
# 2. 创建 存放单字的文件夹
# 2. 根据 a.字体 b.曝光度 c.像素密度 生成字体文件

FONTS=("微软雅黑" "楷体" "宋体")
FONTS_DIR=("msyh" "simkai" "simsun")

mkdir -p samples
for i in `seq ${#FONTS[@]}`
do
    let j=$i-1
    for exp in  -1 0 1;
    do
        mkdir -p ./samples/${FONTS_DIR[$j]}$exp
        for size in 48;
        do
            mkdir -p ./samples/${FONTS_DIR[$j]}${exp}/$size
            text2image \
                --text=./source/common3000_chi.txt \
                --font=${FONTS[$j]} \
                --fonts_dir=./fonts/ \
                --resolution=72 \
                --ptsize=$size \
                --char_spacing=0.2 \
                --exposure $exp \
                --outputbase=./pictures/cv.${FONTS_DIR[$j]}.exp${exp}.size$size
        done
    done
done
