#!/bin/bash

# magic.sh
# 用于从文本和字体文件中生成特定的图片，切分成单字

FONTS=("微软雅黑" "楷体" "宋体")
FONTS_DIR=("msyh" "simkai" "simsun")

for i in `seq ${#FONTS[@]}`
do
    let j=$i-1
    if [ ! -x ${FONTS_DIR[$j]} ]; then
        mkdir ${FONTS_DIR[$j]}
    fi

    for exp in -3 -2 -1 0 1 2;
    do
        text2image \
            --text=./source/common3000_chi.txt \
            --font=${FONTS[$j]} \
            --fonts_dir=./fonts/ \
            --resolution=72 \
            --ptsize=48 \
            --char_spacing=0.2 \
            --exposure $exp \
            --outputbase=./pictures/cv.${FONTS_DIR[$j]}.exp$exp
    done
done
