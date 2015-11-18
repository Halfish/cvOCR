#!/bin/bash

# magic.sh
# 用于从待训练文本和字体文件中生成大图片和文字信息
# 1. 读取 ./source/common3000_chi.txt 中的3000+汉字
# 2. 字体文件在 ./fonts/ 下，共5种常见字体
# 3. 分辨率取 36, 42
# 4. 生成图片放在 ./bigpic/ 共 4 × 2 = 8 张图片

FONTS=("楷体" "宋体" "黑体" "仿宋")
FONTS_DIR=("simkai" "simsun" "simhei" "simfang")

mkdir -p bigpic 
for i in `seq ${#FONTS[@]}`
do
    let j=$i-1
    for size in 36 42;
    do
        text2image \
            --text=./source/common3000_chi.txt \
            --font=${FONTS[$j]} \
            --fonts_dir=./fonts/ \
            --resolution=110 \
            --ptsize=$size \
            --char_spacing=0.2 \
            --exposure=0 \
            --outputbase=./bigpic/cv.${FONTS_DIR[$j]}$size \
            --xsize=4800 \
            --ysize=4800
    done
done

ls ./bigpic/*.tif | wc
