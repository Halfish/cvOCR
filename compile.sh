#g++ $1 -o $2 -I/usr/local/include/tesseract/ -I/usr/local/include/leptonica/ -L/usr/local/lib/ -llept -ltesseract

#g++ test.cpp i18nText.cpp -o test -I/usr/include/freetype2 -L/usr/local/lib -lfreetype `pkg-config --cflags --libs tesseract opencv`  

g++ $1 -o $2 `pkg-config --cflags --libs tesseract opencv` -g 
