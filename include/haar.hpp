#ifndef _HAAR_
#define _HAAR_

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <stdio.h>

using namespace cv;
using namespace std;

void cvHaarWavelet(Mat &src,Mat &dst,int NIter);

void cvInvHaarWavelet(Mat &src,Mat &dst,int NIter, int SHRINKAGE_TYPE=0, float SHRINKAGE_T=50);

#endif //_HAAR_
