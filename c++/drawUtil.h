#ifndef __DRAW_UTIL_FUNCTIONS
#define __DRAW_UTIL_FUNCTIONS

#include <stdio.h>
#include <string>
#include <vector>
#include <fstream>

#include <opencv2/opencv.hpp>

#include "ImageView.h"
#include "ImagePair.h"

cv::Mat draw_features(ImageView image);
cv::Mat draw_matches(ImagePair pair);
cv::Mat draw_epipolar_lines(ImagePair pair);

#endif