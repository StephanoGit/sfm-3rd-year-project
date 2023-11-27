#ifndef __DRAW_UTIL_FUNCTIONS
#define __DRAW_UTIL_FUNCTIONS

#include <opencv2/opencv.hpp>

cv::Mat draw_features(cv::Mat image);
cv::Mat draw_matches(cv::Mat left, cv::Mat right);
cv::Mat draw_epipolar_lines(cv::Mat left, cv::Mat right);

#endif
