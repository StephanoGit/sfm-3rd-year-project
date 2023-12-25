#ifndef __PLOT_UTIL_FUNCTIONS
#define __PLOT_UTIL_FUNCTIONS

#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>

#include "SfmStructures.h"

cv::Mat draw_features(cv::Mat image, Features features);
cv::Mat draw_matches(cv::Mat left, cv::Mat right, Features left_features,
                     Features right_features, std::vector<cv::DMatch> matches);
cv::Mat draw_epipolar_lines(cv::Mat left, cv::Mat right);

#endif
