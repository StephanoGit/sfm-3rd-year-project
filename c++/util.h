#ifndef __UTIL_FUNCTIONS
#define __UTIL_FUNCTIONS

#include <stdio.h>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "ImageView.h"
#include "ImagePair.h"

std::vector<ImageView> load_images(std::string directory);
cv::Mat compute_K();
cv::Mat compute_F(ImagePair pair);
cv::Mat compute_E(cv::Mat K, cv::Mat F);

#endif
