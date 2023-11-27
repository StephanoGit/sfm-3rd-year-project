#ifndef __UTIL_FUNCTIONS
#define __UTIL_FUNCTIONS

#include <opencv2/opencv.hpp>

std::vector<cv::Mat> load_images(std::string directory);
std::vector<cv::Mat> video_to_images(std::string directory, int step);
cv::Mat downscale_image(cv::Mat image, int width, int height);
void export_point_cloud(cv::Mat point_cloud, std::string directory);

bool import_intrinsics(std::string directory, cv::Mat &K, cv::Mat &d);

#endif
