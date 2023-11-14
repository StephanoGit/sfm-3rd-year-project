#ifndef __UTIL_FUNCTIONS
#define __UTIL_FUNCTIONS

#include <stdio.h>
#include <string>
#include <vector>
#include <fstream>

#include <opencv2/opencv.hpp>

#include "ImageView.h"
#include "ImagePair.h"

std::vector<ImageView> load_images(std::string directory);

// cv::Mat compute_E(cv::Mat K, cv::Mat F, ImagePair &pair);
void drawEpipolarLines(const std::string &title, const cv::Mat F,
                       const cv::Mat &img1, const cv::Mat &img2,
                       const std::vector<cv::Point2f> points1,
                       const std::vector<cv::Point2f> points2,
                       const float inlierDistance);

void depl(cv::Mat image_left, cv::Mat image_right, cv::Mat fundemental, std::vector<cv::Point2f> selPoints1, std::vector<cv::Point2f> selPoints2);
void export_3d_points_to_txt(std::string file_name, cv::Mat points);
void perform_triangulation(ImagePair pair, cv::Mat K, cv::Mat prev_R, cv::Mat prev_t, cv::Mat curr_R, cv::Mat curr_t, int i);
// void sfm_between_images(ImageView img1, ImageView img2, cv::Mat K);

#endif
