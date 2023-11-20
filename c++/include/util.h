#ifndef __UTIL_FUNCTIONS
#define __UTIL_FUNCTIONS

#include <stdio.h>

#include <opencv2/opencv.hpp>

#include "../include/ImagePair.h"
#include "../include/ImageView.h"

std::vector<cv::Mat> load_images(std::string directory);
std::vector<ImageView> load_images_as_object(std::string directory, bool down_size);
void export_3d_points_to_txt(std::string file_name, std::vector<cv::Point3f> points);
std::vector<ImageView> extract_frames_from_video(std::string directory, int step);

void export_K_to_json(std::string file_name, cv::Mat K);
void export_K_to_xml(std::string file_name, cv::Mat K);

#endif
