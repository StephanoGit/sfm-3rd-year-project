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
void export_3d_points_to_txt(std::string file_name, cv::Mat points);
std::vector<ImageView> extract_frames_from_video(std::string directory, int step);

#endif
