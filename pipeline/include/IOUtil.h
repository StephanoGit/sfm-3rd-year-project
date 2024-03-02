#ifndef __IOUTIL_FUNCTIONS
#define __IOUTIL_FUNCTIONS

#include "SfmStructures.h"
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>

#include <pugixml.hpp>

std::vector<cv::Mat> load_images(std::string directory, int resize_val,
                                 std::vector<std::string> &images_paths);
std::vector<cv::Mat> video_to_images(std::string directory, int step);
cv::Mat downscale_image(cv::Mat image, int width, int height);
void export_point_cloud(std::vector<PointCloudPoint> point_cloud,
                        std::string directory);
bool load_camera_intrinsics(std::string path, Intrinsics &intrinsics,
                            int downscale_factor);
#endif
