#ifndef __UTIL_FUNCTIONS
#define __UTIL_FUNCTIONS

#include <stdio.h>

#include <opencv2/opencv.hpp>

#include "ImageView.h"

std::vector<cv::Mat> load_images(std::string directory);
std::vector<ImageView> load_images_as_object(std::string directory,
                                             bool down_size);
std::vector<ImageView> extract_frames_from_video(std::string directory,
                                                 int step);

void export_3d_points_to_txt(std::string file_name,
                             std::vector<cv::Point3f> points);
void export_3d_points_to_txt(std::string file_name, cv::Mat points);
void export_K_to_json(std::string file_name, cv::Mat K);
void export_K_to_xml(std::string file_name, cv::Mat K);

struct ImagePair {
  size_t left, right;
};

struct Image2D3DPair {
  std::vector<cv::Point2f> points_2D;
  std::vector<cv::Point3f> points_3D;
};

struct Features {
  std::vector<cv::KeyPoint> key_points;
  std::vector<cv::Point2f> points;
  cv::Mat descriptors;
};

struct PointCloudPoint {
  cv::Point3f point;
  std::map<int, int> orgin_view;
};

// used to represent a point within the point cloud
struct Point_3D {
  // coordinates
  cv::Point3d point;

  // plotted through who
  std::pair<int, int> idxImage;

  // 2D correspondence
  std::pair<cv::Point2f, cv::Point2f> point_2D;
};

#endif
