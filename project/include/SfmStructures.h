#ifndef __SFM_STRUCTURES
#define __SFM_STRUCTURES

#include <opencv2/opencv.hpp>
#include <stdio.h>

struct Intrinsics {
  cv::Mat K;
  cv::Mat K_inv;
  cv::Mat d;
};

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

#endif
