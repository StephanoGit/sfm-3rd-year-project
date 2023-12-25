#ifndef __COMMON_UTIL
#define __COMMON_UTIL

#include "SfmStructures.h"
#include <opencv2/core/types.hpp>

std::vector<cv::DMatch>
apply_lowes_ratio(const std::vector<std::vector<cv::DMatch>> knn_matches);

void keypoints_to_points(const std::vector<cv::KeyPoint> &kps,
                         std::vector<cv::Point2f> &pts);

void remove_outliers(const std::vector<cv::DMatch> &matches, cv::Mat &mask,
                     std::vector<cv::DMatch> &mask_matches);

void align_points_from_matches(const Features &left, const Features &right,
                               const std::vector<cv::DMatch> &matches,
                               Features &aligned_left, Features &aligned_right,
                               std::vector<int> &left_origin,
                               std::vector<int> &right_origin);

#endif
