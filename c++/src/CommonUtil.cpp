#include "../include/SfmStructures.h"
#include <opencv2/core/mat.hpp>

std::vector<cv::DMatch>
apply_lowes_ratio(const std::vector<std::vector<cv::DMatch>> knn_matches) {
    std::vector<cv::DMatch> good_matches;
    for (int i = 0; i < knn_matches.size(); i++) {
        if (knn_matches[i][0].distance < 0.75 * knn_matches[i][1].distance) {
            good_matches.push_back(knn_matches[i][0]);
        }
    }
    return good_matches;
}

void keypoints_to_points(const std::vector<cv::KeyPoint> &kps,
                         std::vector<cv::Point2f> &pts) {
    pts.clear();
    for (const auto &kp : kps) {
        pts.push_back(kp.pt);
    }
}

void remove_outliers(const std::vector<cv::DMatch> &matches, cv::Mat &mask,
                     std::vector<cv::DMatch> &mask_matches) {
    mask_matches.clear();
    for (size_t i = 0; i < matches.size(); i++) {
        if (mask.at<uchar>(i)) {
            mask_matches.push_back(matches[i]);
        }
    }
}

void align_points_from_matches(const Features &left, const Features &right,
                               const std::vector<cv::DMatch> &matches,
                               Features &aligned_left, Features &aligned_right,
                               std::vector<int> &left_origin,
                               std::vector<int> &right_origin) {
    aligned_left.key_points.clear();
    aligned_right.key_points.clear();
    aligned_left.descriptors = cv::Mat();
    aligned_right.descriptors = cv::Mat();

    for (size_t i = 0; i < matches.size(); i++) {
        aligned_left.key_points.push_back(left.key_points[matches[i].queryIdx]);
        aligned_right.key_points.push_back(
            right.key_points[matches[i].trainIdx]);

        aligned_left.descriptors.push_back(
            left.descriptors.row(matches[i].queryIdx));
        aligned_right.descriptors.push_back(
            right.descriptors.row(matches[i].trainIdx));

        left_origin.push_back(matches[i].queryIdx);
        right_origin.push_back(matches[i].trainIdx);
    }

    keypoints_to_points(aligned_left.key_points, aligned_left.points);
    keypoints_to_points(aligned_right.key_points, aligned_right.points);
}
