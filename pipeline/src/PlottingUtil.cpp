#include "../include/PlottingUtil.h"
#include "../include/SfmStructures.h"
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>

cv::Mat draw_features(cv::Mat image, Features features) {
    cv::Mat features_image;
    cv::drawKeypoints(image, features.key_points, features_image,
                      cv::Scalar::all(-1),
                      cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    return features_image;
}

cv::Mat draw_matches(cv::Mat left, cv::Mat right, Features features_left,
                     Features features_right, std::vector<cv::DMatch> matches) {
    cv::Mat matches_image;
    cv::drawMatches(
        left, features_left.key_points, right, features_right.key_points,
        matches, matches_image, cv::Scalar::all(-1), cv::Scalar::all(-1),
        std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    return matches_image;
}
