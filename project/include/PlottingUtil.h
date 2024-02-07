#ifndef __PLOT_UTIL_FUNCTIONS
#define __PLOT_UTIL_FUNCTIONS

#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>

#include "SfmStructures.h"
/**
 * @brief Draws keypoints from given features on an image.
 *
 * Utilizes `cv::drawKeypoints` to overlay keypoints from `features` onto
 * `image`. The output is a new image with the keypoints visualized.
 *
 * @param image Input image to draw keypoints on.
 * @param features `Features` structure containing the keypoints.
 * @return New image with drawn keypoints.
 */
cv::Mat draw_features(cv::Mat image, Features features);

/**
 * @brief Visualizes matches between keypoints of two images.
 *
 * Creates an image showing lines connecting matched keypoints between `left`
 * and `right` images using `cv::drawMatches`.
 *
 * @param left First image with keypoints to match.
 * @param right Second image with keypoints to match.
 * @param features_left Keypoints from the first image.
 * @param features_right Keypoints from the second image.
 * @param matches Vector of keypoint matches between the two images.
 * @return Image visualizing the matches.
 */
cv::Mat draw_matches(cv::Mat left, cv::Mat right, Features left_features,
                     Features right_features, std::vector<cv::DMatch> matches);

#endif
