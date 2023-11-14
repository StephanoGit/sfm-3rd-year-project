#include "drawUtil.h"

#include <stdio.h>
#include <opencv2/opencv.hpp>

cv::Mat draw_features(ImageView image)
{
    cv::Mat features_image;
    cv::drawKeypoints(image.get_image(), image.get_keypoints(), features_image, cv::Scalar::all(-1), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    return features_image;
}

cv::Mat draw_matches(ImagePair pair)
{
    cv::Mat matches_image;
    cv::drawMatches(pair.get_image1().get_image(), pair.get_image1().get_keypoints(),
                    pair.get_image2().get_image(), pair.get_image2().get_keypoints(),
                    pair.get_good_matches(), matches_image,
                    cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(),
                    cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    return matches_image;
}