#include "../include/drawUtil.h"

#include <stdio.h>

#include <opencv2/opencv.hpp>

cv::Mat draw_features(ImageView image) {
    cv::Mat features_image;
    cv::drawKeypoints(image.get_image(), image.get_keypoints(), features_image, cv::Scalar::all(-1), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    return features_image;
}

cv::Mat draw_matches(ImagePair pair) {
    cv::Mat matches_image;
    cv::drawMatches(pair.get_image1().get_image(), pair.get_image1().get_keypoints(),
                    pair.get_image2().get_image(), pair.get_image2().get_keypoints(),
                    pair.get_good_matches(), matches_image,
                    cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(),
                    cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    return matches_image;
}

cv::Mat draw_epipolar_lines(ImagePair pair) {
    cv::RNG rng(0);
    cv::Mat epipolar_lines_image(pair.get_image1().get_image().rows,
                                 pair.get_image1().get_image().cols + pair.get_image2().get_image().rows,
                                 pair.get_image1().get_image().type());

    cv::Mat image1 = pair.get_image1().get_image().clone();
    cv::Mat image2 = pair.get_image2().get_image().clone();

    std::vector<cv::Vec3f> epilines1, epilines2;
    cv::computeCorrespondEpilines(pair.get_image1_good_matches(), 1, pair.get_F(), epilines1);
    cv::computeCorrespondEpilines(pair.get_image2_good_matches(), 2, pair.get_F(), epilines2);

    // Draw the epipolar lines on images
    for (size_t i = 0; i < epilines1.size(); i++) {
        cv::Scalar color(rng(256), rng(256), rng(256));
        // Draw line on img1
        cv::line(image1, cv::Point(0, -epilines1[i][2] / epilines1[i][1]),
                 cv::Point(image1.cols, -(epilines1[i][2] + epilines1[i][0] * image1.cols) / epilines1[i][1]),
                 color);

        cv::circle(image1, pair.get_image2_good_matches()[i], 3, cv::Scalar(0, 0, 255), -1);  // Blue points in img2

        // Draw line on img2
        cv::line(image2, cv::Point(0, -epilines2[i][2] / epilines2[i][1]),
                 cv::Point(image2.cols, -(epilines2[i][2] + epilines2[i][0] * image2.cols) / epilines2[i][1]),
                 color);

        cv::circle(image2, pair.get_image1_good_matches()[i], 3, cv::Scalar(0, 0, 255), -1);  // Green points in img1
    }

    cv::hconcat(image1, image2, epipolar_lines_image);
    return epipolar_lines_image;
}