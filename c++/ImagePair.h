#ifndef __IMAGE_PAIR
#define __IMAGE_PAIR

#include <stdio.h>
#include <string>
#include <iostream>
#include <filesystem>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

#include "ImageView.h"


enum FeatureMatchingType {
    // NONE,
    BF,
    FLANN
};


class ImagePair {
    private:
        /* data */
        ImageView img1;
        ImageView img2;
        cv::Mat matches_image;

        FeatureMatchingType type;

        // std::vector<cv::DMatch> matches;
        std::vector<cv::DMatch> good_matches;

    public:
        ImagePair();
        ImagePair(ImageView img1, ImageView img2);
        ~ImagePair();

        /**
         * Match the descriptors of two images within a pair
         * @note this should check for the type of matching, BF or FLANN
         */
        void match_descriptors(FeatureMatchingType type);

        // void set_matches(std::vector<cv::DMatch> matches);
        // std::vector<cv::DMatch> get_matches();

        void set_good_matches(std::vector<cv::DMatch> good_matches);
        std::vector<cv::DMatch> get_good_matches();

        void draw_matches();
        cv::Mat get_matches_image();

        void set_image1(ImageView image1);
        ImageView get_image1();

        void set_image2(ImageView image2);
        ImageView get_image2();

        void set_matching_type(FeatureMatchingType type);
        FeatureMatchingType get_matching_type();
};

#endif
