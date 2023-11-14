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

enum FeatureMatchingType
{
    BF,
    FLANN
};

class ImagePair
{
private:
    ImageView image1;
    ImageView image2;

    FeatureMatchingType type;

    std::vector<cv::DMatch> good_matches;

    std::vector<cv::Point2f> image1_good_matches;
    std::vector<cv::Point2f> image2_good_matches;

    std::vector<cv::KeyPoint> image1_good_kps;
    std::vector<cv::KeyPoint> image2_good_kps;

    cv::Mat F, E;
    cv::Mat R, t;
    cv::Mat points_3d;

public:
    ImagePair();
    ImagePair(ImageView image1, ImageView image2);
    ~ImagePair();

    void match_descriptors(FeatureMatchingType type);
    void compute_F();
    void compute_E(cv::Mat K);
    void compute_Rt(cv::Mat K);
    void triangulate(cv::Mat K, int index);

    void set_good_matches(std::vector<cv::DMatch> good_matches);
    std::vector<cv::DMatch> get_good_matches();

    void set_image1(ImageView image1);
    ImageView get_image1();

    void set_image2(ImageView image2);
    ImageView get_image2();

    void set_matching_type(FeatureMatchingType type);
    FeatureMatchingType get_matching_type();

    std::vector<cv::Point2f> get_image1_good_matches();
    void set_image1_good_matches(std::vector<cv::Point2f> image1_good_matches);

    std::vector<cv::Point2f> get_image2_good_matches();
    void set_image2_good_matches(std::vector<cv::Point2f> image2_good_matches);

    std::vector<cv::KeyPoint> get_image1_good_kps();
    void set_image1_good_kps(std::vector<cv::KeyPoint> image1_good_kps);

    std::vector<cv::KeyPoint> get_image2_good_kps();
    void set_image2_good_kps(std::vector<cv::KeyPoint> image2_good_kps);

    void set_E(cv::Mat E);
    cv::Mat get_E();

    void set_F(cv::Mat F);
    cv::Mat get_F();

    void set_R(cv::Mat R);
    cv::Mat get_R();

    void set_t(cv::Mat t);
    cv::Mat get_t();

    void set_points_3d(cv::Mat points_3d);
    cv::Mat get_points_3d();
};

#endif
