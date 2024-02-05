#include "../include/FeatureUtil.h"
#include "../include/CommonUtil.h"

#include <opencv2/core/types.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

FeatureUtil::FeatureUtil() {}

FeatureUtil::FeatureUtil(FeatureExtractionType extract_type,
                         FeatureMatchingType match_type) {
    this->extract_type = extract_type;
    this->match_type = match_type;
}

FeatureUtil::~FeatureUtil() {}

Features FeatureUtil::extract_features(const cv::Mat &image) {
    Features features;
    switch (this->extract_type) {
    case SIFT: {
        std::cout << "Applying SIFT ..." << std::endl;
        cv::Ptr<cv::SIFT> detector = cv::SIFT::create();
        detector->detectAndCompute(image, cv::noArray(), features.key_points,
                                   features.descriptors);
        break;
    }
    case SURF: {
        std::cout << "Applying SURF ..." << std::endl;
        cv::Ptr<cv::xfeatures2d::SurfFeatureDetector> detector =
            cv::xfeatures2d::SurfFeatureDetector::create();
        detector->detectAndCompute(image, cv::noArray(), features.key_points,
                                   features.descriptors);
        break;
    }
    case BRISK: {
        std::cout << "Applying BRISK ..." << std::endl;
        cv::Ptr<cv::BRISK> detector = cv::BRISK::create();
        detector->detectAndCompute(image, cv::noArray(), features.key_points,
                                   features.descriptors);
        break;
    }
    case KAZE: {
        std::cout << "Applying KAZE ..." << std::endl;
        cv::Ptr<cv::KAZE> detector = cv::KAZE::create();
        detector->detectAndCompute(image, cv::noArray(), features.key_points,
                                   features.descriptors);
        break;
    }
    case AKAZE: {
        std::cout << "Applying AKAZE ..." << std::endl;
        cv::Ptr<cv::AKAZE> detector = cv::AKAZE::create();
        detector->detectAndCompute(image, cv::noArray(), features.key_points,
                                   features.descriptors);
        break;
    }
    case ORB: {
        std::cout << "Applying ORB ..." << std::endl;
        cv::Ptr<cv::ORB> detector = cv::ORB::create(5000);
        detector->detectAndCompute(image, cv::noArray(), features.key_points,
                                   features.descriptors);
        break;
    }
    default: {
        std::cout
            << "Please provide a valid feature detector (SIFT, SURF, FAST or "
               "ORB) ..."
            << std::endl;
        break;
    }
    }
    keypoints_to_points(features.key_points, features.points);
    return features;
}

std::vector<cv::DMatch>
FeatureUtil::match_features(const Features &features_left,
                            const Features &features_right) {
    std::vector<std::vector<cv::DMatch>> knn_matches;
    switch (this->match_type) {
    case BF: {
        std::cout << "Matching images using Brute Force..." << std::endl;
        if (this->extract_type == ORB) {
            std::cout << "Using Norm Hamming..." << std::endl;
            cv::BFMatcher bf(cv::NORM_HAMMING);
            bf.knnMatch(features_left.descriptors, features_right.descriptors,
                        knn_matches, 2);
        } else {
            std::cout << "Using Norm L1..." << std::endl;
            cv::BFMatcher bf(cv::NORM_L1);
            bf.knnMatch(features_left.descriptors, features_right.descriptors,
                        knn_matches, 2);
        }
        break;
    }
    case FLANN: {
        std::cout << "Matching images using FLANN..." << std::endl;
        if (this->extract_type == ORB) {
            cv::FlannBasedMatcher flann(
                new cv::flann::LshIndexParams(6, 12, 1));
            flann.knnMatch(features_left.descriptors,
                           features_right.descriptors, knn_matches, 2);
        } else {
            cv::FlannBasedMatcher flann(new cv::flann::KDTreeIndexParams(5));
            flann.knnMatch(features_left.descriptors,
                           features_right.descriptors, knn_matches, 2);
        }
        break;
    }
    default: {
        std::cout
            << "Please provide a valid feature matching type (BF or FLANN)"
            << std::endl;
        break;
    }
    }
    return apply_lowes_ratio(knn_matches);
}
