#include "../include/FeatureUtil.h"
#include "../include/util.h"
#include <opencv2/core/types.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

FeatureUtil::FeatureUtil(FeatureExtractionType extract_type,
                         FeatureMatchingType match_type) {
  this->extract_type = extract_type;
  this->match_type = match_type;
}

FeatureUtil::~FeatureUtil() {}

Features FeatureUtil::extract_features(cv::Mat &image) {
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
  case FAST: {
    std::cout << "Applying FAST ..." << std::endl;
    cv::Ptr<cv::FastFeatureDetector> detector =
        cv::FastFeatureDetector::create();
    detector->detectAndCompute(image, cv::noArray(), features.key_points,
                               features.descriptors);
    break;
  }
  case ORB: {
    std::cout << "Applying ORB ..." << std::endl;
    cv::Ptr<cv::ORB> detector = cv::ORB::create();
    detector->detectAndCompute(image, cv::noArray(), features.key_points,
                               features.descriptors);
    break;
  }
  default: {
    std::cout << "Please provide a valid feature detector (SIFT, SURF, FAST or "
                 "ORB) ..."
              << std::endl;
    break;
  }
  }
  return features;
}

std::vector<cv::DMatch>
apply_lowes_ratio(std::vector<std::vector<cv::DMatch>> knn_matches) {
  std::vector<cv::DMatch> good_matches;
  for (int i = 0; i < knn_matches.size(); i++) {
    if (knn_matches[i][0].distance < 0.75 * knn_matches[i][1].distance) {
      good_matches.push_back(knn_matches[i][0]);
    }
  }
}

std::vector<cv::DMatch> FeatureUtil::match_features(Features &left,
                                                    Features &right) {
  std::vector<std::vector<cv::DMatch>> knn_matches;
  switch (this->match_type) {
  case BF: {
    std::cout << "Matching images using Brute Force..." << std::endl;
    if (this->extract_type == ORB) {
      std::cout << "Using Norm Hamming..." << std::endl;
      cv::BFMatcher bf(cv::NORM_HAMMING);
      bf.knnMatch(left.descriptors, right.descriptors, knn_matches, 2);
    } else {
      std::cout << "Using Norm L1..." << std::endl;
      cv::BFMatcher bf(cv::NORM_L1);
      bf.knnMatch(left.descriptors, right.descriptors, knn_matches, 2);
    }
    break;
  }
  case FLANN: {
    std::cout << "Matching images using FLANN..." << std::endl;
    std::vector<std::vector<cv::DMatch>> knn_matches;
    if (this->extract_type == ORB) {
      cv::FlannBasedMatcher flann(new cv::flann::LshIndexParams(6, 12, 1));
      flann.knnMatch(left.descriptors, right.descriptors, knn_matches, 2);
    } else {
      cv::FlannBasedMatcher flann(new cv::flann::KDTreeIndexParams(5));
      flann.knnMatch(left.descriptors, right.descriptors, knn_matches, 2);
    }
    break;
  }
  default: {
    std::cout << "Please provide a valid feature matching type (BF or FLANN)"
              << std::endl;
    break;
  }
  }
  return apply_lowes_ratio(knn_matches);
}
