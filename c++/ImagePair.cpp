#include "ImagePair.h"

ImagePair::ImagePair(ImageView img1, ImageView img2) {
    this->img1 = img1;
    this->img2 = img2;
}

ImagePair::ImagePair(){};

ImagePair::~ImagePair(){}

void ImagePair::match_descriptors(FeatureMatchingType type){
    switch(type){
        case BF:
        {
            std::cout << "Matching images using Brute Force..." << std::endl;
            std::vector<std::vector<cv::DMatch>> knn_matches;

            // NORM_L1 for SIFT, FAST, SURF
            if (this->img1.get_type() == ORB){
                cv::BFMatcher bf(cv::NORM_HAMMING);
                bf.knnMatch(this->img1.get_descriptors(), this->img2.get_descriptors(), knn_matches, 2);

                for (int i = 0; i < knn_matches.size(); i++) {
                    if (knn_matches[i][0].distance < 0.75 * knn_matches[i][1].distance) {
                        this->good_matches.push_back(knn_matches[i][0]);
                        this->img1_good_matches.push_back(this->img1.get_keypoints()[knn_matches[i][0].queryIdx].pt);
                        this->img2_good_matches.push_back(this->img2.get_keypoints()[knn_matches[i][0].trainIdx].pt);
                    }
                }
            // For binary string based like ORB, BRISK and BRIEF, use NORM_HAMMING
            } else {
                cv::BFMatcher bf(cv::NORM_L1);
                bf.knnMatch(this->img1.get_descriptors(), this->img2.get_descriptors(), knn_matches, 2);

                for (int i = 0; i < knn_matches.size(); i++) {
                    if (knn_matches[i][0].distance < 0.75 * knn_matches[i][1].distance) {
                        this->good_matches.push_back(knn_matches[i][0]);
                        this->img1_good_matches.push_back(this->img1.get_keypoints()[knn_matches[i][0].queryIdx].pt);
                        this->img2_good_matches.push_back(this->img2.get_keypoints()[knn_matches[i][0].trainIdx].pt);
                    }
                }
            }
            break;
        }
        case FLANN:
        {
            std::cout << "Matching images using FLANN..." << std::endl;
            // ORB
            if (this->img1.get_type() == ORB){
                cv::FlannBasedMatcher flann(new cv::flann::LshIndexParams(6,12,1));
                std::vector<std::vector<cv::DMatch>> knn_matches;

                flann.knnMatch(this->img1.get_descriptors(), this->img2.get_descriptors(), knn_matches, 2);
                for (size_t i = 0; i < knn_matches.size(); i++) {
                    if (knn_matches[i][0].distance < 0.7 * knn_matches[i][1].distance) {
                        this->good_matches.push_back(knn_matches[i][0]);
                        this->img1_good_matches.push_back(this->img1.get_keypoints()[knn_matches[i][0].queryIdx].pt);
                        this->img2_good_matches.push_back(this->img2.get_keypoints()[knn_matches[i][0].trainIdx].pt);
                    }
                }
            // SIFT, FAST, SURF
            } else {
                cv::FlannBasedMatcher flann(new cv::flann::KDTreeIndexParams(5));
                std::vector<std::vector<cv::DMatch>> knn_matches;

                flann.knnMatch(this->img1.get_descriptors(), this->img2.get_descriptors(), knn_matches, 2);
                for (size_t i = 0; i < knn_matches.size(); i++) {
                    if (knn_matches[i][0].distance < 0.7 * knn_matches[i][1].distance) {
                        this->good_matches.push_back(knn_matches[i][0]);
                        this->img1_good_matches.push_back(this->img1.get_keypoints()[knn_matches[i][0].queryIdx].pt);
                        this->img2_good_matches.push_back(this->img2.get_keypoints()[knn_matches[i][0].trainIdx].pt);
                    }
                }
            }
            break;
        }
        default:
        {
            std::cout << "Please provide a valid feature matching type (BF or FLANN)" << std::endl;
            break;
        }
    }
}

void ImagePair::draw_matches(){
    cv::drawMatches(this->img1.get_image(),
                    this->img1.get_keypoints(),
                    this->img2.get_image(),
                    this->img2.get_keypoints(),
                    this->good_matches,
                    this->matches_image, 
                    cv::Scalar::all(-1), cv::Scalar::all(-1), 
                    std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
}

cv::Mat ImagePair::get_matches_image(){
    return this->matches_image;
}

void ImagePair::set_image1(ImageView image1){
    this->img1 = image1;
}
ImageView ImagePair::get_image1(){
    return this->img1;
}

void ImagePair::set_image2(ImageView image2){
    this->img2 = image2;
}

ImageView ImagePair::get_image2(){
    return this->img2;
}

// void ImagePair::set_matches(std::vector<cv::DMatch> matches){
//     this->matches = matches;
// }
// std::vector<cv::DMatch> ImagePair::get_matches(){
//     return this->matches;
// }

std::vector<cv::Point2f> ImagePair::get_img1_good_matches(){
    return this->img1_good_matches;
}
std::vector<cv::Point2f> ImagePair::get_img2_good_matches(){
    return this->img2_good_matches;
}

void ImagePair::set_good_matches(std::vector<cv::DMatch> good_matches){
    this->good_matches = good_matches;
}

std::vector<cv::DMatch> ImagePair::get_good_matches(){
    return this->good_matches;
}

void ImagePair::set_matching_type(FeatureMatchingType type){
    this->type = type;
}

FeatureMatchingType ImagePair::get_matching_type(){
    return this->type;
}

