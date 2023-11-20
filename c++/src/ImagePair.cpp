
#include "../include/ImagePair.h"

#include "../include/ImageView.h"
#include "../include/drawUtil.h"
#include "../include/util.h"

ImagePair::ImagePair(ImageView image1, ImageView image2) {
    this->image1 = image1;
    this->image2 = image2;
}

ImagePair::ImagePair(){};
ImagePair::~ImagePair(){};

void ImagePair::apply_lowes_ratio(std::vector<std::vector<cv::DMatch>> knn_matches) {
    for (int i = 0; i < knn_matches.size(); i++) {
        if (knn_matches[i][0].distance < 0.75 * knn_matches[i][1].distance) {
            this->good_matches.push_back(knn_matches[i][0]);
            this->image1_good_matches.push_back(this->image1.get_keypoints()[knn_matches[i][0].queryIdx].pt);
            this->image2_good_matches.push_back(this->image2.get_keypoints()[knn_matches[i][0].trainIdx].pt);
        }
    }
}

void ImagePair::match_descriptors(FeatureMatchingType type) {
    switch (type) {
        case BF: {
            std::cout << "Matching images using Brute Force..." << std::endl;
            std::vector<std::vector<cv::DMatch>> knn_matches;
            if (this->image1.get_type() == ORB) {
                cv::BFMatcher bf(cv::NORM_HAMMING);
                bf.knnMatch(this->image1.get_descriptors(), this->image2.get_descriptors(), knn_matches, 2);
                apply_lowes_ratio(knn_matches);
            } else {
                cv::BFMatcher bf(cv::NORM_L1);
                bf.knnMatch(this->image1.get_descriptors(), this->image2.get_descriptors(), knn_matches, 2);
                apply_lowes_ratio(knn_matches);
            }
            break;
        }
        case FLANN: {
            std::cout << "Matching images using FLANN..." << std::endl;
            std::vector<std::vector<cv::DMatch>> knn_matches;
            if (this->image1.get_type() == ORB) {
                cv::FlannBasedMatcher flann(new cv::flann::LshIndexParams(6, 12, 1));
                flann.knnMatch(this->image1.get_descriptors(), this->image2.get_descriptors(), knn_matches, 2);
                apply_lowes_ratio(knn_matches);
            } else {
                cv::FlannBasedMatcher flann(new cv::flann::KDTreeIndexParams(5));
                flann.knnMatch(this->image1.get_descriptors(), this->image2.get_descriptors(), knn_matches, 2);
                apply_lowes_ratio(knn_matches);
            }
            break;
        }
        default: {
            std::cout << "Please provide a valid feature matching type (BF or FLANN)" << std::endl;
            break;
        }
    }
}

void ImagePair::remove_outliers(cv::Mat mask) {
    std::vector<cv::DMatch> inliers;
    std::vector<cv::Point2f> inliers1, inliers2;
    for (size_t i = 0; i < mask.rows; i++) {
        if (mask.at<uchar>(i) != 0) {
            inliers.push_back(this->good_matches[i]);
            inliers1.push_back(this->image1_good_matches[i]);
            inliers2.push_back(this->image2_good_matches[i]);
        }
    }

    std::cout << "No. Matches: " << this->good_matches.size() << std::endl
              << "No. Inliers: " << inliers.size() << std::endl
              << std::endl;

    this->good_matches = inliers;
    this->image1_good_matches = inliers1;
    this->image2_good_matches = inliers2;
}

void ImagePair::compute_F() {
    if (this->image1_good_matches.size() == 0 || this->image1_good_matches.size() == 0) {
        std::cout << "Error: Please compute matches between images" << std::endl;
        return;
    }

    cv::Mat mask(this->image1_good_matches.size(), 1, CV_8U);
    this->F = cv::findFundamentalMat(this->image1_good_matches, this->image2_good_matches,
                                     cv::FM_RANSAC, 1.0, 0.999, mask);
    remove_outliers(mask);
}

void ImagePair::compute_E(cv::Mat K) {
    if (this->image1_good_matches.size() == 0 || this->image1_good_matches.size() == 0) {
        std::cout << "Error: Please compute matches between images" << std::endl;
        return;
    }

    cv::Mat mask;
    this->E = cv::findEssentialMat(this->image1_good_matches, this->image2_good_matches,
                                   K, cv::RANSAC, 0.999, 1.0, mask);
    remove_outliers(mask);
}

void ImagePair::compute_Rt(cv::Mat K) {
    cv::recoverPose(this->E, this->image1_good_matches, this->image2_good_matches, K, this->R, this->t, cv::noArray());
}

std::vector<cv::Point3f> ImagePair::triangulate(cv::Mat K, std::vector<double> d) {
    cv::Mat R0 = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat t0 = cv::Mat::zeros(3, 1, CV_64F);

    cv::Mat P1(3, 4, CV_64F);
    cv::hconcat(R0, t0, P1);
    P1 = K * P1;

    cv::Mat P2(3, 4, CV_64F);
    cv::hconcat(this->R, this->t, P2);
    P2 = K * P2;

    // NORMALIZE IMAGE COORDINATE TO CAMERA COORDINATE (pixels --> metric)
    // std::cout << "Normalizing points..." << std::endl;
    // std::vector<cv::Point2f> norm_points_image1, norm_points_image2;
    // cv::undistortPoints(this->image1_good_matches, norm_points_image1, K, d);
    // cv::undistortPoints(this->image2_good_matches, norm_points_image2, K, d);

    cv::Mat points_4d;
    cv::triangulatePoints(P1, P2, this->image1_good_matches, this->image2_good_matches, points_4d);
    // cv::triangulatePoints(P1, P2, norm_points_image1, norm_points_image2, points_4d);

    cv::convertPointsFromHomogeneous(points_4d.t(), this->points_3d);

    // extract kps and desc used in the reconstruction
    this->extract_kps_from_matches();
    this->extract_desc_from_matches();

    return this->points_3d;
}

void ImagePair::extract_desc_from_matches() {
    cv::Mat desc1, desc2;
    for (const auto &match : this->good_matches) {
        desc1.push_back(this->image1.get_descriptors().row(match.queryIdx));
        desc2.push_back(this->image2.get_descriptors().row(match.trainIdx));
    }
    this->image1_good_desc = desc1;
    this->image2_good_desc = desc2;
}

void ImagePair::extract_kps_from_matches() {
    std::vector<cv::KeyPoint> kp1, kp2;
    for (auto &match : this->good_matches) {
        kp1.push_back(this->image1.get_keypoints()[match.queryIdx]);
        kp2.push_back(this->image2.get_keypoints()[match.trainIdx]);
    }
    this->image1_good_kps = kp1;
    this->image2_good_kps = kp2;
}

std::vector<cv::Point3f> ImagePair::init_reconstruction(FeatureDetectionType detection_type,
                                                        FeatureMatchingType matching_type,
                                                        cv::Mat K, std::vector<double> d) {
    this->match_descriptors(matching_type);
    this->compute_F();
    this->compute_E(K);
    this->compute_Rt(K);
    this->extract_kps_from_matches();

    return this->triangulate(K, d);
}

void ImagePair::set_image1(ImageView image1) {
    this->image1 = image1;
}

ImageView ImagePair::get_image1() {
    return this->image1;
}

void ImagePair::set_image2(ImageView image2) {
    this->image2 = image2;
}

ImageView ImagePair::get_image2() {
    return this->image2;
}

std::vector<cv::Point2f> ImagePair::get_image1_good_matches() {
    return this->image1_good_matches;
}

void ImagePair::set_image1_good_matches(std::vector<cv::Point2f> image1_good_matches) {
    this->image1_good_matches = image1_good_matches;
}

std::vector<cv::Point2f> ImagePair::get_image2_good_matches() {
    return this->image2_good_matches;
}

void ImagePair::set_image2_good_matches(std::vector<cv::Point2f> image2_good_matches) {
    this->image2_good_matches = image2_good_matches;
}

void ImagePair::set_good_matches(std::vector<cv::DMatch> good_matches) {
    this->good_matches = good_matches;
}

std::vector<cv::DMatch> ImagePair::get_good_matches() {
    return this->good_matches;
}

void ImagePair::set_matching_type(FeatureMatchingType type) {
    this->type = type;
}

FeatureMatchingType ImagePair::get_matching_type() {
    return this->type;
}

std::vector<cv::KeyPoint> ImagePair::get_image1_good_kps() {
    return this->image1_good_kps;
}

void ImagePair::set_image1_good_kps(std::vector<cv::KeyPoint> image1_good_kps) {
    this->image1_good_kps = image1_good_kps;
}

std::vector<cv::KeyPoint> ImagePair::get_image2_good_kps() {
    return this->image2_good_kps;
}

void ImagePair::set_image2_good_kps(std::vector<cv::KeyPoint> image2_good_kps) {
    this->image2_good_kps = image2_good_kps;
}

void ImagePair::set_F(cv::Mat F) {
    this->F = F;
}

cv::Mat ImagePair::get_F() {
    return this->F;
}

void ImagePair::set_E(cv::Mat E) {
    this->E = E;
}

cv::Mat ImagePair::get_E() {
    return this->E;
}

void ImagePair::set_R(cv::Mat R) {
    this->R = R;
}

cv::Mat ImagePair::get_R() {
    return this->R;
}

void ImagePair::set_t(cv::Mat t) {
    this->t = t;
}

cv::Mat ImagePair::get_t() {
    return this->t;
}

void ImagePair::set_points_3d(std::vector<cv::Point3f> points_3d) {
    this->points_3d = points_3d;
}

std::vector<cv::Point3f> ImagePair::get_points_3d() {
    return this->points_3d;
}

cv::Mat ImagePair::get_image1_good_desc() {
    return this->image1_good_desc;
}
void ImagePair::set_image1_good_desc(cv::Mat image1_good_desc) {
    this->image1_good_desc = image1_good_desc;
}

cv::Mat ImagePair::get_image2_good_desc() {
    return this->image2_good_desc;
}

void ImagePair::set_image2_good_desc(cv::Mat image2_good_desc) {
    this->image2_good_desc = image2_good_desc;
}