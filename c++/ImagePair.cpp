#include "ImagePair.h"

ImagePair::ImagePair(ImageView image1, ImageView image2)
{
    this->image1 = image1;
    this->image2 = image2;
}

ImagePair::ImagePair(){};
ImagePair::~ImagePair(){};

void ImagePair::match_descriptors(FeatureMatchingType type)
{
    switch (type)
    {
    case BF:
    {
        std::cout << "Matching images using Brute Force..." << std::endl;
        std::vector<std::vector<cv::DMatch>> knn_matches;
        if (this->image1.get_type() == ORB)
        {
            cv::BFMatcher bf(cv::NORM_HAMMING);
            bf.knnMatch(this->image1.get_descriptors(), this->image2.get_descriptors(), knn_matches, 2);

            for (int i = 0; i < knn_matches.size(); i++)
            {
                if (knn_matches[i][0].distance < 0.75 * knn_matches[i][1].distance)
                {
                    this->good_matches.push_back(knn_matches[i][0]);
                    this->image1_good_matches.push_back(this->image1.get_keypoints()[knn_matches[i][0].queryIdx].pt);
                    this->image2_good_matches.push_back(this->image2.get_keypoints()[knn_matches[i][0].trainIdx].pt);
                }
            }
        }
        else
        {
            cv::BFMatcher bf(cv::NORM_L1);
            bf.knnMatch(this->image1.get_descriptors(), this->image2.get_descriptors(), knn_matches, 2);

            for (int i = 0; i < knn_matches.size(); i++)
            {
                if (knn_matches[i][0].distance < 0.75 * knn_matches[i][1].distance)
                {
                    this->good_matches.push_back(knn_matches[i][0]);
                    this->image1_good_matches.push_back(this->image1.get_keypoints()[knn_matches[i][0].queryIdx].pt);
                    this->image2_good_matches.push_back(this->image2.get_keypoints()[knn_matches[i][0].trainIdx].pt);
                }
            }
        }
        break;
    }
    case FLANN:
    {
        std::cout << "Matching images using FLANN..." << std::endl;
        std::vector<std::vector<cv::DMatch>> knn_matches;
        if (this->image1.get_type() == ORB)
        {
            cv::FlannBasedMatcher flann(new cv::flann::LshIndexParams(6, 12, 1));
            flann.knnMatch(this->image1.get_descriptors(), this->image2.get_descriptors(), knn_matches, 2);
            for (size_t i = 0; i < knn_matches.size(); i++)
            {
                if (knn_matches[i][0].distance < 0.7 * knn_matches[i][1].distance)
                {
                    this->good_matches.push_back(knn_matches[i][0]);
                    this->image1_good_matches.push_back(this->image1.get_keypoints()[knn_matches[i][0].queryIdx].pt);
                    this->image2_good_matches.push_back(this->image2.get_keypoints()[knn_matches[i][0].trainIdx].pt);
                }
            }
        }
        else
        {
            cv::FlannBasedMatcher flann(new cv::flann::KDTreeIndexParams(5));
            flann.knnMatch(this->image1.get_descriptors(), this->image2.get_descriptors(), knn_matches, 2);
            for (size_t i = 0; i < knn_matches.size(); i++)
            {
                if (knn_matches[i][0].distance < 0.7 * knn_matches[i][1].distance)
                {
                    this->good_matches.push_back(knn_matches[i][0]);
                    this->image1_good_matches.push_back(this->image1.get_keypoints()[knn_matches[i][0].queryIdx].pt);
                    this->image2_good_matches.push_back(this->image2.get_keypoints()[knn_matches[i][0].trainIdx].pt);
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

void ImagePair::compute_F()
{
    if (this->image1_good_matches.size() == 0)
    {
        std::cout << "Error: Please compute matches between images" << std::endl;
        return;
    }

    cv::Mat mask(this->image1_good_matches.size(), 1, CV_8U);
    this->F = cv::findFundamentalMat(this->image1_good_matches, this->image2_good_matches,
                                     cv::FM_RANSAC, 1.0, 0.999, mask);

    std::cout << "vvv F vvv\n"
              << this->F << std::endl
              << this->image1.get_name() + " and " + this->image2.get_name() << std::endl
              << std::endl;

    std::vector<cv::DMatch> inliers;
    std::vector<cv::Point2f> inliers1, inliers2;
    for (size_t i = 0; i < mask.rows; i++)
    {
        if (mask.at<uchar>(i) != 0)
        {
            inliers.push_back(this->good_matches[i]);
            inliers1.push_back(this->image1_good_matches[i]);
            inliers1.push_back(this->image1_good_matches[i]);
        }
    }

    std::cout << "No. Matches: " << this->good_matches.size() << std::endl
              << "No. Inliers: " << inliers.size() << std::endl
              << std::endl;

    this->good_matches = inliers;
    this->image1_good_matches = inliers1;
    this->image2_good_matches = inliers2;
}

void ImagePair::set_image1(ImageView image1)
{
    this->image1 = image1;
}

ImageView ImagePair::get_image1()
{
    return this->image1;
}

void ImagePair::set_image2(ImageView image2)
{
    this->image2 = image2;
}

ImageView ImagePair::get_image2()
{
    return this->image2;
}

std::vector<cv::Point2f> ImagePair::get_image1_good_matches()
{
    return this->image1_good_matches;
}

void ImagePair::set_image1_good_matches(std::vector<cv::Point2f> image1_good_matches)
{
    this->image1_good_matches = image1_good_matches;
}

std::vector<cv::Point2f> ImagePair::get_image2_good_matches()
{
    return this->image2_good_matches;
}

void ImagePair::set_image2_good_matches(std::vector<cv::Point2f> image2_good_matches)
{
    this->image2_good_matches = image2_good_matches;
}

void ImagePair::set_good_matches(std::vector<cv::DMatch> good_matches)
{
    this->good_matches = good_matches;
}

std::vector<cv::DMatch> ImagePair::get_good_matches()
{
    return this->good_matches;
}

void ImagePair::set_matching_type(FeatureMatchingType type)
{
    this->type = type;
}

FeatureMatchingType ImagePair::get_matching_type()
{
    return this->type;
}

std::vector<cv::KeyPoint> ImagePair::get_image1_good_kps()
{
    return this->image1_good_kps;
}

void ImagePair::set_image1_good_kps(std::vector<cv::KeyPoint> image1_good_kps)
{
    this->image1_good_kps = image1_good_kps;
}

std::vector<cv::KeyPoint> ImagePair::get_image2_good_kps()
{
    return this->image2_good_kps;
}

void ImagePair::set_image2_good_kps(std::vector<cv::KeyPoint> image2_good_kps)
{
    this->image2_good_kps = image2_good_kps;
}

void ImagePair::set_R(cv::Mat R)
{
    this->R = R;
}

cv::Mat ImagePair::get_R()
{
    return this->R;
}

void ImagePair::set_t(cv::Mat t)
{
    this->t = t;
}

cv::Mat ImagePair::get_t()
{
    return this->t;
}

void ImagePair::set_points_3d(cv::Mat points_3d)
{
    this->points_3d = points_3d;
}

cv::Mat ImagePair::get_points_3d()
{
    return this->points_3d;
}