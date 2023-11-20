#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

#include "ImageView.h"

ImageView::ImageView() {}

ImageView::ImageView(cv::Mat image, std::string name,
                     FeatureDetectionType type,
                     std::vector<cv::KeyPoint> keypoints,
                     cv::Mat descriptors) : image(image), name(name),
                                            type(type), keypoints(keypoints),
                                            descriptors(descriptors) {}

ImageView::~ImageView() {}

void ImageView::compute_kps_des(FeatureDetectionType type)
{
    switch (type)
    {
    case SIFT:
    {
        std::cout << "Applying SIFT ..." << std::endl;
        cv::Ptr<cv::SIFT> detector = cv::SIFT::create();
        this->type = type;
        detector->detectAndCompute(this->image, cv::noArray(), this->keypoints, this->descriptors);
        std::cout << this->keypoints.size() << std::endl;
        std::cout << this->descriptors.size() << std::endl;

        break;
    }
    case SURF:
    {
        std::cout << "Applying SURF ..." << std::endl;
        cv::Ptr<cv::xfeatures2d::SurfFeatureDetector> detector = cv::xfeatures2d::SurfFeatureDetector::create();
        this->type = type;
        detector->detectAndCompute(this->image, cv::noArray(), this->keypoints, this->descriptors);
        break;
    }
    case FAST:
    {
        std::cout << "Applying FAST ..." << std::endl;
        cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create();
        this->type = type;
        detector->detectAndCompute(this->image, cv::noArray(), this->keypoints, this->descriptors);
        break;
    }
    case ORB:
    {
        std::cout << "Applying ORB ..." << std::endl;
        cv::Ptr<cv::ORB> detector = cv::ORB::create();
        this->type = type;
        detector->detectAndCompute(this->image, cv::noArray(), this->keypoints, this->descriptors);
        break;
    }
    default:
        std::cout << "Please provide a valid feature detector (SIFT, SURF, FAST or ORB) ..." << std::endl;
        break;
    }
}

void ImageView::set_image(cv::Mat image)
{
    this->image = image;
}

cv::Mat ImageView::get_image()
{
    return this->image;
}

void ImageView::set_name(std::string name)
{
    this->name = name;
}

std::string ImageView::get_name()
{
    return this->name;
}

void ImageView::set_type(FeatureDetectionType type)
{
    this->type = type;
}

FeatureDetectionType ImageView::get_type()
{
    return this->type;
}

void ImageView::set_keypoints(std::vector<cv::KeyPoint> keypoints)
{
    this->keypoints = keypoints;
}

std::vector<cv::KeyPoint> ImageView::get_keypoints()
{
    return this->keypoints;
}

void ImageView::set_descriptors(cv::Mat descriptors)
{
    this->descriptors = descriptors;
}

cv::Mat ImageView::get_descriptors()
{
    return this->descriptors;
}