#ifndef __IMAGE_VIEW
#define __IMAGE_VIEW

#include <stdio.h>
#include <opencv2/opencv.hpp>

enum FeatureDetectionType
{
    SIFT,
    SURF,
    FAST,
    ORB
};

class ImageView
{
private:
    cv::Mat image;
    std::string name;
    FeatureDetectionType type;

    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

public:
    ImageView();
    ImageView(cv::Mat image, std::string name,
              FeatureDetectionType type,
              std::vector<cv::KeyPoint> keypoints,
              cv::Mat descriptors);
    ~ImageView();

    void compute_kps_des(FeatureDetectionType type);

    void set_image(cv::Mat image);
    cv::Mat get_image();

    void set_name(std::string name);
    std::string get_name();

    void set_type(FeatureDetectionType type);
    FeatureDetectionType get_type();

    void set_keypoints(std::vector<cv::KeyPoint> keypoints);
    std::vector<cv::KeyPoint> get_keypoints();

    void set_descriptors(cv::Mat descriptors);
    cv::Mat get_descriptors();
};

#endif