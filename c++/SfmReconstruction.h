#ifndef __SFM_RECONSTRUCTION
#define __SFM_RECONSTRUCTION

#include <stdio.h>
#include <string>
#include <vector>
#include <fstream>

#include <opencv2/opencv.hpp>

#include "ImageView.h"
#include "ImagePair.h"

class SfmReconstruction
{
private:
    cv::Mat K;
    std::vector<double> distortion;

    std::vector<ImageView> views;
    std::vector<cv::Point3f> point_cloud;

    FeatureDetectionType detection_type;
    FeatureMatchingType matching_type;

    ImageView last_image;
    std::vector<cv::KeyPoint> last_image_kps;
    cv::Mat last_image_desc;

    // future pointcloud variable

public:
    SfmReconstruction(std::vector<ImageView> views,
                      FeatureDetectionType detection_type,
                      FeatureMatchingType matching_type);

    void add_new_view(ImageView new_image, ImageView last_image);

    std::vector<cv::Point3f> get_point_cloud();
    void set_point_cloud(std::vector<cv::Point3f> point_cloud);

    cv::Mat get_K();
    std::vector<double> get_distortion();

    ~SfmReconstruction();
};

#endif