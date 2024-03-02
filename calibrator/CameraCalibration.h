#ifndef __CAMERA_CALIBRATION
#define __CAMERA_CALIBRATION

#include <opencv2/opencv.hpp>
#include <pugixml.hpp>
#include <stdio.h>

class CameraCalibration {
public:
    cv::Mat K;
    cv::Mat d;
    float error;

    std::vector<int> checkerboard_size;
    int square_size;

    CameraCalibration(std::string directory, bool show_images, int resize_val,
                      std::string file_name);
    ~CameraCalibration();

    void calibrateCamera(std::string directory,
                         std::vector<int> checkerboard_size, int square_size,
                         bool show_images, int resize_val,
                         std::string file_name);

    void export_to_xml(const std::string &file_name, const cv::Mat &K,
                       const std::vector<float> &d_values);
};

#endif
