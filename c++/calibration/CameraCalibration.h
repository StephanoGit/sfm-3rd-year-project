#ifndef __CAMERA_CALIBRATION
#define __CAMERA_CALIBRATION

#include <stdio.h>

#include <opencv2/opencv.hpp>

class CameraCalibration {
   private:
    cv::Mat K;
    cv::Vec<float, 5> d;
    float error;

    std::vector<int> checkerboard_size;
    int square_size;

   public:
    CameraCalibration(std::string directory, bool show_images);
    ~CameraCalibration();

    void calibrateCamera(std::string directory, std::vector<int> checkerboard_size, int square_size, bool show_images);

    cv::Mat get_K();
    cv::Vec<float, 5> get_d();
    float get_error();
};

#endif