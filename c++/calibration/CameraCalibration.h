#ifndef __CAMERA_CALIBRATION
#define __CAMERA_CALIBRATION

#include <stdio.h>

#include <opencv2/opencv.hpp>

class CameraCalibration {
  private:
    cv::Mat K;
    cv::Mat d;
    float error;

    std::vector<int> checkerboard_size;
    int square_size;

  public:
    CameraCalibration(std::string directory, bool show_images, int resize_val);
    ~CameraCalibration();

    void calibrateCamera(std::string directory, std::vector<int> checkerboard_size, int square_size, bool show_images, int resize_val);

    cv::Mat get_K();
    cv::Mat get_d();
    float get_error();
};

#endif
