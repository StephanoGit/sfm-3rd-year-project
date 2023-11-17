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
    cv::Mat R0, t0;

    std::vector<ImagePair> frames;
    std::vector<cv::Mat> P_mats;

    // future pointcloud variable

public:
    SfmReconstruction(std::vector<ImagePair> frames);
    ~SfmReconstruction();

    cv::Mat get_K();
    std::vector<double> get_distortion();
    void triangulation();

    void set_P_mats(std::vector<cv::Mat> P_mats);
    void append_P_mat(cv::Mat P);
    std::vector<cv::Mat> get_P_mats();
};

#endif