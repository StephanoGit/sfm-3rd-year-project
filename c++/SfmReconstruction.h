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
    cv::Mat R0, t0;
    std::vector<ImagePair> frames;

    // future pointcloud variable

public:
    SfmReconstruction(std::vector<ImagePair> frames);
    ~SfmReconstruction();

    cv::Mat get_K();
};

#endif