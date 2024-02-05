#ifndef __PMVS2
#define __PMVS2
#include "SfmStructures.h"
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>

class PMVS2Reconstruction {
public:
    void dense_reconstruction(std::vector<cv::Mat> &images,
                              std::vector<std::string> &images_paths,
                              std::vector<cv::Matx34f> &camera_poses,
                              Intrinsics &intrinsics);
};

#endif
