#include <stdio.h>
#include <string>
#include <iostream>
#include <filesystem>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include <opencv2/sfm/triangulation.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "ImageView.h"
#include "ImagePair.h"
#include "util.h"
#include "drawUtil.h"
#include "SfmReconstruction.h"
#include "./camera_calibration/CameraCalibration.h"

#define CHECKERBOARD_DIR "../camera_calibration/images"

#define IMG_DIR "../images/fountain-P11-rev"
#define VID_DIR "../videos/test1.MOV"

int main(int argc, char **argv)
{

    std::vector<ImageView> images;
    // std::vector<ImagePair> pairs;

    // images = extract_frames_from_video(VID_DIR, 50);
    images = load_images_as_object(IMG_DIR);

    SfmReconstruction reconstruction(images, FeatureDetectionType::SIFT, FeatureMatchingType::FLANN);

    return 0;
}