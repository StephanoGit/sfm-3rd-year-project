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
    CameraCalibration calibrate(CHECKERBOARD_DIR, false);
    cv::Mat K = calibrate.get_K();
    cv::Vec<float, 5> d = calibrate.get_d();
    float error = calibrate.get_error();

    std::cout << K << std::endl;
    std::cout << d << std::endl;
    std::cout << error << std::endl;

    // std::vector<ImageView> images;
    // std::vector<ImagePair> pairs;

    // // images = extract_frames_from_video(VID_DIR, 50);
    // images = load_images(IMG_DIR);

    // // compute descriptors and keypoints for each image
    // for (int i = 0; i < images.size(); i++)
    // {
    //     images[i].compute_kps_des(FeatureDetectionType::SIFT);
    //     std::cout << "Feature detection done for: " << images[i].get_name() << std::endl;
    // }

    // // create image pairs (i, i+1)
    // for (int i = 0; i < images.size() - 1; i++)
    // {
    //     ImagePair pair(images[i], images[i + 1]);
    //     pairs.push_back(pair);
    // }

    // SfmReconstruction reconstruction(pairs);
    // // reconstruction.triangulation();

    // cv::Mat K = reconstruction.get_K();
    // std::vector<double> d = reconstruction.get_distortion();
    // std::cout << K << std::endl;

    // for (int i = 0; i < pairs.size(); i++)
    // {
    //     pairs[i].match_descriptors(FeatureMatchingType::FLANN);
    //     // cv::Mat matches_image = draw_matches(pairs[i]);
    //     // cv::imshow("Matches", matches_image);
    //     pairs[i].compute_F();
    //     // cv::Mat matches_image2 = draw_matches(pairs[i]);
    //     // cv::imshow("Matches2", matches_image2);
    //     pairs[i].compute_E(K);
    //     // cv::Mat matches_image3 = draw_matches(pairs[i]);
    //     // cv::imshow("Matches3", matches_image3);

    //     pairs[i].compute_Rt(K);
    //     pairs[i].triangulate(K, i);

    //     // cv::waitKey(0);
    // }

    return 0;
}