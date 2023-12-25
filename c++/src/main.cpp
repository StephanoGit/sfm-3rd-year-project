#include <stdio.h>

#include <opencv2/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/sfm/triangulation.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "../calibration/CameraCalibration.h"
#include "../include/IOUtil.h"
#include "../include/PlottingUtil.h"
#include "../include/SfmReconstruction.h"

#define CHECKERBOARD_DIR "../calibration/images"
#define IMG_DIR "../images/fountain-P11-rev"
#define VID_DIR "../videos/test1.MOV"

#define PHOTO_WIDTH 3072
#define PHOTO_HEIGHT 2048
#define NEW_PHOTO_WIDTH 640
#define NEW_PHOTO_HEIGHT 480

#define VIDEO_WIDTH 3840
#define VIDEO_HEIGHT 2160
#define NEW_VIDEO_WIDTH 1280
#define NEW_VIDEO_HEIGHT 720

// double new_fx = (2759.48 * NEW_PHOTO_WIDTH) / PHOTO_WIDTH;
// double new_fy = (2764.16 * NEW_PHOTO_HEIGHT) / PHOTO_HEIGHT;
// double new_cx = (1520.69 * NEW_PHOTO_WIDTH) / PHOTO_WIDTH;
// double new_cy = (1006.81 * NEW_PHOTO_HEIGHT) / PHOTO_HEIGHT;

// double new_fx = (3278.68 * NEW_VIDEO_WIDTH) / VIDEO_WIDTH;
// double new_fy = (3278.68 * NEW_VIDEO_HEIGHT) / VIDEO_HEIGHT;
// double new_cx = ((VIDEO_WIDTH / 2) * NEW_VIDEO_WIDTH) / VIDEO_WIDTH;
// double new_cy = ((VIDEO_HEIGHT / 2) * NEW_VIDEO_HEIGHT) / VIDEO_HEIGHT;

double new_fx = 2759.48;
double new_fy = 2764.16;
double new_cx = 1520.69;
double new_cy = 1006.81;

double data[9] = {new_fx, 0, new_cx, 0, new_fy, new_cy, 0, 0, 1};

int main(int argc, char **argv) {
  // CameraCalibration(CHECKERBOARD_DIR, false);

  if (argc != 5) {
    std::cout << "Please specify: " << std::endl;
    std::cout << "(1) camera:        iphone or dev" << std::endl;
    std::cout << "(2) input type:    video  or images" << std::endl;
    std::cout << "(3) resize:        true   or false" << std::endl;
    std::cout << "(4) directory:     <../file/path>" << std::endl;
    return 0;
  }

  bool downscale;
  std::string camera(argv[1]);
  std::string input_type(argv[2]);
  std::string resize_val(argv[3]);
  std::string directory(argv[4]);

  if (resize_val == "true") {
    downscale = true;
  } else if (resize_val == "false") {
    downscale = false;
  } else {
    std::cout << "Please provide a valid value for resizing..." << std::endl;
  }

  Intrinsics intrinsics;
  intrinsics.K = cv::Mat(3, 3, CV_64F, data);
  intrinsics.K_inv = intrinsics.K.inv();
  intrinsics.d = cv::Mat::zeros(1, 5, CV_32F);

  SfmReconstruction reconstruction(directory, FeatureExtractionType::SIFT,
                                   FeatureMatchingType::BF, intrinsics);
  reconstruction.run_sfm_reconstruction(downscale);

  return 0;
}
