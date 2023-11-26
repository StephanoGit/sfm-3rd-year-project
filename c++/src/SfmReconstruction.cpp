#include "../include/SfmReconstruction.h"

#include <iostream>

#include "../include/ImagePair.h"
#include "../include/ImageView.h"
#include "../include/drawUtil.h"
#include "../include/util.h"
#include <opencv2/opencv.hpp>

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

SfmReconstruction::SfmReconstruction(std::vector<ImageView> views,
                                     FeatureDetectionType detection_type,
                                     FeatureMatchingType matching_type) {
  if (views.size() < 2) {
    std::cout << "Please provide at least two images..." << std::endl;
    return;
  }
  this->K = cv::Mat(3, 3, CV_64F, data);
  this->distortion = {0.0, 0.0, 0.0, 0.0, 0.0};
  this->views = views;
  this->detection_type = detection_type;
  this->matching_type = matching_type;

  cv::Mat image;

  // extract kp and desc of each image
  for (int i = 0; i < views.size(); i++) {
    views[i].compute_kps_des(detection_type);
  }

  // create the initial pair (the first two views)
  ImagePair init_pair(views[4], views[5]);

  // start the initial reconstruction and return the point cloud
  this->point_cloud = init_pair.init_reconstruction(
      detection_type, matching_type, this->K, this->distortion);

  // get info about the last view
  this->last_image = init_pair.get_image2();
  this->last_image_kps = init_pair.get_image2_good_kps();
  this->last_image_desc = init_pair.get_image2_good_desc();
  this->last_P = init_pair.get_P();

  // add the next views (this will be a loop in the future)
  // for now we are testing with a third view
  this->add_new_view(views[6], init_pair.get_image2());
}

void SfmReconstruction::add_new_view(ImageView new_image,
                                     ImageView last_image_raw) {
  // create a new ImageView object representing the last image
  // it should have the filtered descriptors and kps present in the current
  // point cloud
  ImageView last_image(last_image_raw.get_image(), last_image_raw.get_name(),
                       this->detection_type, this->last_image_kps,
                       this->last_image_desc);

  // create a pair between the last image and the new image and match them
  ImagePair pair(last_image, new_image);
  pair.match_descriptors(this->matching_type);
  // filter the outliers
  pair.compute_F();
  pair.compute_E(this->K);
  pair.extract_desc_from_matches();

  // find a way to keep only the 3d points represnted by the good_matches
  // between the two
  std::vector<Point_3D> common_points_3d;
  std::vector<cv::Point3d> object_points;
  std::vector<cv::Point2f> common_points_2d;
  std::set<std::pair<float, float>> test;

  std::cout << "3d_points from cloud: " << object_points.size() << std::endl;
  std::cout << "2d_points from img2: " << common_points_2d.size() << std::endl;
  std::cout << "2d_points in img2: " << pair.get_image1_good_matches().size()
            << std::endl;

  cv::Mat rvec, tvec, R_new;
  cv::solvePnPRansac(object_points, common_points_2d, this->K, this->distortion,
                     rvec, tvec);
  cv::Rodrigues(rvec, R_new);

  cv::Mat P_new(3, 4, CV_64F);
  cv::hconcat(R_new, tvec, P_new);
  P_new = this->K * P_new;

  cv::Mat points_4d;
  cv::triangulatePoints(this->last_P, P_new, pair.get_image1_good_matches(),
                        pair.get_image2_good_matches(), points_4d);

  std::vector<cv::Point3f> points_3d;
  cv::convertPointsFromHomogeneous(points_4d.t(), points_3d);

  export_3d_points_to_txt("../points-3d/norm2.json", points_3d);

  std::cout << rvec << std::endl;
  std::cout << tvec << std::endl;
  std::cout << R_new << std::endl;
}

std::vector<Point_3D> SfmReconstruction::get_point_cloud() {
  return this->point_cloud;
}

void SfmReconstruction::set_point_cloud(std::vector<Point_3D> point_cloud) {
  this->point_cloud = point_cloud;
}

cv::Mat SfmReconstruction::get_K() { return this->K; }

std::vector<double> SfmReconstruction::get_distortion() {
  return this->distortion;
}

SfmReconstruction::~SfmReconstruction(){};
