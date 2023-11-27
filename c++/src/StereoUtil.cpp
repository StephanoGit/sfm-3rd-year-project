#include "../include/StereoUtil.h"
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>

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

StereoUtil::StereoUtil() {
  this->K = cv::Mat(3, 3, CV_64F, data);
  this->K_inverse = this->K.inv();
  this->d = cv::Mat::zeros(1, 5, CV_32F);
}

StereoUtil::~StereoUtil() {}

void remove_outliers(std::vector<cv::DMatch> &matches, cv::Mat &mask,
                     std::vector<cv::DMatch> &mask_matches) {
  mask_matches.clear();
  for (size_t i = 0; i < matches.size(); i++) {
    if (mask.at<uchar>(i)) {
      mask_matches.push_back(matches[i]);
    }
  }
}

void keypoints_to_points(std::vector<cv::KeyPoint> &keypoints,
                         std::vector<cv::Point2f> &points_2d) {
  points_2d.clear();
  for (const auto &kp : keypoints) {
    points_2d.push_back(kp.pt);
  }
}

void align_points_from_matches(Features &left, Features &right,
                               std::vector<cv::DMatch> &matches,
                               Features &aligned_left, Features &aligned_right,
                               std::vector<int> &left_origin,
                               std::vector<int> &right_origin) {
  aligned_left.key_points.clear();
  aligned_right.key_points.clear();
  aligned_left.descriptors = cv::Mat();
  aligned_right.descriptors = cv::Mat();

  for (size_t i = 0; i < matches.size(); i++) {
    aligned_left.key_points.push_back(left.key_points[matches[i].queryIdx]);
    aligned_right.key_points.push_back(right.key_points[matches[i].trainIdx]);

    aligned_left.descriptors.push_back(
        left.descriptors.row(matches[i].queryIdx));
    aligned_right.descriptors.push_back(
        right.descriptors.row(matches[i].trainIdx));

    left_origin.push_back(matches[i].queryIdx);
    right_origin.push_back(matches[i].trainIdx);
  }

  keypoints_to_points(aligned_left.key_points, aligned_left.points);
  keypoints_to_points(aligned_right.key_points, aligned_right.points);
}

bool StereoUtil::compute_F(Features &left, Features &right,
                           std::vector<cv::DMatch> &matches, cv::Mat &F,
                           std::vector<cv::DMatch> &mask_matches) {

  std::cout << "=======================================" << std::endl;
  std::cout << "Computing the Fundamental Matrix (F)..." << std::endl;
  std::cout << "=======================================" << std::endl;

  if (matches.size() == 0) {
    std::cout << "Error: Please compute matches between images..." << std::endl;
    return false;
  }

  Features aligned_left;
  Features aligned_right;
  std::vector<int> left_origin, right_origin;
  align_points_from_matches(left, right, matches, aligned_left, aligned_right,
                            left_origin, right_origin);

  cv::Mat mask;
  F = cv::findFundamentalMat(aligned_left.points, aligned_right.points,
                             cv::FM_RANSAC, 1.0, 0.999, mask);
  remove_outliers(matches, mask, mask_matches);

  return true;
}

bool StereoUtil::compute_E(Features &left, Features &right,
                           std::vector<cv::DMatch> &matches, cv::Mat &E,
                           std::vector<cv::DMatch> &mask_matches) {

  std::cout << "=======================================" << std::endl;
  std::cout << " Computing the Essential Matrix (E)... " << std::endl;
  std::cout << "=======================================" << std::endl;

  if (matches.size() == 0) {
    std::cout << "Error: Please compute matches between images..." << std::endl;
    return false;
  }

  Features aligned_left;
  Features aligned_right;
  std::vector<int> left_origin, right_origin;
  align_points_from_matches(left, right, matches, aligned_left, aligned_right,
                            left_origin, right_origin);

  cv::Mat mask;
  E = cv::findEssentialMat(aligned_left.points, aligned_right.points, this->K,
                           cv::RANSAC, 0.999, 1.0, mask);
  remove_outliers(matches, mask, mask_matches);
  return true;
}

bool StereoUtil::compute_Rt(Features &left, Features &right,
                            std::vector<cv::DMatch> &matches, cv::Mat &E,
                            cv::Mat &R, cv::Mat &t) {

  std::cout << "=======================================" << std::endl;
  std::cout << " Computing Rotation and transaltion... " << std::endl;
  std::cout << "=======================================" << std::endl;

  if (matches.size() == 0) {
    std::cout << "Error: Please compute matches between images..." << std::endl;
    return false;
  }

  Features aligned_left;
  Features aligned_right;
  std::vector<int> left_origin, right_origin;
  align_points_from_matches(left, right, matches, aligned_left, aligned_right,
                            left_origin, right_origin);

  cv::recoverPose(E, aligned_left.points, aligned_right.points, this->K, R, t,
                  cv::noArray());

  return true;
}

void compute_P(cv::Mat &P_left, cv::Mat &P_right, cv::Mat &R, cv::Mat &t) {
  P_left = cv::Mat::eye(3, 4, CV_32F);
  cv::hconcat(R, t, P_right);
}

bool StereoUtil::triangulate_views(ImagePair &image_pair,
                                   std::vector<cv::DMatch> &matches,
                                   Features &left, Features &right,
                                   cv::Mat &P_left, cv::Mat &P_right,
                                   std::vector<PointCloudPoint> &point_cloud) {

  std::cout << "=======================================" << std::endl;
  std::cout << "      Performing Triangulation...      " << std::endl;
  std::cout << "=======================================" << std::endl;

  if (matches.size() == 0) {
    std::cout << "Error: Please compute matches between images..." << std::endl;
    return false;
  }

  Features aligned_left;
  Features aligned_right;
  std::vector<int> left_origin, right_origin;
  align_points_from_matches(left, right, matches, aligned_left, aligned_right,
                            left_origin, right_origin);

  cv::Mat points_4D;
  cv::triangulatePoints(P_left, P_right, aligned_left.points,
                        aligned_right.points, points_4D);

  cv::Mat points_3D;
  cv::convertPointsFromHomogeneous(points_4D.t(), points_3D);

  cv::Matx34f P_left_(P_left);
  cv::Mat rvec_left;
  cv::Rodrigues(P_left_.get_minor<3, 3>(0, 0), rvec_left);
  cv::Mat tvec_left(P_left_.get_minor<3, 1>(0, 3).t());

  std::vector<cv::Point2f> projected_left(aligned_left.points.size());
  cv::projectPoints(points_3D, rvec_left, tvec_left, this->K, this->d,
                    projected_left);

  cv::Matx34f P_right_(P_right);
  cv::Mat rvec_right;
  cv::Rodrigues(P_right_.get_minor<3, 3>(0, 0), rvec_right);
  cv::Mat tvec_right(P_right_.get_minor<3, 1>(0, 3).t());

  std::vector<cv::Point2f> projected_right(aligned_right.points.size());
  cv::projectPoints(points_3D, rvec_right, tvec_right, this->K, this->d,
                    projected_right);

  for (size_t i = 0; i < points_3D.rows; i++) {
    // should check for reprojection Error

    PointCloudPoint pt;
    pt.point = cv::Point3f(points_3D.at<float>(i, 0), points_3D.at<float>(i, 1),
                           points_3D.at<float>(i, 2));

    // pt.orgin_view[image_pair.left] = left_origin[i];
    // pt.orgin_view[image_pair.right] = right_origin[i];

    point_cloud.push_back(pt);
  }

  return true;
}
