#include "../include/StereoUtil.h"
#include "../include/FeatureUtil.h"

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/matx.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/xfeatures2d.hpp>

StereoUtil::StereoUtil() {}

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

void StereoUtil::align_points_from_matches(Features &left, Features &right,
                                           std::vector<cv::DMatch> &matches,
                                           Features &aligned_left,
                                           Features &aligned_right,
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

  FeatureUtil::keypoints_to_points(aligned_left.key_points,
                                   aligned_left.points);
  FeatureUtil::keypoints_to_points(aligned_right.key_points,
                                   aligned_right.points);
}

bool StereoUtil::compute_homography(Features &left, Features &right,
                                    std::vector<cv::DMatch> &matches,
                                    std::vector<cv::DMatch> &mask_matches) {

  std::cout << "=======================================" << std::endl;
  std::cout << " Computing the Homography Matrix (H)..." << std::endl;
  std::cout << "=======================================" << std::endl;

  if (matches.size() == 0) {
    std::cout << "Error: Please compute matches between images..." << std::endl;
    return false;
  }

  mask_matches.clear();
  Features aligned_left;
  Features aligned_right;
  std::vector<int> left_origin, right_origin;
  align_points_from_matches(left, right, matches, aligned_left, aligned_right,
                            left_origin, right_origin);
  if (matches.size() < 4) {
    std::cout << "Error: Not enough matches..." << std::endl;
  }
  cv::Mat mask;
  cv::Mat H;
  H = cv::findHomography(aligned_left.points, aligned_right.points, cv::RANSAC,
                         10.0, mask);
  remove_outliers(matches, mask, mask_matches);

  return true;
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

  mask_matches.clear();
  Features aligned_left;
  Features aligned_right;
  std::vector<int> left_origin, right_origin;
  align_points_from_matches(left, right, matches, aligned_left, aligned_right,
                            left_origin, right_origin);

  std::cout << aligned_left.points.size() << std::endl;
  cv::Mat mask;
  F = cv::findFundamentalMat(aligned_left.points, aligned_right.points,
                             cv::FM_RANSAC, 1.0, 0.999, mask);
  remove_outliers(matches, mask, mask_matches);
  return true;
}

bool StereoUtil::compute_E(Features &left, Features &right,
                           std::vector<cv::DMatch> &matches,
                           Intrinsics &intrinsics, cv::Mat &E,
                           std::vector<cv::DMatch> &mask_matches) {

  std::cout << "=======================================" << std::endl;
  std::cout << " Computing the Essential Matrix (E)... " << std::endl;
  std::cout << "=======================================" << std::endl;

  if (matches.size() == 0) {
    std::cout << "Error: Please compute matches between images..." << std::endl;
    return false;
  }

  mask_matches.clear();
  Features aligned_left;
  Features aligned_right;
  std::vector<int> left_origin, right_origin;
  align_points_from_matches(left, right, matches, aligned_left, aligned_right,
                            left_origin, right_origin);

  cv::Mat mask;
  E = cv::findEssentialMat(aligned_left.points, aligned_right.points,
                           intrinsics.K, cv::RANSAC, 0.999, 1.0, mask);
  remove_outliers(matches, mask, mask_matches);
  return true;
}

bool StereoUtil::compute_Rt(Features &left, Features &right,
                            std::vector<cv::DMatch> &matches,
                            Intrinsics &intrinsics, cv::Mat &E, cv::Mat &R,
                            cv::Mat &t) {

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

  cv::recoverPose(E, aligned_left.points, aligned_right.points, intrinsics.K, R,
                  t, cv::noArray());

  return true;
}

void StereoUtil::compute_P(cv::Mat &P_left, cv::Mat &P_right, cv::Mat &R,
                           cv::Mat &t) {
  P_left = cv::Mat::eye(3, 4, CV_32F);

  cv::hconcat(R, t, P_right);
  // P_right = this->K * P_right;
}

bool StereoUtil::triangulate_views(ImagePair &image_pair,
                                   std::vector<cv::DMatch> &matches,
                                   Features &left, Features &right,
                                   Intrinsics &intrinsics, cv::Mat &P_left,
                                   cv::Mat &P_right,
                                   std::vector<PointCloudPoint> &point_cloud) {

  std::cout << "=======================================" << std::endl;
  std::cout << "      Performing Triangulation...      " << std::endl;
  std::cout << "=======================================" << std::endl;

  if (matches.size() == 0) {
    std::cout << "Error: Please compute matches between images..." << std::endl;
    return false;
  }

  point_cloud.clear();
  Features aligned_left;
  Features aligned_right;
  std::vector<int> left_origin, right_origin;
  align_points_from_matches(left, right, matches, aligned_left, aligned_right,
                            left_origin, right_origin);

  cv::Mat norm_left, norm_right;
  cv::undistortPoints(aligned_left.points, norm_left, intrinsics.K, cv::Mat());
  cv::undistortPoints(aligned_right.points, norm_right, intrinsics.K,
                      cv::Mat());

  cv::Mat points_4D;
  cv::triangulatePoints(P_left, P_right, norm_left, norm_right, points_4D);

  cv::Mat points_3D;
  cv::convertPointsFromHomogeneous(points_4D.t(), points_3D);

  cv::Matx34f P_left_(P_left);
  cv::Mat rvec_left;
  cv::Rodrigues(P_left_.get_minor<3, 3>(0, 0), rvec_left);
  cv::Mat tvec_left(P_left_.get_minor<3, 1>(0, 3).t());

  std::vector<cv::Point2f> projected_left(aligned_left.points.size());
  cv::projectPoints(points_3D, rvec_left, tvec_left, intrinsics.K, intrinsics.d,
                    projected_left);

  cv::Matx34f P_right_(P_right);
  cv::Mat rvec_right;
  cv::Rodrigues(P_right_.get_minor<3, 3>(0, 0), rvec_right);
  cv::Mat tvec_right(P_right_.get_minor<3, 1>(0, 3).t());

  std::vector<cv::Point2f> projected_right(aligned_right.points.size());
  cv::projectPoints(points_3D, rvec_right, tvec_right, intrinsics.K,
                    intrinsics.d, projected_right);

  float error_left;
  float error_right;

  for (size_t i = 0; i < points_3D.rows; i++) {
    // should check for reprojection Error
    error_left = cv::norm(projected_left[i] - aligned_left.points[i]);
    error_right = cv::norm(projected_left[i] - aligned_left.points[i]);
    if (error_left > 5 || error_right > 5) {
      std::cout << "Reprojection error for point: " << i << std::endl;
      std::cout << "     -> left: " << error_left << std::endl;
      std::cout << "     -> right: " << error_right << std::endl;
      continue;
    }

    PointCloudPoint pt;
    pt.point = cv::Point3f(points_3D.at<float>(i, 0), points_3D.at<float>(i, 1),
                           points_3D.at<float>(i, 2));

    pt.orgin_view[image_pair.left] = left_origin[i];
    pt.orgin_view[image_pair.right] = right_origin[i];
    point_cloud.push_back(pt);
  }

  return true;
}

bool StereoUtil::camera_pose_from_2D3D_matches(Image2D3DPair &match,
                                               Intrinsics &intrinsics,
                                               cv::Mat &P) {
  cv::Mat rvec, tvec;
  cv::Mat mask;
  std::cout << intrinsics.K << std::endl;
  std::cout << intrinsics.d << std::endl;
  std::cout << match.points_3D.size() << std::endl;
  std::cout << match.points_2D.size() << std::endl;
  cv::solvePnPRansac(match.points_3D, match.points_2D, intrinsics.K,
                     intrinsics.d, rvec, tvec, mask);

  // check for inliers ratio

  cv::Mat R;
  cv::Rodrigues(rvec, R);
  cv::hconcat(R, tvec, P);

  return true;
}
