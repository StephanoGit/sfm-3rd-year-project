#ifndef __STEREO_UTIL
#define __STEREO_UTIL

#include "SfmStructures.h"
#include <opencv2/core/matx.hpp>
#include <opencv2/core/types.hpp>
#include <vector>

class StereoUtil {
public:
  StereoUtil();
  virtual ~StereoUtil();

  static bool camera_matrix_from_match(const Intrinsics &intrinsics,
                                       const std::vector<cv::DMatch> &matches,
                                       const Features &features_left,
                                       const Features &features_right,
                                       std::vector<cv::DMatch> mask_matches,
                                       cv::Matx34f &P_left,
                                       cv::Matx34f &P_right);

  static bool camera_matrices_from_matches(
      const Intrinsics &intrinsics, const std::vector<cv::DMatch> &matches,
      const Features &features_left, const Features &features_right,
      std::vector<cv::DMatch> mask_matches, cv::Matx34f &P_left,
      cv::Matx34f &P_right);

  static int homography_inliers(const Features &features_left,
                                const Features &features_right,
                                const std::vector<cv::DMatch> &matches);

  static bool triangulate_views_homography(
      const Intrinsics &intrinsics, const ImagePair image_pair,
      const std::vector<cv::DMatch> &matches, const Features &features_left,
      const Features &features_right, const cv::Matx34f &P_left,
      const cv::Matx34f &P_right, std::vector<PointCloudPoint> &point_cloud);

  static bool P_from_2D3D_matches(const Intrinsics &intrinsics,
                                  const Image2D3DPair &match, cv::Matx34f &P);

  bool compute_homography(Features &left, Features &right,
                          std::vector<cv::DMatch> &matches,
                          std::vector<cv::DMatch> &mask_matches);

  bool compute_F(Features &left, Features &right,
                 std::vector<cv::DMatch> &matches, cv::Mat &F,
                 std::vector<cv::DMatch> &mask_matches);
  bool compute_E(Features &left, Features &right,
                 std::vector<cv::DMatch> &matches, Intrinsics &intrinsics,
                 cv::Mat &E, std::vector<cv::DMatch> &mask_matches);
  bool compute_Rt(Features &left, Features &right,
                  std::vector<cv::DMatch> &matches, Intrinsics &intrinsics,
                  cv::Mat &E, cv::Mat &R, cv::Mat &t);
  void compute_P(cv::Matx34f &P_left, cv::Matx34f &P_right, cv::Mat &R,
                 cv::Mat &t);
  bool triangulate_views(ImagePair &image_pair,
                         std::vector<cv::DMatch> &matches, Features &left,
                         Features &right, Intrinsics &intrinsics,
                         cv::Matx34f &P_left, cv::Matx34f &P_right,
                         std::vector<PointCloudPoint> &point_cloud);

  bool camera_pose_from_2D3D_matches(Image2D3DPair &match,
                                     Intrinsics &intrinsics, cv::Matx34f &P);
};

#endif
