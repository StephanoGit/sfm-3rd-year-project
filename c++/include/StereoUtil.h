#include "SfmStructures.h"
#include <opencv2/core/types.hpp>

class StereoUtil {
public:
  StereoUtil();
  virtual ~StereoUtil();

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
  void compute_P(cv::Mat &P_left, cv::Mat &P_right, cv::Mat &R, cv::Mat &t);
  bool triangulate_views(ImagePair &image_pair,
                         std::vector<cv::DMatch> &matches, Features &left,
                         Features &right, Intrinsics &intrinsics,
                         cv::Mat &P_left, cv::Mat &P_right,
                         std::vector<PointCloudPoint> &point_cloud);

  void align_points_from_matches(Features &left, Features &right,
                                 std::vector<cv::DMatch> &matches,
                                 Features &aligned_left,
                                 Features &aligned_right,
                                 std::vector<int> &left_origin,
                                 std::vector<int> &right_origin);

  bool camera_pose_from_2D3D_matches(Image2D3DPair &match,
                                     Intrinsics &intrinsics, cv::Mat &P);
};
