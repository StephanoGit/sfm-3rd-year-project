#include "SfmStructures.h"
#include <opencv2/core/types.hpp>

class StereoUtil {
public:
  StereoUtil();
  virtual ~StereoUtil();

  bool compute_F(Features &left, Features &right,
                 std::vector<cv::DMatch> &matches, cv::Mat &F,
                 std::vector<cv::DMatch> &mask_matches);
  bool compute_E(Features &left, Features &right,
                 std::vector<cv::DMatch> &matches, cv::Mat &E,
                 std::vector<cv::DMatch> &mask_matches);
  bool compute_Rt(Features &left, Features &right,
                  std::vector<cv::DMatch> &matches, cv::Mat &E, cv::Mat &R,
                  cv::Mat &t);
  bool compute_P(cv::Mat &P_left, cv::Mat &P_right);
  bool triangulate_views(ImagePair &image_pair,
                         std::vector<cv::DMatch> &matches, Features &left,
                         Features &right, cv::Mat &P_left, cv::Mat &P_right,
                         std::vector<PointCloudPoint> &point_cloud);

private:
  cv::Mat K;
  cv::Mat K_inverse;
  cv::Mat d;
};
