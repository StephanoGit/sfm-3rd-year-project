#include "util.h"
#include <opencv2/core/types.hpp>

class StereoUtil {
public:
  StereoUtil();
  virtual ~StereoUtil();

  bool compute_F(Features &left, Features &right,
                 std::vector<cv::DMatch> matches, cv::Mat &F);
  bool compute_E(Features &left, Features &right,
                 std::vector<cv::DMatch> matches, cv::Mat &E);
  bool compute_Rt(Features &left, Features &right,
                  std::vector<cv::DMatch> macthes, cv::Mat &R, cv::Mat &t);
  bool compute_P(cv::Mat &P_left, cv::Mat &P_right);
  bool triangulate_views();

private:
  cv::Mat K;
  cv::Mat K_inverse;
  cv::Mat d;
};
