#ifdef __BUNDLE_ADJUSTMENT
#define __BUNDLE_ADJUSTMENT

#include "util.h"

class SfmBundleAdjustment {
public:
  static void adjust_bundle(std::vector<PointCloudPoint> &pointcloud,
                            std::vector<cv::Mat> &P_mats, cv::Mat K, cv::Mat d,
                            std::vector<Features> &features);
};

#endif
