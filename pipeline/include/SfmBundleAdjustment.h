#ifndef __BUNDLE_ADJUSTMENT
#define __BUNDLE_ADJUSTMENT

#include "SfmStructures.h"
#include "util.h"
#include <opencv2/core/matx.hpp>

class SfmBundleAdjustment {
public:
  static void adjust_bundle(std::vector<PointCloudPoint> &pointcloud,
                            std::vector<cv::Matx34f> &P_mats,
                            Intrinsics &intrinsics,
                            const std::vector<Features> &features);
};

#endif
