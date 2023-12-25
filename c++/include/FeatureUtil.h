#ifndef __FEATURE_UTIL
#define __FEATURE_UTIL

#include "SfmStructures.h"
#include <opencv2/core/cvstd_wrapper.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/features2d.hpp>

enum FeatureExtractionType {
  SIFT,
  ORB,
  FAST,
  SURF,
};

enum FeatureMatchingType {
  BF,
  FLANN,
};

class FeatureUtil {
public:
  FeatureUtil();
  FeatureUtil(FeatureExtractionType extract_type,
              FeatureMatchingType match_type);
  virtual ~FeatureUtil();

  Features extract_features(const cv::Mat &image);
  std::vector<cv::DMatch> match_features(const Features &features_left,
                                         const Features &features_right);

private:
  FeatureExtractionType extract_type;
  FeatureMatchingType match_type;
};

#endif
