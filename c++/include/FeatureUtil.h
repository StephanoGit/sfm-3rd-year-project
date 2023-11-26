
#include "util.h"
#include <opencv2/core/types.hpp>

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
  FeatureUtil(FeatureExtractionType extract_type,
              FeatureMatchingType match_type);
  virtual ~FeatureUtil();

  Features extract_features(cv::Mat image);
  std::vector<cv::DMatch> match_features(Features left, Features right);

private:
  FeatureExtractionType extract_type;
  FeatureMatchingType match_type;
};
