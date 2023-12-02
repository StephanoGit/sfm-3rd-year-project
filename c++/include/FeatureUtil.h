
#include "SfmStructures.h"
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

  Features extract_features(cv::Mat &image);
  std::vector<cv::DMatch> match_features(Features &left, Features &right);

  static void keypoints_to_points(std::vector<cv::KeyPoint> &kps,
                                  std::vector<cv::Point2f> &pts);

private:
  FeatureExtractionType extract_type;
  FeatureMatchingType match_type;
};
