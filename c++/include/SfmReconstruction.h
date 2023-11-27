#ifndef __SFM_RECONSTRUCTION
#define __SFM_RECONSTRUCTION

#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>

#include "FeatureUtil.h"
#include "SfmStructures.h"
#include "StereoUtil.h"

class SfmReconstruction {
  // matrix of macthes from image i to image j
  typedef std::vector<std::vector<std::vector<cv::DMatch>>> match_matrix;
  typedef std::map<int, Image2D3DPair> Image2D3DMatches;

private:
  std::string directory;
  FeatureExtractionType extract_type;
  FeatureMatchingType match_type;

public:
  SfmReconstruction(std::string directory, FeatureExtractionType extract_type,
                    FeatureMatchingType match_type);
  virtual ~SfmReconstruction();

  bool run_sfm_reconstruction();

  void extract_features();
  void create_match_matrix();
  void add_view_to_reconstruction();

  Image2D3DMatches find_2D3D_matches();

  void merge_point_cloud(std::vector<PointCloudPoint> point_cloud);
};

#endif
