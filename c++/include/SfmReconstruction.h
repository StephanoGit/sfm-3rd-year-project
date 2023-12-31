#ifndef __SFM_RECONSTRUCTION
#define __SFM_RECONSTRUCTION

#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>

#include "FeatureUtil.h"
#include "SfmStructures.h"
#include "StereoUtil.h"

class SfmReconstruction {
  // matrix of macthes from image i to image j
  typedef std::vector<std::vector<std::vector<cv::DMatch>>> MatchMatrix;
  typedef std::map<int, Image2D3DPair> Image2D3DMatches;

private:
  std::string directory;
  FeatureExtractionType extract_type;
  FeatureMatchingType match_type;

  Intrinsics intrinsics;
  MatchMatrix match_matrix;
  std::vector<Features> images_features;
  std::vector<cv::Mat> images;

  std::vector<PointCloudPoint> n_point_cloud;
  std::vector<cv::Matx34f> n_P_mats;
  std::set<int> n_done_views;
  std::set<int> n_good_views;

public:
  SfmReconstruction(std::string directory, FeatureExtractionType extract_type,
                    FeatureMatchingType match_type, Intrinsics intrinsics);
  virtual ~SfmReconstruction();

  bool run_sfm_reconstruction(bool downscale);
  bool initial_triangulation(ImagePair image_pair, Features features_left,
                             Features features_right,
                             std::vector<cv::DMatch> matches);

  void create_match_matrix(FeatureUtil feature_util,
                           std::vector<Features> images_features);
  void add_view_to_reconstruction();

  Image2D3DMatches find_2D3D_matches();

  void merge_point_cloud(std::vector<PointCloudPoint> point_cloud);
};

#endif
