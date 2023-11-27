#include "../include/SfmReconstruction.h"
#include "../include/util.h"
#include <opencv2/opencv.hpp>

SfmReconstruction::SfmReconstruction(std::string directory,
                                     FeatureExtractionType extract_type,
                                     FeatureMatchingType match_type) {
  this->directory = directory;
  this->extract_type = extract_type;
  this->match_type = match_type;
};
SfmReconstruction::~SfmReconstruction(){};

bool SfmReconstruction::run_sfm_reconstruction() {

  std::cout << "=======================================" << std::endl;
  std::cout << "     Starting SfM Reconstruction...    " << std::endl;
  std::cout << "=======================================" << std::endl;
  // load images
  std::vector<cv::Mat> images = load_images(directory);
  if (images.size() < 2) {
    std::cout << "Error: Please provide at least 2 images..." << std::endl;
    return false;
  }

  // extract features
  FeatureUtil feature_util(this->extract_type, this->match_type);
  std::vector<Features> images_features(images.size());
  for (size_t i = 0; i < images.size(); i++) {
    images_features[i] = feature_util.extract_features(images[i]);
    std::cout << images_features[i].key_points.size() << std::endl;
  }

  // match features

  // initial triangulation

  // add more views
  return true;
}
