#include "../include/SfmReconstruction.h"
#include "../include/SfmBundleAdjustment.h"
#include "../include/drawUtil.h"
#include "../include/util.h"
#include <cstddef>
#include <iostream>
#include <opencv2/core/matx.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

SfmReconstruction::SfmReconstruction(std::string directory,
                                     FeatureExtractionType extract_type,
                                     FeatureMatchingType match_type,
                                     Intrinsics intrinsics) {
  this->directory = directory;
  this->extract_type = extract_type;
  this->match_type = match_type;
  this->intrinsics = intrinsics;
};
SfmReconstruction::~SfmReconstruction(){};

bool SfmReconstruction::run_sfm_reconstruction(bool downscale) {

  std::cout << "=======================================" << std::endl;
  std::cout << "     Starting SfM Reconstruction...    " << std::endl;
  std::cout << "=======================================" << std::endl;
  // load images
  this->images = load_images(directory, downscale);
  if (this->images.size() < 2) {
    std::cout << "Error: Please provide at least 2 images..." << std::endl;
    return false;
  }

  // setting up more stuff
  this->n_P_mats = std::vector<cv::Mat>(this->images.size());

  // extract features
  FeatureUtil feature_util(this->extract_type, this->match_type);
  this->images_features = std::vector<Features>(this->images.size());
  for (size_t i = 0; i < this->images.size(); i++) {
    this->images_features[i] = feature_util.extract_features(this->images[i]);
  }

  // match features
  create_match_matrix(feature_util, this->images_features);

  // initial triangulation
  ImagePair image_pair{7, 8};
  initial_triangulation(image_pair, this->images_features[image_pair.left],
                        this->images_features[image_pair.right],
                        this->match_matrix[image_pair.left][image_pair.right]);
  // add more views
  //  add_view_to_reconstruction();
  export_point_cloud(this->n_point_cloud, "final.json");
  return true;
}

void SfmReconstruction::create_match_matrix(
    FeatureUtil feature_util, std::vector<Features> images_features) {
  size_t number_of_images = this->images.size();
  this->match_matrix.resize(
      number_of_images, std::vector<std::vector<cv::DMatch>>(number_of_images));

  for (size_t i = 0; i < number_of_images; i++) {
    for (size_t j = i + 1; j < number_of_images; j++) {
      this->match_matrix[i][j] =
          feature_util.match_features(images_features[i], images_features[j]);
      std::cout << "Match (pair " << i << ", " << j
                << "): " << this->match_matrix[i][j].size()
                << " matched features" << std::endl;
    }
  }
  // for (size_t i = 0; i < number_of_images - 1; i++) {
  //   this->match_matrix[i][i + 1] =
  //      feature_util.match_features(images_features[i], images_features[i +
  //      1]);
  //  std::cout << "Match (pair " << i << ", " << i + 1
  //           << "): " << this->match_matrix[i][i + 1].size()
  //           << " matched features" << std::endl;
  //}
}

bool SfmReconstruction::initial_triangulation(ImagePair image_pair,
                                              Features features_left,
                                              Features features_right,
                                              std::vector<cv::DMatch> matches) {
  std::cout << "=======================================" << std::endl;
  std::cout << "  Performing initial triangulation...  " << std::endl;
  std::cout << "=======================================" << std::endl;

  if (matches.size() == 0) {
    std::cout << "Error: No matches between features..." << std::endl
              << std::endl;
  }
  std::cout << "No. Matches: " << matches.size() << std::endl << std::endl;
  StereoUtil stereo_util;
  std::cout << features_left.key_points.size() << "  "
            << features_right.key_points.size() << std::endl;
  // Compute the Fundamental Matrix
  // Remove outliers using F mask
  cv::Mat F;
  std::vector<cv::DMatch> pruned_matches_F;
  stereo_util.compute_F(features_left, features_right, matches, F,
                        pruned_matches_F);
  std::cout << "No. Matches after F mask: " << pruned_matches_F.size()
            << std::endl
            << std::endl;

  // Compute the Essential Matrix
  // Remove outliers using E mask
  cv::Mat E;
  std::vector<cv::DMatch> pruned_matches_E;
  stereo_util.compute_E(features_left, features_right, pruned_matches_F,
                        this->intrinsics, E, pruned_matches_E);
  std::cout << "No. Matches after E mask: " << pruned_matches_E.size()
            << std::endl
            << std::endl;

  // Compute the Rotation and translation
  cv::Mat R, t;
  stereo_util.compute_Rt(features_left, features_right, pruned_matches_E,
                         this->intrinsics, E, R, t);

  // Compute Projection Matrices
  cv::Mat P_left, P_right;
  stereo_util.compute_P(P_left, P_right, R, t);

  // Triangulate initial views
  std::vector<PointCloudPoint> point_cloud;
  stereo_util.triangulate_views(image_pair, pruned_matches_E, features_left,
                                features_right, this->intrinsics, P_left,
                                P_right, point_cloud);

  this->n_point_cloud = point_cloud;

  this->n_P_mats[image_pair.left] = P_left;
  this->n_P_mats[image_pair.right] = P_right;

  this->n_done_views.insert(image_pair.left);
  this->n_done_views.insert(image_pair.right);

  this->n_good_views.insert(image_pair.left);
  this->n_good_views.insert(image_pair.right);

  // adjust_bundle(this->n_point_cloud, n_P_mats, cv::Mat K, cv::Mat d,
  //               std::vector<Features> & features);

  return true;
}

void SfmReconstruction::add_view_to_reconstruction() {
  std::cout << "=======================================" << std::endl;
  std::cout << "          Adding more views...         " << std::endl;
  std::cout << "=======================================" << std::endl;

  StereoUtil stereo_util;
  while (this->n_done_views.size() != this->images.size()) {
    Image2D3DMatches matches_2D3D = find_2D3D_matches();

    size_t best_view;
    size_t best_number_matches = 0;
    for (const auto &match : matches_2D3D) {
      int number_matches = match.second.points_2D.size();
      if (number_matches > best_number_matches) {
        best_view = match.first;
        best_number_matches = number_matches;
      }
    }

    std::cout << "Best view to add next: " << best_view << " with "
              << best_number_matches << " matches" << std::endl;

    this->n_done_views.insert(best_view);

    cv::Mat new_P;
    bool success = stereo_util.camera_pose_from_2D3D_matches(
        matches_2D3D[best_view], this->intrinsics, new_P);

    if (!success) {
      std::cout << "Error: Cannot recover camera pose from view: " << best_view
                << std::endl;
      continue;
    }

    this->n_P_mats[best_view] = new_P;

    bool new_view_success_triangulation = false;
    for (const int good_view : n_good_views) {
      size_t left_view_idx = (good_view < best_view) ? good_view : best_view;
      size_t right_view_idx = (good_view < best_view) ? best_view : good_view;

      StereoUtil stereo_util;
      std::vector<cv::DMatch> matches =
          this->match_matrix[left_view_idx][right_view_idx];
      Features features_left = this->images_features[left_view_idx];
      Features features_right = this->images_features[right_view_idx];

      // Compute the Fundamental Matrix
      // Remove outliers using F mask
      cv::Mat F;
      std::vector<cv::DMatch> pruned_matches_F;
      stereo_util.compute_F(features_left, features_right, matches, F,
                            pruned_matches_F);
      std::cout << "No. Matches after F mask: " << pruned_matches_F.size()
                << std::endl
                << std::endl;

      // std::vector<cv::DMatch> pruned_matches_H;
      // stereo_util.compute_homography(features_left, features_right, matches,
      //                                pruned_matches_H);
      // std::cout << "No. Matches after H mask: " << pruned_matches_H.size()
      //           << std::endl
      //           << std::endl;

      // Compute the Essential Matrix
      // Remove outliers using E mask
      cv::Mat E;
      std::vector<cv::DMatch> pruned_matches_E;
      stereo_util.compute_E(features_left, features_right, pruned_matches_F,
                            this->intrinsics, E, pruned_matches_E);
      std::cout << "No. Matches after E mask: " << pruned_matches_E.size()
                << std::endl
                << std::endl;

      this->match_matrix[left_view_idx][right_view_idx] = pruned_matches_E;

      // Compute the Rotation and translation
      cv::Mat R, t;
      stereo_util.compute_Rt(features_left, features_right, pruned_matches_E,
                             this->intrinsics, E, R, t);

      // Compute Projection Matrices
      cv::Mat P_left, P_right;
      stereo_util.compute_P(P_left, P_right, R, t);

      // Triangulate initial views
      ImagePair pair;
      pair.left = left_view_idx;
      pair.right = right_view_idx;
      std::vector<PointCloudPoint> point_cloud;
      bool success = stereo_util.triangulate_views(
          pair, pruned_matches_E, features_left, features_right,
          this->intrinsics, P_left, P_right, point_cloud);
      if (success) {
        // merge pointcloud
        merge_point_cloud(point_cloud);
        new_view_success_triangulation = true;
      }

      break;
    }
    if (new_view_success_triangulation) {
      // bundle adjustment needed
    }
    this->n_good_views.insert(best_view);
    break;
  }
}

SfmReconstruction::Image2D3DMatches SfmReconstruction::find_2D3D_matches() {
  Image2D3DMatches matches;

  // scan through the remaining images
  for (size_t view_idx = 0; view_idx < this->images.size(); view_idx++) {
    // skip the done views
    if (this->n_done_views.find(view_idx) != this->n_done_views.end()) {
      continue;
    }
    Image2D3DPair pair_2D3D;

    // scan through each point within the current point cloud
    for (const PointCloudPoint &cloud_point : this->n_point_cloud) {
      bool found_2D_point = false;

      // for each origin view of the poin cloud
      for (const auto &origin_view : cloud_point.orgin_view) {
        // 2D - 2D matching between views
        int origin_view_idx = origin_view.first;
        int origin_view_feature_idx = origin_view.second;

        // left index < right index (the match matrix is upper triangular, i <
        // j)
        //   | a00 a01 a02 a03|
        //   |  0  a11 a12 a13|
        //   |  0   0  a22 a23|
        //   |  0   0   0  a33|
        int left_view_idx =
            (origin_view_idx < view_idx) ? origin_view_idx : view_idx;
        int right_view_idx =
            (origin_view_idx < view_idx) ? view_idx : origin_view_idx;

        // scan all 2D - 2D matches between the new view and the origin view
        for (const cv::DMatch &m :
             this->match_matrix[left_view_idx][right_view_idx]) {
          int matched_2D_point_in_new_view = -1;

          if (origin_view_idx < view_idx) {
            // origin is on the left
            if (m.queryIdx == origin_view_feature_idx) {
              matched_2D_point_in_new_view = m.trainIdx;
            }
          } else {
            // origin is on the right
            if (m.trainIdx == origin_view_feature_idx) {
              matched_2D_point_in_new_view = m.queryIdx;
            }
          }

          if (matched_2D_point_in_new_view >= 0) {
            // point found
            Features &new_view_features = this->images_features[view_idx];
            pair_2D3D.points_2D.push_back(
                new_view_features.points[matched_2D_point_in_new_view]);

            pair_2D3D.points_3D.push_back(cloud_point.point);
            found_2D_point = true;
            break;
          }
        }
        if (found_2D_point) {
          break;
        }
      }
    }
    matches[view_idx] = pair_2D3D;
  }
  return matches;
}

void SfmReconstruction::merge_point_cloud(
    std::vector<PointCloudPoint> new_pointcloud) {
  std::cout << "=======================================" << std::endl;
  std::cout << "        Merging Point Clouds...        " << std::endl;
  std::cout << "=======================================" << std::endl;

  std::vector<std::vector<std::vector<cv::DMatch>>> merged_match_matrix;
  merged_match_matrix.resize(
      this->images.size(),
      std::vector<std::vector<cv::DMatch>>(this->images.size()));

  size_t new_points = 0;
  size_t merged_points = 0;

  for (const PointCloudPoint &pt : new_pointcloud) {
    const cv::Point3f new_point = pt.point;

    bool found_matching_existing_views = false;
    bool found_3D_point_match = false;

    for (PointCloudPoint &existing_point : this->n_point_cloud) {
      if (cv::norm(existing_point.point - new_point) < 0.01) {
        found_3D_point_match = true;

        for (const auto &new_view : pt.orgin_view) {
          for (const auto &existing_view : existing_point.orgin_view) {
            bool found_matching_feature = false;

            const bool new_is_left = new_view.first < existing_view.first;
            const int left_view_idx =
                (new_is_left) ? new_view.first : existing_view.first;
            const int left_view_feature_idx =
                (new_is_left) ? new_view.second : existing_view.second;
            const int right_view_idx =
                (new_is_left) ? existing_view.first : new_view.first;
            const int right_view_feature_idx =
                (new_is_left) ? existing_view.second : new_view.second;

            const std::vector<cv::DMatch> &matching =
                this->match_matrix[left_view_idx][right_view_idx];
            for (const cv::DMatch &match : matching) {
              if (match.queryIdx == left_view_feature_idx &&
                  match.trainIdx == right_view_feature_idx &&
                  match.distance < 20.0) {
                merged_match_matrix[left_view_idx][right_view_idx].push_back(
                    match);
                found_matching_feature = true;
                break;
              }
            }

            if (found_matching_feature) {
              existing_point.orgin_view[new_view.first] = new_view.second;
              found_matching_existing_views = true;
            }
          }
        }
      }
      if (found_matching_existing_views) {
        merged_points++;
        break;
      }
    }

    if (!found_matching_existing_views && !found_3D_point_match) {
      this->n_point_cloud.push_back(pt);
      new_points++;
    }
  }
}
