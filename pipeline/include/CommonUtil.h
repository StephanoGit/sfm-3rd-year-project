#ifndef __COMMON_UTIL
#define __COMMON_UTIL

#include "SfmStructures.h"
#include <opencv2/core/types.hpp>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/vtk_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/surface/poisson.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>

std::vector<cv::DMatch>
apply_lowes_ratio(const std::vector<std::vector<cv::DMatch>> knn_matches);

void keypoints_to_points(const std::vector<cv::KeyPoint> &kps,
                         std::vector<cv::Point2f> &pts);

void remove_outliers(const std::vector<cv::DMatch> &matches, cv::Mat &mask,
                     std::vector<cv::DMatch> &mask_matches);

void align_points_from_matches(const Features &left, const Features &right,
                               const std::vector<cv::DMatch> &matches,
                               Features &aligned_left, Features &aligned_right,
                               std::vector<int> &left_origin,
                               std::vector<int> &right_origin);

bool ply_to_pcd(const std::string file_path, const std::string file_name);

bool pcd_to_mesh(const std::string file_path, const std::string file_name);

void display_mesh(pcl::PolygonMesh mesh);

void display_point_cloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud);
#endif
