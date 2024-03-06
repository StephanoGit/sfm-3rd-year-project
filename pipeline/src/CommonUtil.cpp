#include "../include/SfmStructures.h"
#include <opencv2/core/mat.hpp>
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
apply_lowes_ratio(const std::vector<std::vector<cv::DMatch>> knn_matches) {
    std::vector<cv::DMatch> good_matches;
    for (int i = 0; i < knn_matches.size(); i++) {
        if (knn_matches[i][0].distance < 0.75 * knn_matches[i][1].distance) {
            good_matches.push_back(knn_matches[i][0]);
        }
    }
    return good_matches;
}

void keypoints_to_points(const std::vector<cv::KeyPoint> &kps,
                         std::vector<cv::Point2f> &pts) {
    pts.clear();
    for (const auto &kp : kps) {
        pts.push_back(kp.pt);
    }
}

void remove_outliers(const std::vector<cv::DMatch> &matches, cv::Mat &mask,
                     std::vector<cv::DMatch> &mask_matches) {
    mask_matches.clear();
    for (size_t i = 0; i < matches.size(); i++) {
        if (mask.at<uchar>(i)) {
            mask_matches.push_back(matches[i]);
        }
    }
}

void align_points_from_matches(const Features &left, const Features &right,
                               const std::vector<cv::DMatch> &matches,
                               Features &aligned_left, Features &aligned_right,
                               std::vector<int> &left_origin,
                               std::vector<int> &right_origin) {
    aligned_left.key_points.clear();
    aligned_right.key_points.clear();
    aligned_left.descriptors = cv::Mat();
    aligned_right.descriptors = cv::Mat();

    for (size_t i = 0; i < matches.size(); i++) {
        aligned_left.key_points.push_back(left.key_points[matches[i].queryIdx]);
        aligned_right.key_points.push_back(
            right.key_points[matches[i].trainIdx]);

        aligned_left.descriptors.push_back(
            left.descriptors.row(matches[i].queryIdx));
        aligned_right.descriptors.push_back(
            right.descriptors.row(matches[i].trainIdx));

        left_origin.push_back(matches[i].queryIdx);
        right_origin.push_back(matches[i].trainIdx);
    }

    keypoints_to_points(aligned_left.key_points, aligned_left.points);
    keypoints_to_points(aligned_right.key_points, aligned_right.points);
}

bool ply_to_pcd(const std::string file_path, const std::string file_name) {
    std::cout << "===========================================" << std::endl;
    std::cout << "                PLY to PCD                 " << std::endl;
    std::cout << "===========================================" << std::endl;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_ply(
        new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PLYReader ply;

    ply.read(file_path, *cloud_ply);

    if (cloud_ply->size() <= 0) {
        std::cout << "ERROR: sparse pointcloud -> dense pointcloud failed -- "
                     "ply file empty [❌]"
                  << std::endl;
        return false;
    }

    std::cout << "LOG: sparse pointcloud -> dense pointcloud [✅]" << std::endl;

    pcl::io::savePCDFile("../reconstructions/" + file_name +
                             "/dense/point_cloud.pcd",
                         *cloud_ply);
    pcl::io::savePLYFile("../reconstructions/" + file_name +
                             "/dense/point_cloud.ply",
                         *cloud_ply);
    return true;
}

void display_mesh(pcl::PolygonMesh mesh) {
    if (mesh.polygons.empty()) {
        std::cout << "ERROR: Failed to load mesh file -- [❌]" << std::endl;
        return;
    }

    // visualize mesh
    pcl::shared_ptr<pcl::visualization::PCLVisualizer> viewer(
        new pcl::visualization::PCLVisualizer("MAP3D MESH"));
    viewer->setBackgroundColor(0, 0, 0);
    viewer->addPolygonMesh(mesh, "meshes", 0);
    // viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();

    std::cout << "Press q to exit." << std::endl;
    while (!viewer->wasStopped()) {
        viewer->spin();
    }
}

void display_point_cloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud) {
    if (cloud->size() <= 0) {
        std::cout << "ERROR: Failed to load point cloud file [❌]" << std::endl;
        return;
    }

    // visualize point cloud
    pcl::shared_ptr<pcl::visualization::PCLVisualizer> viewer(
        new pcl::visualization::PCLVisualizer("MAP3D MESH"));
    viewer->setBackgroundColor(0, 0, 0);
    viewer->addPointCloud(cloud, "CLOUD", 0);
    // viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();

    std::cout << "Press q to exit." << std::endl;
    while (!viewer->wasStopped()) {
        viewer->spin();
    }
}
