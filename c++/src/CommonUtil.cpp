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

    std::cout << "LOG: saving file with prefix -- " << file_name << "_MAP3D.pcd"
              << std::endl;
    pcl::io::savePCDFile(file_name + "_MAP3D.pcd", *cloud_ply);

    return true;
}

bool pcd_to_mesh(const std::string file_path) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(
        new pcl::PointCloud<pcl::PointXYZ>);
    pcl::io::loadPCDFile(file_path, *cloud);

    if (cloud->size() <= 0) {
        std::cout << "ERROR: Failed to load pointcloud -- pcd file empty [❌]"
                  << std::endl;
        return false;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(
        new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PolygonMesh mesh;

    // filter pointcloud -- remove points where x NOT ∈ [0.003, 0.83]
    pcl::PassThrough<pcl::PointXYZ> pass_through;
    pass_through.setInputCloud(cloud);
    pass_through.setFilterFieldName("x");
    pass_through.setFilterLimits(0.003, 0.83);
    pass_through.filter(*filtered_cloud);

    // remove outliers -- must have at least 150 neighbours in a 0.07 radius
    pcl::RadiusOutlierRemoval<pcl::PointXYZ> radius_outlier_removal;
    radius_outlier_removal.setInputCloud(cloud);
    radius_outlier_removal.setRadiusSearch(0.07);
    radius_outlier_removal.setMinNeighborsInRadius(150);
    radius_outlier_removal.filter(*filtered_cloud);

    // create mesh
    pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> ne;
    ne.setNumberOfThreads(8);
    ne.setInputCloud(cloud);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(
        new pcl::search::KdTree<pcl::PointXYZ>());
    ne.setSearchMethod(tree);
    ne.setKSearch(10); // 20
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(
        new pcl::PointCloud<pcl::Normal>());
    ne.compute(*cloud_normals);

    for (std::size_t i = 0; i < cloud_normals->size(); ++i) {
        cloud_normals->points[i].normal_x *= -1;
        cloud_normals->points[i].normal_y *= -1;
        cloud_normals->points[i].normal_z *= -1;
    }
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_smoothed_normals(
        new pcl::PointCloud<pcl::PointNormal>());
    pcl::concatenateFields(*cloud, *cloud_normals,
                           *cloud_smoothed_normals); // x

    pcl::Poisson<pcl::PointNormal> poisson;
    poisson.setDepth(7); // 9
    poisson.setInputCloud(cloud_smoothed_normals);
    poisson.setPointWeight(4);      // 4
    poisson.setSamplesPerNode(1.5); // 1.5
    poisson.setScale(1.1);          // 1.1
    poisson.setIsoDivide(8);        // 8
    poisson.setConfidence(1);
    poisson.setManifold(0);
    poisson.setOutputPolygons(0);
    poisson.setSolverDivide(8); // 8
    poisson.reconstruct(mesh);

    // visualize mesh
    pcl::shared_ptr<pcl::visualization::PCLVisualizer> viewer(
        new pcl::visualization::PCLVisualizer("MAP3D MESH"));
    viewer->setBackgroundColor(0, 0, 0);
    viewer->addPolygonMesh(mesh, "meshes", 0);
    // viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();

    std::cout
        << "Press q to finish 3D mapping and start segmentation process..."
        << std::endl;
    while (!viewer->wasStopped()) {
        viewer->spin();
    }

    return true;
}
