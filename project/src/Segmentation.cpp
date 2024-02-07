#include "../include/Segmentation.h"
#include <cstdlib>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
bool Segmentation::run_segmentation(std::string file_path) {
    pcl::search::Search<pcl::PointXYZRGB>::Ptr tree =
        pcl::shared_ptr<pcl::search::Search<pcl::PointXYZRGB>>(
            new pcl::search::KdTree<pcl::PointXYZRGB>);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(
        new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::io::loadPCDFile(file_path, *cloud);

    std::cout << "=======================================" << std::endl;
    std::cout << "   Colour Based Growing Segmentation   " << std::endl;
    std::cout << "=======================================" << std::endl;

    if (cloud->size() <= 0) {
        std::cout << "ERROR: Failed to load pointcloud -- pcd file empty [❌]"
                  << std::endl;
        return false;
    }

    std::cout << "Setting up options..." << std::endl;

    pcl::IndicesPtr indices(new std::vector<int>);
    pcl::PassThrough<pcl::PointXYZRGB> pass;
    pass.setInputCloud(cloud);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(0.0, 14.0);
    pass.filter(*indices);

    pcl::RegionGrowingRGB<pcl::PointXYZRGB> reg;
    const auto distace_treshold = 10;
    const auto point_color_threshold = 6;
    const auto region_color_threshold = 5;
    const auto min_cluster_size = 2;
    reg.setInputCloud(cloud);
    reg.setIndices(indices);
    reg.setSearchMethod(tree);
    reg.setDistanceThreshold(distace_treshold);          // 10
    reg.setPointColorThreshold(point_color_threshold);   // 6
    reg.setRegionColorThreshold(region_color_threshold); // 5
    reg.setMinClusterSize(min_cluster_size);             // 600

    std::cout << "Input cloud:" << cloud->size() << "\n"
              << "Distance threshold:" << distace_treshold << "\n"
              << "Point color threshold:" << point_color_threshold << "\n"
              << "Region color threshold:" << region_color_threshold << "\n"
              << "Clusters size:" << min_cluster_size << std::endl;

    std::cout << "Extracting clusters..." << std::endl;
    std::vector<pcl::PointIndices> clusters;
    reg.extract(clusters);
    if (clusters.size() <= 0) {
        std::cerr << "Error: could not extract enough clusters." << std::endl;
        std::cout << "Extract:" << clusters.size() << " clusters. Min=600"
                  << std::endl;
        return false;
    }
    std::cout << "Extract:" << clusters.size() << " clusters" << std::endl;

    std::cout << "Getting color cloud..." << std::endl;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud =
        reg.getColoredCloud();

    std::cout << "Showing 3D Mapping segmented..." << std::endl;

    pcl::visualization::PCLVisualizer viewer =
        pcl::visualization::PCLVisualizer("MAP3D Segmented", true);
    viewer.addPointCloud(colored_cloud);
    std::cout << "Press q to finish segmentation proccess" << std::endl;
    while (!viewer.wasStopped()) {
        viewer.spin();
    }

    return true;
}
