#include <pcl/console/parse.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <thread>

using namespace std::chrono_literals;

pcl::visualization::PCLVisualizer::Ptr simpleVis(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud) {
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer->setBackgroundColor(0, 0, 0);
    viewer->addPointCloud<pcl::PointXYZ>(cloud, "sample cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
    // viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();
    return (viewer);
}

int main(int argc, char **argv) {
    std::ifstream file("../../points-3d/homoo.json");
    nlohmann::json j;
    file >> j;
    file.close();

    // Extract points
    std::vector<float> points_3d = j["points"].get<std::vector<float>>();

    // Create a new point cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud(new pcl::PointCloud<pcl::PointXYZ>);

    for (size_t i = 0; i < points_3d.size(); i += 3) {
        pcl::PointXYZ point;
        point.x = points_3d[i] * 100;
        point.y = points_3d[i + 1] * 100;
        point.z = points_3d[i + 2] * 100;
        point_cloud->push_back(point);
    }

    point_cloud->width = point_cloud->size();
    point_cloud->height = 1;

    pcl::visualization::PCLVisualizer::Ptr viewer;
    viewer = simpleVis(point_cloud);

    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
        std::this_thread::sleep_for(100ms);
    }
}
