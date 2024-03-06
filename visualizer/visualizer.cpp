#include <boost/program_options.hpp>
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
namespace po = boost::program_options;

void display_mesh(const pcl::PolygonMesh &mesh) {
    if (mesh.polygons.empty()) {
        std::cout << "ERROR: Failed to load mesh file -- [❌]" << std::endl;
        return;
    }

    pcl::shared_ptr<pcl::visualization::PCLVisualizer> viewer(
        new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer->setBackgroundColor(0, 0, 0);
    viewer->addPolygonMesh(mesh, "meshes");
    viewer->initCameraParameters();

    std::cout << "Press q to exit." << std::endl;
    while (!viewer->wasStopped()) {
        viewer->spin();
    }
}

void display_point_cloud(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud) {
    if (cloud->points.empty()) {
        std::cout << "ERROR: Failed to load point cloud -- [❌]" << std::endl;
        return;
    }

    pcl::shared_ptr<pcl::visualization::PCLVisualizer> viewer(
        new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer->setBackgroundColor(0, 0, 0);
    viewer->addPointCloud<pcl::PointXYZRGB>(cloud, "cloud");
    viewer->initCameraParameters();

    std::cout << "Press q to exit." << std::endl;
    while (!viewer->wasStopped()) {
        viewer->spin();
    }
}

int main(int argc, char **argv) {
    std::string input_file;
    std::string file_type;
    // Define and parse the program options
    po::options_description desc("Options");
    desc.add_options()("help,h", "Print help messages")(
        "input-file,i", po::value<std::string>(&input_file)->required(),
        "Input file")("type,t", po::value<std::string>(&file_type)->required(),
                      "File type (ply, pcd, mesh)");

    po::variables_map vm;
    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);

        if (vm.count("help")) {
            std::cout << "Basic Command Line Parameter App" << std::endl
                      << desc << std::endl;
            return 0;
        }

        po::notify(vm); // throws on error, so do after help in case there are
                        // any problems

    } catch (po::error &e) {
        std::cerr << "ERROR: " << e.what() << std::endl << std::endl;
        std::cerr << desc << std::endl;
        return 1;
    }

    if (file_type == "PLY" || file_type == "ply" || file_type == "PCD" ||
        file_type == "pcd") {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(
            new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::io::loadPLYFile<pcl::PointXYZRGB>(input_file, *cloud);
        display_point_cloud(cloud);
    } else if (file_type == "MESH" || file_type == "mesh") {
        pcl::PolygonMesh mesh;
        pcl::io::loadPLYFile(input_file, mesh);
        display_mesh(mesh);
    }

    return 0;
}
