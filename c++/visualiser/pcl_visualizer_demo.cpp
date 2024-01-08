#include <pcl/console/parse.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h> // Include for PLY support
#include <pcl/visualization/pcl_visualizer.h>

#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_types.h>
#include <pcl/surface/poisson.h>

#include <boost/program_options.hpp>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <thread>

using namespace std::chrono_literals;
namespace po = boost::program_options;

pcl::visualization::PCLVisualizer::Ptr simpleVis(pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud) {
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer->setBackgroundColor(0, 0, 0);
    viewer->addPointCloud<pcl::PointXYZRGB>(cloud, "sample cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
    // viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();
    return (viewer);
}

int main(int argc, char **argv) {
    std::string input_file;
    po::options_description desc("Allowed options");
    desc.add_options()("help,h", "Produce help message")("input-file,f", po::value<std::string>(&input_file)->required(), "Input file path (PLY or JSON)");

    po::variables_map vm;
    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);

        if (vm.count("help")) {
            std::cout << desc << "\n";
            return 0;
        }

        po::notify(vm);
    } catch (po::error &e) {
        std::cerr << "ERROR: " << e.what() << std::endl << std::endl;
        std::cerr << desc << std::endl;
        return 1;
    }

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    // Determine file type from extension and read accordingly
    if (input_file.substr(input_file.find_last_of(".") + 1) == "json") {
        std::ifstream file(input_file);
        if (!file.is_open()) {
            std::cerr << "Failed to open " << input_file << std::endl;
            return 2;
        }

        nlohmann::json j;
        file >> j;
        file.close();

        // Extract points
        std::vector<float> points_3d = j["points"].get<std::vector<float>>();

        for (size_t i = 0; i < points_3d.size(); i += 3) {
            pcl::PointXYZRGB point;
            point.x = points_3d[i];
            point.y = points_3d[i + 1];
            point.z = points_3d[i + 2];
            // Set color (example: white)
            point.r = 255;
            point.g = 255;
            point.b = 255;
            point_cloud->push_back(point);
        }
    } else if (input_file.substr(input_file.find_last_of(".") + 1) == "ply") {
        if (pcl::io::loadPLYFile<pcl::PointXYZRGB>(input_file, *point_cloud) == -1) {
            std::cerr << "Failed to load " << input_file << std::endl;
            return 2;
        }
    } else {
        std::cerr << "Unsupported file format" << std::endl;
        return 3;
    }

    pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne;
    ne.setInputCloud(point_cloud);

    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>());
    ne.setSearchMethod(tree);

    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    ne.setRadiusSearch(0.03);
    ne.compute(*normals);

    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloudWithNormals(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    pcl::concatenateFields(*point_cloud, *normals, *cloudWithNormals);

    pcl::Poisson<pcl::PointXYZRGBNormal> poisson;
    poisson.setDepth(12); // Adjust this as needed
    poisson.setInputCloud(cloudWithNormals);

    pcl::PolygonMesh mesh;
    poisson.reconstruct(mesh);

    pcl::io::savePLYFile("mesh.ply", mesh);

    point_cloud->width = point_cloud->size();
    point_cloud->height = 1;

    pcl::visualization::PCLVisualizer::Ptr viewer = simpleVis(point_cloud);

    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
        std::this_thread::sleep_for(100ms);
    }

    return 0;
}
