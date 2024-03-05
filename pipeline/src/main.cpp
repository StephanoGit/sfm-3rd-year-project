#include <opencv2/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/sfm/triangulation.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <stdexcept>
#include <stdio.h>

#include <boost/program_options.hpp>

#include <pugixml.hpp>

#include "../include/CommonUtil.h"
#include "../include/IOUtil.h"
#include "../include/Segmentation.h"
#include "../include/SfmReconstruction.h"

FeatureExtractionType parse_feature_extraction(std::string fe) {
    if (fe == "SIFT")
        return SIFT;
    else if (fe == "SURF")
        return SURF;
    else if (fe == "BRISK")
        return BRISK;
    else if (fe == "KAZE")
        return KAZE;
    else if (fe == "AKAZE")
        return AKAZE;
    else if (fe == "ORB")
        return ORB;
    else
        throw std::invalid_argument("ERROR: Invalid Feature Detection");
}

FeatureMatchingType parse_feature_matching(std::string fm) {
    if (fm == "FLANN")
        return FLANN;
    else if (fm == "BF")
        return BF;
    else
        throw std::invalid_argument("ERROR: Invalid Feature Detection");
}

int main(int argc, char **argv) {

    std::string input_type;
    std::string directory;
    std::string camera_directory;
    std::string detection_type;
    std::string matching_type;
    bool verbose;
    int downscale_factor;
    boost::program_options::options_description desc("Allowed options");
    desc.add_options()("help,h", "Produce help message")(
        "camera-directory,c",
        boost::program_options::value<std::string>(&camera_directory)
            ->required(),
        "Camera Directory: <../file/path>")(
        "input-type,i",
        boost::program_options::value<std::string>(&input_type)->required(),
        "Input type: video or images")(
        "downscale-factor,s",
        boost::program_options::value<int>(&downscale_factor)->default_value(1),
        "Downscale factor: (int) | 1: no downscale")(
        "directory,d",
        boost::program_options::value<std::string>(&directory)->required(),
        "Directory: <../file/path>")(
        "detection,x",
        boost::program_options::value<std::string>(&detection_type)
            ->default_value("SIFT"),
        "Feature Detection type: SIFT/SURF/FAST/ORB")(
        "matching,m",
        boost::program_options::value<std::string>(&matching_type)
            ->default_value("BF"),
        "Feature Matching type: BF/FLANN")(
        "verbose,v", boost::program_options::bool_switch(&verbose),
        "Verbose (logs and intermediate images)");

    boost::program_options::variables_map vm;
    try {
        boost::program_options::store(
            boost::program_options::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help")) {
            std::cout << desc << "\n";
            return 1;
        }
        boost::program_options::notify(vm);
    } catch (boost::program_options::error &e) {
        std::cerr << "ERROR: " << e.what() << std::endl << std::endl;
        std::cerr << desc << std::endl;
        return 2;
    }

    Intrinsics intrinsics;
    if (!load_camera_intrinsics(camera_directory, intrinsics,
                                downscale_factor)) {
        return 0;
    }

    std::string reconstruction_name =
        directory.substr(directory.find_last_of('/') + 1) + "_" +
        detection_type + "_" + matching_type;

    // start the reconstruction -> generate sparse pointcloud
    // this also generates the dense pointcloud, should call the dense
    // reconstr. in main
    SfmReconstruction reconstruction(directory, reconstruction_name,
                                     parse_feature_extraction(detection_type),
                                     parse_feature_matching(matching_type),
                                     intrinsics, verbose);
    reconstruction.run_sfm_reconstruction(downscale_factor);

    // export the sparse pointcloud
    reconstruction.export_pointcloud_to_PLY("../pointclouds/" +
                                            reconstruction_name);

    // ply file to pcd
    ply_to_pcd("../build/denseCloud/models/options.txt.ply",
               reconstruction_name);

    // pcd to mesh
    pcd_to_mesh("../reconstructions/" + reconstruction_name +
                    "/dense/point_cloud.pcd",
                reconstruction_name);

    return 0;
}
