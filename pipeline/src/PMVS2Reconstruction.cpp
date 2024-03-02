#include "../include/PMVS2Reconstruction.h"
#include "../include/SfmStructures.h"
#include <fstream>
#include <iostream>
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>

void PMVS2Reconstruction::dense_reconstruction(
    std::vector<cv::Mat> &images, std::vector<std::string> &images_paths,
    std::vector<cv::Matx34f> &camera_poses, Intrinsics &intrinsics) {
    /*FOLDERS FOR PMVS2*/
    std::cout << "Creating folders for PMVS2..." << std::endl;
    int dont_care;
    dont_care = std::system("mkdir -p denseCloud/visualize");
    dont_care = std::system("mkdir -p denseCloud/txt");
    dont_care = std::system("mkdir -p denseCloud/models");
    std::cout << "Created: \nfolder:visualize"
              << "\n"
              << "folder:txt"
              << "\n"
              << "folder:models" << std::endl;

    /*OPTIONS CONFIGURATION FILE FOR PMVS2*/
    std::cout << "Creating options file for PMVS2..." << std::endl;
    std::ofstream option("denseCloud/options.txt");
    option << "minImageNum 5" << std::endl;
    option << "CPU 4" << std::endl;
    option << "timages  -1 " << 0 << " " << (images.size() - 1) << std::endl;
    option << "oimages 0" << std::endl;
    option << "level 1" << std::endl;
    option.close();
    std::cout << "Created: options.txt" << std::endl;

    /*CAMERA POSES AND IMAGES INPUT FOR PMVS2*/
    std::cout << "Saving camera poses for PMVS2..." << std::endl;
    std::cout << "Saving camera images for PMVS2..." << std::endl;
    for (int i = 0; i < camera_poses.size(); i++) {
        char str[256];
        boost::filesystem::directory_entry x(images_paths[i]);
        std::string extension = x.path().extension().string();
        boost::algorithm::to_lower(extension);

        std::sprintf(str, "cp -f %s denseCloud/visualize/%04d.jpg",
                     images_paths[i].c_str(), (int)i);
        dont_care = std::system(str);
        cv::imwrite(str, images[i]);

        std::sprintf(str, "denseCloud/txt/%04d.txt", (int)i);
        std::ofstream ofs(str);
        cv::Matx34d pose = camera_poses[i];

        // K*P
        pose = (cv::Matx33d)intrinsics.K * pose;

        ofs << "CONTOUR" << std::endl;
        ofs << pose(0, 0) << " " << pose(0, 1) << " " << pose(0, 2) << " "
            << pose(0, 3) << "\n"
            << pose(1, 0) << " " << pose(1, 1) << " " << pose(1, 2) << " "
            << pose(1, 3) << "\n"
            << pose(2, 0) << " " << pose(2, 1) << " " << pose(2, 2) << " "
            << pose(2, 3) << std::endl;

        ofs << std::endl;
        ofs.close();
    }
    std::cout << "Camera poses saved."
              << "\n"
              << "Camera images saved." << std::endl;
}
