#include "../include/PMVS2Reconstruction.h"
#include "../include/SfmStructures.h"
#include <fstream>
#include <iostream>
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>

namespace fs = std::filesystem;

void PMVS2Reconstruction::dense_reconstruction(
    std::vector<cv::Mat> &images, std::vector<std::string> &image_paths,
    std::vector<cv::Matx34f> &camera_poses, const Intrinsics &intrinsics) {

    /* Folders for PMVS2 */
    fs::create_directories("denseCloud/visualize");
    fs::create_directories("denseCloud/txt");
    fs::create_directories("denseCloud/models");

    /* Options configuration file for PMVS2 */
    std::ofstream option("denseCloud/options.txt");
    option << "minImageNum 5" << std::endl
           << "CPU 4" << std::endl
           << "timages  -1 " << 0 << " " << (images.size() - 1) << std::endl
           << "oimages 0" << std::endl
           << "level 1" << std::endl;
    option.close();

    /* Camera poses and images input for PMVS2 */
    for (size_t i = 0; i < camera_poses.size(); ++i) {
        // Saving images with zero-padded numbering
        std::ostringstream imgStream;
        imgStream << "denseCloud/visualize/" << std::setw(4)
                  << std::setfill('0') << i << ".jpg";
        cv::imwrite(imgStream.str(), images[i]);

        // Saving camera poses with zero-padded numbering
        std::ostringstream poseStream;
        poseStream << "denseCloud/txt/" << std::setw(4) << std::setfill('0')
                   << i << ".txt";
        std::ofstream ofs(poseStream.str());
        cv::Matx34d pose = camera_poses[i];
        pose = static_cast<cv::Matx33d>(intrinsics.K) *
               pose; // Assuming intrinsics.K is cv::Matx33f or cv::Matx33d

        ofs << "CONTOUR" << std::endl;
        for (int row = 0; row < pose.rows; ++row) {
            for (int col = 0; col < pose.cols; ++col) {
                ofs << pose(row, col) << " ";
            }
            ofs << std::endl;
        }
        ofs.close();
    }
}
