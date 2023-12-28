#include <stdio.h>

#include <opencv2/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/sfm/triangulation.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include <boost/program_options.hpp>

#include "../calibration/CameraCalibration.h"
#include "../include/IOUtil.h"
#include "../include/PlottingUtil.h"
#include "../include/SfmReconstruction.h"

#define CHECKERBOARD_DIR "../calibration/images/iphone 15 pro"
#define IMG_DIR "../images/fountain-P11-rev"
#define VID_DIR "../videos/test1.MOV"

int main(int argc, char **argv) {
    // CameraCalibration(CHECKERBOARD_DIR, true, false);

    std::string input_type;
    std::string directory;
    std::string camera_type;
    int downscale_factor;
    boost::program_options::options_description desc("Allowed options");
    desc.add_options()("help,h", "Produce help message")("camera-type,c", boost::program_options::value<std::string>(&camera_type)->required(), "Camera type: iphone/default")(
        "input-type,i", boost::program_options::value<std::string>(&input_type)->required(),
        "Input type: video or images")("downscale-factor,s", boost::program_options::value<int>(&downscale_factor)->default_value(1),
                                       "Downscale factor: (int) | 1: no downscale")("directory,d", boost::program_options::value<std::string>(&directory)->required(), "Directory: <../file/path>");

    boost::program_options::variables_map vm;
    try {
        boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);

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

    if (camera_type == "default") {
        intrinsics.K = (cv::Mat_<float>(3, 3) << 2759.48, 0, 1520.69, 0, 2764.16, 1006.81, 0, 0, 1);
        intrinsics.d = cv::Mat_<float>::zeros(1, 4);
    } else {
        intrinsics.K = (cv::Mat_<float>(3, 3) << 4020.5203 / downscale_factor, 0, 2141.5 / downscale_factor, 0, 4020.5203 / downscale_factor, 2855.5 / downscale_factor, 0, 0, 1);
        intrinsics.d = (cv::Mat_<float>(1, 4) << 0, 0, 0, 0);
    }

    intrinsics.K_inv = intrinsics.K.inv();

    SfmReconstruction reconstruction(directory, FeatureExtractionType::SIFT, FeatureMatchingType::BF, intrinsics);
    reconstruction.run_sfm_reconstruction(downscale_factor);

    return 0;
}
