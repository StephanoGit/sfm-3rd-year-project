#include <stdio.h>

#include <opencv2/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/sfm/triangulation.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "../calibration/CameraCalibration.h"
#include "../include/ImagePair.h"
#include "../include/ImageView.h"
#include "../include/SfmReconstruction.h"
#include "../include/drawUtil.h"
#include "../include/util.h"

#define CHECKERBOARD_DIR "../calibration/images"
#define IMG_DIR "../images/fountain-P11-rev"
#define VID_DIR "../videos/test1.MOV"

int main(int argc, char **argv) {
    // CameraCalibration(CHECKERBOARD_DIR, false);

    if (argc != 5) {
        std::cout << "Please specify: " << std::endl;
        std::cout << "(1) camera:        iphone or dev" << std::endl;
        std::cout << "(2) input type:    video  or images" << std::endl;
        std::cout << "(3) resize:        true   or false" << std::endl;
        std::cout << "(4) directory:     <../file/path>" << std::endl;
        return 0;
    }

    bool resize;
    std::string camera(argv[1]);
    std::string input_type(argv[2]);
    std::string resize_val(argv[3]);
    std::string directory(argv[4]);

    if (resize_val == "true") {
        resize = true;
    } else if (resize_val == "false") {
        resize = false;
    } else {
        std::cout << "Please provide a valid value for resizing..."
                  << std::endl;
    }

    std::vector<ImageView> images;
    if (input_type == "video") {
        images = extract_frames_from_video(VID_DIR, 50);
    } else if (input_type == "images") {
        images = load_images_as_object(directory, resize);
    } else {
        std::cout << "Please provide a valid value for input type..."
                  << std::endl;
    }

    SfmReconstruction reconstruction(images, FeatureDetectionType::SIFT,
                                     FeatureMatchingType::FLANN);
    return 0;
}