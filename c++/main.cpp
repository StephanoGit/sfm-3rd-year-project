#include <stdio.h>
#include <string>
#include <iostream>
#include <filesystem>
#include <vector>


#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "ImageView.h"
#include "ImagePair.h"

#define IMG_DIR "../images"

std::vector<ImageView> load_images(std::string directory){
    std::string current_file = "";
    std::vector<ImageView> images;
    ImageView current_image;

    // get all images within the directory
    for (const auto & entry : std::__fs::filesystem::directory_iterator(directory)){
        current_file = entry.path();
        current_image.set_name(current_file.substr(10));

        // might have to change the image colour channels
        current_image.set_image(cv::imread(current_file, cv::IMREAD_UNCHANGED));

        // check image for corrent format or existence
        if (current_image.get_image().data == NULL){
            printf("Image not found or incorrect file type!\n");
            continue;
        }

        // add image to vector
        images.push_back(current_image);
    }

    return images;
}



int main(int argc, char **argv){
    std::vector <ImageView> images_raw;

    images_raw = load_images(IMG_DIR);

    for(int i = 0; i < images_raw.size();i++){
        images_raw[i].compute_kps_des(FeatureDetectionType::ORB);
        std::cout << "Feature detection done for: " << images_raw[i].get_name() << "\n";
    }

    ImagePair pair_12(images_raw[0], images_raw[1]);
    pair_12.match_descriptors(FeatureMatchingType::FLANN);
    std::cout << "Descriptors matched for: " << pair_12.get_image1().get_name() << " and " << pair_12.get_image2().get_name() <<"\n";

    // computeKeyPointsDescriptors(imagesRaw, ORB);
    // matches = matchDescriptors(imagesRaw[0].descriptors, imagesRaw[1].descriptors, cv::DescriptorMatcher::BRUTEFORCE_HAMMING);

    pair_12.draw_matches();
    std::cout << "Images matched: " << pair_12.get_image1().get_name() << " and " << pair_12.get_image2().get_name() <<"\n";


    cv::imshow("Good matches", pair_12.get_matches_image());
    cv::waitKey();
    return 0;
}