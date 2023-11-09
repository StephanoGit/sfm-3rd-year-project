#include "util.h"

#include <stdio.h>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#define WIDTH 3072
#define HEIGHT 2048
#define NEW_WIDTH 640
#define NEW_HEIGHT 480


float new_fx = (2759.48 * NEW_WIDTH) / WIDTH;
float new_fy = (2764.16 * NEW_HEIGHT) / HEIGHT;
float new_cx = (1520.69 * NEW_WIDTH) / WIDTH;
float new_cy = (1006.81 * NEW_HEIGHT) / HEIGHT;


float data[9] = {new_fx, 0     , new_cx,
                0     , new_fy, new_cy,
                0     , 0     , 1      };
cv::Mat K(3, 3, CV_32F, data);




cv::Mat compute_K(){
    return K;
}

cv::Mat compute_F(ImagePair pair){
    cv::Mat F = cv::findFundamentalMat(pair.get_img1_good_matches(), pair.get_img2_good_matches(), cv::FM_RANSAC, 3.0, 0.99);
    return F;
}

cv::Mat compute_E(cv::Mat K, cv::Mat F){
    // cv::Mat E = K.t() * F * K;
    return K;
}

bool sortByName(const std::__fs::filesystem::directory_entry &entry1, const std::__fs::filesystem::directory_entry &entry2) {
    return entry1.path().filename() < entry2.path().filename();
}

std::vector<ImageView> load_images(std::string directory){
    std::string current_file = "";
    std::vector<ImageView> images;
    ImageView current_image;

    std::vector<std::__fs::filesystem::directory_entry> entries;
    for (const auto &entry : std::__fs::filesystem::directory_iterator(directory)) {
        entries.push_back(entry);
    }

    // Sort entries by name
    std::sort(entries.begin(), entries.end(), sortByName);

    // get all images within the directory
    for (const auto & entry : entries){
        current_file = entry.path();
        current_image.set_name(current_file.substr(10));

        // might have to change the image colour channels
        current_image.set_image(cv::imread(current_file, cv::IMREAD_UNCHANGED));

        // check image for corrent format or existence
        if (current_image.get_image().data == NULL){
            printf("Image not found or incorrect file type!\n");
            continue;
        }

        //resize
        cv::Mat original_image = current_image.get_image();
        cv::Mat resized_image;
        cv::resize(original_image, resized_image, cv::Size(NEW_WIDTH, NEW_HEIGHT));
        current_image.set_image(resized_image);


        // add image to vector
        images.push_back(current_image);
    }

    return images;
}


