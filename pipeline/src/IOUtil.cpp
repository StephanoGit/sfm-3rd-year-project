#include "../include/IOUtil.h"
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/core/persistence.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>
#include <pugixml.hpp>
#include <stdio.h>

cv::Mat downscale_image(cv::Mat image, int width, int height) {
    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(width, height));
    return resized_image;
}

bool is_image_blurred(cv::Mat image, double threshold) {
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    cv::Mat laplacianImage;
    cv::Laplacian(gray, laplacianImage, CV_64F);

    cv::Scalar mean, stddev;
    cv::meanStdDev(laplacianImage, mean, stddev);

    if (stddev.val[0] * stddev.val[0] < threshold) {
        return true;
    }
    return false;
}

std::vector<cv::Mat> video_to_images(std::string directory, int step,
                                     int downscale_factor) {
    std::vector<cv::Mat> images;
    cv::VideoCapture video(directory);

    if (!video.isOpened()) {
        std::cout << "Cannot open the video file.." << std::endl;
        return images;
    }

    int frameNumber = 0;
    while (video.isOpened()) {
        cv::Mat frame;
        bool success = video.read(frame);
        if (!success) {
            std::cout << "Video finished" << std::endl;
            break;
        }

        if (frameNumber % step == 0 && !is_image_blurred(frame, 10.0)) {
            images.push_back(downscale_image(frame,
                                             frame.cols / downscale_factor,
                                             frame.rows / downscale_factor));
        }
        frameNumber++;
    }

    video.release();
    return images;
}

bool sortByName(const std::__fs::filesystem::directory_entry &entry1,
                const std::__fs::filesystem::directory_entry &entry2) {
    return entry1.path().filename() < entry2.path().filename();
}

std::vector<cv::Mat> load_images(std::string directory, int resize_val,
                                 std::vector<std::string> &images_paths) {

    std::cout << "=======================================" << std::endl;
    std::cout << "            Loading images...          " << std::endl;
    std::cout << "=======================================" << std::endl;

    std::string current_file = "";
    std::vector<cv::Mat> images;
    cv::Mat current_image;
    images_paths.clear();

    std::vector<std::__fs::filesystem::directory_entry> entries;
    for (const auto &entry :
         std::__fs::filesystem::directory_iterator(directory)) {
        entries.push_back(entry);
    }

    // Sort entries by name
    std::sort(entries.begin(), entries.end(), sortByName);

    // get all images within the directory
    for (const auto &entry : entries) {
        current_file = entry.path();
        current_image = cv::imread(current_file, cv::IMREAD_LOAD_GDAL);

        // check image for corrent format or existence
        if (current_image.data == NULL) {
            std::cout << current_file << " -- NOT LOADED [❌]" << std::endl;
            continue;
        }

        images_paths.push_back(current_file);
        // resize
        if (resize_val != 1) {
            current_image =
                downscale_image(current_image, current_image.cols / resize_val,
                                current_image.rows / resize_val);
        }

        // add image to vector
        images.push_back(current_image);

        std::cout << current_file << " -- LOADED [✅]" << std::endl;
    }
    return images;
}

bool load_camera_intrinsics(std::string path, Intrinsics &intrinsics,
                            int downscale_factor) {
    pugi::xml_document intrinsics_file;
    pugi::xml_parse_result intrinsics_file_values =
        intrinsics_file.load_file(path.c_str());

    if (!intrinsics_file_values) {
        std::cout << "Camera Intrinsics file not found, please provide an .xml "
                     "file -- [❌]"
                  << std::endl;
        return false;
    }

    // Parse K values
    float fx = intrinsics_file.child("Intrinsics")
                   .child("K")
                   .child("fx")
                   .text()
                   .as_float() /
               downscale_factor;
    float fy = intrinsics_file.child("Intrinsics")
                   .child("K")
                   .child("fy")
                   .text()
                   .as_float() /
               downscale_factor;
    float cx = intrinsics_file.child("Intrinsics")
                   .child("K")
                   .child("cx")
                   .text()
                   .as_float() /
               downscale_factor;
    float cy = intrinsics_file.child("Intrinsics")
                   .child("K")
                   .child("cy")
                   .text()
                   .as_float() /
               downscale_factor;

    // Parse distortion coefficients
    float k1 = intrinsics_file.child("Intrinsics")
                   .child("D")
                   .child("k1")
                   .text()
                   .as_float();
    float k2 = intrinsics_file.child("Intrinsics")
                   .child("D")
                   .child("k2")
                   .text()
                   .as_float();
    float p1 = intrinsics_file.child("Intrinsics")
                   .child("D")
                   .child("p1")
                   .text()
                   .as_float();
    float p2 = intrinsics_file.child("Intrinsics")
                   .child("D")
                   .child("p2")
                   .text()
                   .as_float();
    float k3 = intrinsics_file.child("Intrinsics")
                   .child("D")
                   .child("k3")
                   .text()
                   .as_float();

    // Create cv::Mat objects with these values
    intrinsics.K = (cv::Mat_<float>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
    intrinsics.d = (cv::Mat_<float>(1, 5) << k1, k2, p1, p2, k3);

    intrinsics.K_inv = intrinsics.K.inv();
    std::cout << "K: " << intrinsics.K << std::endl;
    std::cout << "d: " << intrinsics.d << std::endl;

    return true;
}

void export_point_cloud(std::vector<PointCloudPoint> point_cloud,
                        std::string file_name) {

    std::vector<cv::Point3f> points;
    for (size_t i = 0; i < point_cloud.size(); i++) {
        points.push_back(point_cloud[i].point);
    }

    cv::FileStorage fs("../" + file_name, cv::FileStorage::WRITE);
    fs << "points" << points;
    fs.release();
}
