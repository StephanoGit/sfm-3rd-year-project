#include "CameraCalibration.h"

#include <fstream>
#include <opencv2/core/types.hpp>
#include <stdio.h>

#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <pugixml.hpp>

#include "../pipeline/include/IOUtil.h"

CameraCalibration::CameraCalibration(std::string directory, bool show_images,
                                     int resize_val, std::string file_name)
    : checkerboard_size({13, 9}), square_size(20) {
    calibrateCamera(directory, checkerboard_size, square_size, show_images,
                    resize_val, file_name);
}

void CameraCalibration::calibrateCamera(std::string directory,
                                        std::vector<int> checkerboard_size,
                                        int square_size, bool show_images,
                                        int resize_val, std::string file_name) {
    std::cout << "=======================================" << std::endl;
    std::cout << "         Calibrating camera...         " << std::endl;
    std::cout << "=======================================" << std::endl;

    std::vector<std::string> images_paths;
    std::vector<cv::Mat> images =
        load_images(directory, resize_val, images_paths);
    std::vector<std::vector<cv::Point2f>> q(images.size());
    std::vector<std::vector<cv::Point3f>> Q;
    std::vector<cv::Point3f> world_points;

    for (int i = 0; i < checkerboard_size[1]; i++) {
        for (int j = 0; j < checkerboard_size[0]; j++) {
            world_points.push_back(
                cv::Point3f(j * square_size, i * square_size, 0));
        }
    }

    cv::Mat frame, gray;

    std::vector<cv::Point2f> image_points;
    bool pattern_found;

    for (int i = 0; i < images.size(); i++) {
        cv::cvtColor(images[i], gray, cv::COLOR_BGR2GRAY);

        pattern_found = cv::findChessboardCorners(
            gray, cv::Size(checkerboard_size[0], checkerboard_size[1]), q[i],
            cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE +
                cv::CALIB_CB_FAST_CHECK);
        if (pattern_found) {
            cv::cornerSubPix(gray, q[i], cv::Size(11, 11), cv::Size(-1, -1),
                             cv::TermCriteria(cv::TermCriteria::EPS +
                                                  cv::TermCriteria::MAX_ITER,
                                              30, 0.1));
            Q.push_back(world_points);
            std::cout << "Pattern found in IMG " << i << std::endl;
        }

        if (show_images) {
            // Manually draw corners
            for (const auto &point : q[i]) {
                cv::circle(images[i], point, 20, cv::Scalar(120, 255, 0), 10);
            }
            for (int j = 0; j < checkerboard_size[1] - 1; j++) {
                for (int k = 0; k < checkerboard_size[0]; k++) {
                    // Current point
                    cv::Point current = q[i][j * checkerboard_size[0] + k];

                    // Draw vertical line to the point below
                    cv::Point below = q[i][(j + 1) * checkerboard_size[0] + k];
                    cv::line(images[i], current, below, cv::Scalar(0, 255, 0),
                             10);
                }
            }

            // Connect the last element of column i to the first corner of
            // column i+1
            for (int k = 0; k < checkerboard_size[0] - 1; k++) {
                cv::Point last_of_column =
                    q[i][(checkerboard_size[1] - 1) * checkerboard_size[0] + k];
                cv::Point first_of_next_column = q[i][k + 1];
                cv::line(images[i], last_of_column, first_of_next_column,
                         cv::Scalar(0, 255, 0), 10);
            }
            // cv::drawChessboardCorners(images[i],
            // cv::Size(checkerboard_size[0], checkerboard_size[1]), q[i],
            // pattern_found);
            cv::imshow("Chekerboard", images[i]);
            cv::waitKey(0);
        }
    }

    cv::Matx33f K(cv::Matx33f::eye());
    cv::Vec<float, 5> d(0, 0, 0, 0, 0);

    std::vector<cv::Mat> rvecs, tvecs;
    int flags = cv::CALIB_FIX_ASPECT_RATIO + cv::CALIB_FIX_K3 +
                cv::CALIB_ZERO_TANGENT_DIST + cv::CALIB_FIX_PRINCIPAL_POINT;
    cv::Size frame_size(images[0].cols, images[0].rows);

    std::cout << "Calibrating..." << std::endl;
    float error = cv::calibrateCamera(Q, q, frame_size, K, d, rvecs, tvecs);

    this->K = cv::Mat(K);
    this->d = d;
    this->error = error;

    std::cout << "Reprojection error: " << error << std::endl;
    std::cout << "K: " << K << std::endl;
    std::cout << "d: " << d << std::endl;

    if (show_images) {
        cv::Mat mapX, mapY;
        cv::initUndistortRectifyMap(K, d, cv::Matx33f::eye(), K, frame_size,
                                    CV_32FC1, mapX, mapY);

        for (int i = 0; i < images.size(); i++) {
            cv::Mat image_undistorted;
            cv::remap(images[i], image_undistorted, mapX, mapY,
                      cv::INTER_LINEAR);

            cv::imshow("Undistorted images", image_undistorted);
            cv::waitKey(0);
        }
    }
    export_to_xml(file_name, this->K, this->d);
}

void CameraCalibration::export_to_xml(const std::string &file_name,
                                      const cv::Mat &K,
                                      const std::vector<float> &d) {
    pugi::xml_document doc;

    // Create root and main nodes
    auto intrinsics_node = doc.append_child("Intrinsics");
    auto K_node = intrinsics_node.append_child("K");
    auto d_node = intrinsics_node.append_child("D");

    // Append K values
    K_node.append_child("fx").text().set(K.at<float>(0, 0));
    K_node.append_child("fy").text().set(K.at<float>(1, 1));
    K_node.append_child("cx").text().set(K.at<float>(0, 2));
    K_node.append_child("cy").text().set(K.at<float>(1, 2));

    // Append d values
    d_node.append_child("k1").text().set(d[0]);
    d_node.append_child("k2").text().set(d[1]);
    d_node.append_child("p1").text().set(d[2]);
    d_node.append_child("p2").text().set(d[3]);
    d_node.append_child("k3").text().set(d[4]);

    // Save XML document to file
    doc.save_file(file_name.c_str());
}
cv::Mat downscale_image(cv::Mat image, int width, int height) {
    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(width, height));
    return resized_image;
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
CameraCalibration::~CameraCalibration(){};
