#include "CameraCalibration.h"

#include <stdio.h>

#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include "../include/util.h"

CameraCalibration::CameraCalibration(std::string directory, bool show_images) : checkerboard_size({6, 8}), square_size(23) {
    calibrateCamera(directory, checkerboard_size, square_size, show_images);
}

void CameraCalibration::calibrateCamera(std::string directory, std::vector<int> checkerboard_size, int square_size, bool show_images) {
    std::vector<cv::Mat> images = load_images(directory);
    std::vector<std::vector<cv::Point2f>> q(images.size());

    std::vector<std::vector<cv::Point3f>> Q;
    std::vector<cv::Point3f> world_points;

    for (int i = 0; i < checkerboard_size[1]; i++) {
        for (int j = 0; j < checkerboard_size[0]; j++) {
            world_points.push_back(cv::Point3f(j * square_size, i * square_size, 0));
        }
    }

    cv::Mat frame, gray;

    std::vector<cv::Point2f> image_points;
    bool pattern_found;

    for (int i = 0; i < images.size(); i++) {
        cv::cvtColor(images[i], gray, cv::COLOR_BGR2GRAY);

        pattern_found = cv::findChessboardCorners(gray, cv::Size(checkerboard_size[0], checkerboard_size[1]),
                                                  q[i],
                                                  cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_FAST_CHECK);
        if (pattern_found) {
            cv::cornerSubPix(gray, q[i], cv::Size(11, 11), cv::Size(-1, -1),
                             cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1));
            Q.push_back(world_points);
        }

        if (show_images) {
            cv::drawChessboardCorners(images[i], cv::Size(checkerboard_size[0], checkerboard_size[1]),
                                      q[i], pattern_found);
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
    float error = cv::calibrateCamera(Q, q, frame_size, K, d, rvecs, tvecs, flags);

    this->K = cv::Mat(K);
    this->d = d;
    this->error = error;

    if (show_images) {
        std::cout << "Reprojection error: " << error << std::endl;
        std::cout << "K: " << K << std::endl;
        std::cout << "d: " << d << std::endl;

        cv::Mat mapX, mapY;
        cv::initUndistortRectifyMap(K, d, cv::Matx33f::eye(), K, frame_size, CV_32FC1, mapX, mapY);

        for (int i = 0; i < images.size(); i++) {
            cv::Mat image_undistorted;
            cv::remap(images[i], image_undistorted, mapX, mapY, cv::INTER_LINEAR);

            cv::imshow("Undistorted images", image_undistorted);
            cv::waitKey(0);
        }
    }

    export_K_to_xml("K", this->K);
    export_K_to_json("K", this->K);
}

cv::Mat CameraCalibration::get_K() {
    return this->K;
}
cv::Vec<float, 5> CameraCalibration::get_d() {
    return this->d;
}

float CameraCalibration::get_error() {
    return this->error;
}

CameraCalibration::~CameraCalibration(){};