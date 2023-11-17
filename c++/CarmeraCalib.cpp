#include <stdio.h>
#include <string>
#include <iostream>
#include <filesystem>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include <opencv2/sfm/triangulation.hpp>

#include "ImageView.h"
#include "ImagePair.h"
#include "util.h"
#include "drawUtil.h"
#include "SfmReconstruction.h"

void calibCamera(cv::Mat &R, cv::Mat &T, cv::Mat &cameraMatrix, cv::Mat &distCoeffs)
{
    int CHECKERBOARD[2]{7, 9};
    // int fieldSize = 25;

    std::vector<std::vector<cv::Point3f>> objpoints;
    std::vector<std::vector<cv::Point2f>> imgpoints;

    std::vector<cv::Point3f> objp;

    // Defining the world coordinates for 3D points
    for (int i = 0; i < CHECKERBOARD[1]; i++)
    {
        for (int j = 0; i < CHECKERBOARD[0]; j++)
        {
            objp.push_back(cv::Point3f(j, i, 0)); // maybe j*fieldsize and i*fieldsize
        }
    }
    std::cout << "SDSDS";
    std::vector<cv::String> images;

    cv::glob("./checkerboardImgs/*.jpg", images);

    cv::Mat frame, gray;
    std::vector<cv::Point2f> corner_points;
    bool patternFound;

    // Looping over all the images in the directory
    for (auto const &image : images)
    {
        frame = cv::imread(image);
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        patternFound = cv::findChessboardCorners(gray, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_points, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE);

        if (patternFound)
        {
            cv::TermCriteria criteria(cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER, 30, 0.001);
            cv::cornerSubPix(gray, corner_points, cv::Size(11, 11), cv::Size(-1, -1), criteria);
            objpoints.push_back(objp);
            imgpoints.push_back(corner_points);
        }

        // Display
        cv::drawChessboardCorners(frame, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_points, patternFound);
        cv::imshow("Checkerboard", frame);
        cv::waitKey(0);
    }

    cv::destroyAllWindows();

    // cv::Mat cameraMatrix,distCoeffs,R,T;

    cv::calibrateCamera(objpoints, imgpoints, cv::Size(gray.rows, gray.cols), cameraMatrix, distCoeffs, R, T);
}