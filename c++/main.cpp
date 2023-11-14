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
#include "util.h"
#include "drawUtil.h"
#include "SfmReconstruction.h"

#define IMG_DIR "../images/fountain-P11-rev"

std::vector<cv::Mat> get_Rt_from_E(cv::Mat E)
{
    // Perform Singular Value Decomposition (SVD) on E
    cv::SVD svd(E, cv::SVD::FULL_UV);

    // Extract the matrices U and V
    cv::Mat U = svd.u;
    cv::Mat Vt = svd.vt;

    // Ensure the determinant of U and V is 1 (they should be rotation matrices)
    if (cv::determinant(U) < 0)
    {
        U.col(2) *= -1;
    }
    if (cv::determinant(Vt) < 0)
    {
        Vt.row(2) *= -1;
    }

    // Construct the skew-symmetric matrix W
    cv::Mat W = (cv::Mat_<double>(3, 3) << 0, -1, 0, 1, 0, 0, 0, 0, 1);

    // Compute possible rotation matrices and translation vectors
    cv::Mat R1 = U * W * Vt;
    cv::Mat R2 = U * W.t() * Vt;
    cv::Mat t1 = U.col(2);
    cv::Mat t2 = -U.col(2);

    std::vector<cv::Mat> rts;
    rts.push_back(R1);
    rts.push_back(R2);
    rts.push_back(t1);
    rts.push_back(t2);

    return rts;
}

int main(int argc, char **argv)
{
    std::vector<ImageView> images;
    std::vector<ImagePair> pairs;

    images = load_images(IMG_DIR);

    // compute descriptors and keypoints for each image
    for (int i = 0; i < images.size(); i++)
    {
        images[i].compute_kps_des(FeatureDetectionType::SURF);
        std::cout << "Feature detection done for: " << images[i].get_name() << std::endl;
    }

    // create image pairs (i, i+1)
    for (int i = 0; i < images.size() - 1; i++)
    {
        ImagePair pair(images[i], images[i + 1]);
        pairs.push_back(pair);
    }

    SfmReconstruction reconstruction(pairs);
    std::cout << reconstruction.get_K() << std::endl;

    for (int i = 0; i < pairs.size(); i++)
    {
        pairs[i].match_descriptors(FeatureMatchingType::FLANN);
        cv::Mat matches_image = draw_matches(pairs[i]);
        cv::imshow("Matches", matches_image);

        pairs[i].compute_F();
        cv::Mat matches_image2 = draw_matches(pairs[i]);
        cv::imshow("Matches2", matches_image2);
        cv::waitKey(0);
    }

    return 0;
}