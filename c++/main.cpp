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
#define VID_DIR "../videos/test1.MOV"

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

void transformPointCloud(const std::vector<cv::Point3f> &src, std::vector<cv::Point3f> &dst, const cv::Mat &R, const cv::Mat &t)
{
    dst.clear();
    for (const auto &pt : src)
    {
        cv::Mat ptMat = (cv::Mat_<double>(3, 1) << pt.x, pt.y, pt.z);
        cv::Mat ptT = R * ptMat + t;
        dst.push_back(cv::Point3f(ptT.at<double>(0, 0), ptT.at<double>(1, 0), ptT.at<double>(2, 0)));
    }
}

int main(int argc, char **argv)
{
    std::vector<ImageView> images;
    std::vector<ImagePair> pairs;

    images = extract_frames_from_video(VID_DIR, 50);

    // for (int i = 0; i < images.size(); i++)
    // {
    //     cv::imshow("frame", images[i].get_image());
    //     cv::waitKey(0);
    // }

    ImagePair pair(images)

        // images = load_images(IMG_DIR);

        // compute descriptors and keypoints for each image
        // for (int i = 0; i < images.size(); i++)
        // {
        //     images[i].compute_kps_des(FeatureDetectionType::SIFT);
        //     std::cout << "Feature detection done for: " << images[i].get_name() << std::endl;
        // }

        // // create image pairs (i, i+1)
        // for (int i = 0; i < images.size() - 1; i++)
        // {
        //     ImagePair pair(images[i], images[i + 1]);
        //     pairs.push_back(pair);
        // }

        // SfmReconstruction reconstruction(pairs);
        // // reconstruction.triangulation();

        // cv::Mat K = reconstruction.get_K();
        // std::vector<double> d = reconstruction.get_distortion();
        // std::cout << K << std::endl;

        // for (int i = 0; i < pairs.size(); i++)
        // {
        //     pairs[i].match_descriptors(FeatureMatchingType::FLANN);
        //     // cv::Mat matches_image = draw_matches(pairs[i]);
        //     // cv::imshow("Matches", matches_image);

        //     pairs[i].compute_F();
        //     // cv::Mat matches_image2 = draw_matches(pairs[i]);
        //     // cv::imshow("Matches2", matches_image2);

        //     pairs[i].compute_E(K);
        //     // cv::Mat matches_image3 = draw_matches(pairs[i]);
        //     // cv::imshow("Matches3", matches_image3);

        //     pairs[i].compute_Rt(K);

        //     pairs[i].triangulate(K, i);

        //     cv::waitKey(0);
        // }

        // ImagePair pair01(images[0], images[1]);
        // pair01.match_descriptors(FeatureMatchingType::FLANN);
        // pair01.compute_F();
        // pair01.compute_E(K);
        // pair01.compute_Rt(K);
        // pair01.triangulate(K, 34);

        // ImagePair pair12(images[1], images[2]);
        // pair12.match_descriptors(FeatureMatchingType::FLANN);
        // pair12.compute_F();
        // pair12.compute_E(K);
        // pair12.compute_Rt(K);

        // // Find corresponding 2D-3D points
        // std::vector<cv::Point2f> points2D;
        // std::vector<cv::Point3f> points3D_correspondences;
        // for (const auto &match : pair12.get_good_matches())
        // {
        //     int img2_idx = match.queryIdx;
        //     int img3_idx = match.trainIdx;

        //     // Check if the feature in the previous image was triangulated
        //     if (img2_idx < pair01.get_points_3d().rows)
        //     {
        //         points2D.push_back(pair12.get_image2_good_kps()[img3_idx].pt);
        //         cv::Point3f pt3D(pair01.get_points_3d().at<float>(img2_idx, 0),
        //                          pair01.get_points_3d().at<float>(img2_idx, 1),
        //                          pair01.get_points_3d().at<float>(img2_idx, 2));
        //         points3D_correspondences.push_back(pt3D);
        //     }
        // }

        // cv::Mat rvec, tvec, mask, R_new;
        // cv::solvePnPRansac(points3D_correspondences, points2D, K, d, rvec, tvec);
        // cv::Rodrigues(rvec, R_new);

        return 0;
}