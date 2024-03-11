#ifndef __STEREO_UTIL
#define __STEREO_UTIL

#include "../include/CommonUtil.h"
#include "SfmStructures.h"
#include <opencv2/core/matx.hpp>
#include <opencv2/core/types.hpp>

#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/xfeatures2d.hpp>
class StereoUtil {
public:
    StereoUtil();
    virtual ~StereoUtil();

    static bool camera_matrices_from_matches(
        const Intrinsics &intrinsics, const std::vector<cv::DMatch> &matches,
        const Features &features_left, const Features &features_right,
        std::vector<cv::DMatch> &mask_matches, cv::Matx34f &P_left,
        cv::Matx34f &P_right);

    static int homography_inliers(const Features &features_left,
                                  const Features &features_right,
                                  const std::vector<cv::DMatch> &matches);
    static bool
    remove_homography_outliers(const Features &features_left,
                               const Features &features_right,
                               const std::vector<cv::DMatch> &matches,
                               std::vector<cv::DMatch> &mask_matches);

    static bool triangulate_views(
        const Intrinsics &intrinsics, const ImagePair image_pair,
        const std::vector<cv::DMatch> &matches, const Features &features_left,
        const Features &features_right, const cv::Matx34f &P_left,
        const cv::Matx34f &P_right, std::vector<PointCloudPoint> &point_cloud);

    static bool P_from_2D3D_matches(const Intrinsics &intrinsics,
                                    const Image2D3DPair &match, cv::Matx34f &P);
};

#endif
