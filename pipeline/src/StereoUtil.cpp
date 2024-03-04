#include "../include/StereoUtil.h"

StereoUtil::StereoUtil() {}
StereoUtil::~StereoUtil() {}

int StereoUtil::homography_inliers(const Features &features_left,
                                   const Features &features_right,
                                   const std::vector<cv::DMatch> &matches) {
    Features aligned_left;
    Features aligned_right;
    std::vector<int> left_origin, right_origin;
    align_points_from_matches(features_left, features_right, matches,
                              aligned_left, aligned_right, left_origin,
                              right_origin);

    cv::Mat mask;
    cv::Mat H;
    if (matches.size() >= 4) {
        H = cv::findHomography(aligned_left.points, aligned_right.points,
                               cv::RANSAC, 10.0, mask);
    }

    if (matches.size() < 4 || H.empty()) {
        std::cout
            << "LOG: Not enough matches were found using Homography -- [❌]"
            << std::endl;
        return 0;
    }

    int inliers = cv::countNonZero(mask);
    std::cout << "LOG: Found " << inliers << " using Homography -- [✅]"
              << std::endl;
    return inliers;
}

bool StereoUtil::camera_matrices_from_matches(
    const Intrinsics &intrinsics, const std::vector<cv::DMatch> &matches,
    const Features &features_left, const Features &features_right,
    std::vector<cv::DMatch> &mask_matches, cv::Matx34f &P_left,
    cv::Matx34f &P_right) {
    if (intrinsics.K.empty()) {
        std::cout << "ERROR: Intrinsic matrix K is empty..." << std::endl;
        return false;
    }

    double focal_length = intrinsics.K.at<float>(0, 0);
    cv::Point2d principal_point(intrinsics.K.at<float>(0, 2),
                                intrinsics.K.at<float>(1, 2));

    Features aligned_left, aligned_right;
    std::vector<int> left_origin, right_origin;
    align_points_from_matches(features_left, features_right, matches,
                              aligned_left, aligned_right, left_origin,
                              right_origin);

    cv::Mat E, R, t;
    cv::Mat mask;
    E = cv::findEssentialMat(aligned_left.points, aligned_right.points,
                             focal_length, principal_point, cv::RANSAC, 0.999,
                             1.0, mask);

    cv::recoverPose(E, aligned_left.points, aligned_right.points, R, t,
                    focal_length, principal_point, mask);

    P_left = cv::Matx34f::eye();
    P_right =
        cv::Matx34f(R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2),
                    t.at<double>(0), R.at<double>(1, 0), R.at<double>(1, 1),
                    R.at<double>(1, 2), t.at<double>(1), R.at<double>(2, 0),
                    R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2));

    remove_outliers(matches, mask, mask_matches);
    return true;
}

bool StereoUtil::triangulate_views(
    const Intrinsics &intrinsics, const ImagePair image_pair,
    const std::vector<cv::DMatch> &matches, const Features &features_left,
    const Features &features_right, const cv::Matx34f &P_left,
    const cv::Matx34f &P_right, std::vector<PointCloudPoint> &point_cloud) {

    Features aligned_left, aligned_right;
    std::vector<int> left_origin, right_origin;
    align_points_from_matches(features_left, features_right, matches,
                              aligned_left, aligned_right, left_origin,
                              right_origin);

    cv::Mat norm_left, norm_right;
    cv::undistortPoints(aligned_left.points, norm_left, intrinsics.K,
                        intrinsics.d);
    cv::undistortPoints(aligned_right.points, norm_right, intrinsics.K,
                        intrinsics.d);

    cv::Mat points_4D;
    cv::triangulatePoints(P_left, P_right, norm_left, norm_right, points_4D);

    cv::Mat points_3D;
    cv::convertPointsFromHomogeneous(points_4D.t(), points_3D);

    cv::Mat rvec_left;
    cv::Rodrigues(P_left.get_minor<3, 3>(0, 0), rvec_left);
    cv::Mat tvec_left(P_left.get_minor<3, 1>(0, 3).t());

    std::vector<cv::Point2f> projected_left(aligned_left.points.size());
    cv::projectPoints(points_3D, rvec_left, tvec_left, intrinsics.K,
                      intrinsics.d, projected_left);

    cv::Mat rvec_right;
    cv::Rodrigues(P_right.get_minor<3, 3>(0, 0), rvec_right);
    cv::Mat tvec_right(P_right.get_minor<3, 1>(0, 3).t());

    std::vector<cv::Point2f> projected_right(aligned_right.points.size());
    cv::projectPoints(points_3D, rvec_right, tvec_right, intrinsics.K,
                      intrinsics.d, projected_right);

    float error_left, error_right;
    std::cout << "Total points: " << points_3D.rows << std::endl;
    for (size_t i = 0; i < points_3D.rows; i++) {
        error_left = cv::norm(projected_left[i] - aligned_left.points[i]);
        error_right = cv::norm(projected_right[i] - aligned_right.points[i]);
        if (error_left > 5 || error_right > 5) {
            std::cout << "Reprojection error for point: " << i << std::endl;
            std::cout << "  -> left: " << error_left << std::endl;
            std::cout << "  -> right: " << error_right << std::endl;
            continue;
        }

        PointCloudPoint pt;
        pt.point =
            cv::Point3f(points_3D.at<float>(i, 0), points_3D.at<float>(i, 1),
                        points_3D.at<float>(i, 2));

        pt.orgin_view[image_pair.left] = left_origin[i];
        pt.orgin_view[image_pair.right] = right_origin[i];
        point_cloud.push_back(pt);
    }

    return true;
}

bool StereoUtil::P_from_2D3D_matches(const Intrinsics &intrinsics,
                                     const Image2D3DPair &match,
                                     cv::Matx34f &P) {
    cv::Mat rvec, tvec;
    cv::Mat mask;

    if (match.points_2D.size() < 5 || match.points_3D.size() < 5) {
        std::cout << "ERROR: Not enough matches for PnPRansac..." << std::endl;
        return false;
    }

    cv::solvePnPRansac(match.points_3D, match.points_2D, intrinsics.K,
                       intrinsics.d, rvec, tvec, false, 100, 10.0f, 0.99, mask);

    if (((float)cv::countNonZero(mask) / (float)match.points_2D.size()) < 0.5) {
        std::cout << "Inliers ratio too small: "
                  << (float)cv::countNonZero(mask) /
                         (float)match.points_2D.size()
                  << std::endl;
        return false;
    }

    cv::Mat R;
    cv::Rodrigues(rvec, R);

    R.copyTo(cv::Mat(3, 4, CV_32FC1, P.val)(cv::Rect(0, 0, 3, 3)));
    tvec.copyTo(cv::Mat(3, 4, CV_32FC1, P.val)(cv::Rect(3, 0, 1, 3)));

    return true;
}
