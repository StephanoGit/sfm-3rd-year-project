#include "../include/SfmBundleAdjustment.h"
#include "../include/util.h"
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <cstddef>
#include <iostream>
#include <opencv2/core/matx.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

void initLogging() { google::InitGoogleLogging("sfm"); }

std::once_flag initLoggingFlag;

struct SimpleReprojectionError {
  SimpleReprojectionError(double observed_x, double observed_y)
      : observed_x(observed_x), observed_y(observed_y) {}
  template <typename T>
  bool operator()(const T *const camera, const T *const point,
                  const T *const focal, T *residuals) const {
    T p[3];
    // Rotate: camera[0,1,2] are the angle-axis rotation.
    ceres::AngleAxisRotatePoint(camera, point, p);

    // Translate: camera[3,4,5] are the translation.
    p[0] += camera[3];
    p[1] += camera[4];
    p[2] += camera[5];

    // Perspective divide
    const T xp = p[0] / p[2];
    const T yp = p[1] / p[2];

    // Compute final projected point position.
    const T predicted_x = *focal * xp;
    const T predicted_y = *focal * yp;

    // The error is the difference between the predicted and observed position.
    residuals[0] = predicted_x - T(observed_x);
    residuals[1] = predicted_y - T(observed_y);
    return true;
  }
  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction *Create(const double observed_x,
                                     const double observed_y) {
    return (
        new ceres::AutoDiffCostFunction<SimpleReprojectionError, 2, 6, 3, 1>(
            new SimpleReprojectionError(observed_x, observed_y)));
  }
  double observed_x;
  double observed_y;
};

void adjust_bundle(std::vector<PointCloudPoint> &pointcloud,
                   std::vector<cv::Matx34f> &P_mats, cv::Mat K, cv::Mat d,
                   const std::vector<Features> &features) {
  std::call_once(initLoggingFlag, initLogging);

  ceres::Problem problem;
  typedef cv::Matx<double, 1, 6> CameraVector;
  std::vector<CameraVector> camera_poses_6D;

  camera_poses_6D.reserve(P_mats.size());
  for (size_t i = 0; i < P_mats.size(); i++) {
    const cv::Matx34f &pose = P_mats[i];

    if (pose(0, 0) == 0 && pose(1, 1) == 0 && pose(2, 2) == 0) {
      camera_poses_6D.push_back(CameraVector());
      continue;
    }
    cv::Vec3f t(pose(0, 3), pose(1, 3), pose(2, 3));
    cv::Matx33f R = pose.get_minor<3, 3>(0, 0);

    float angle_axis[3];
    ceres::RotationMatrixToAngleAxis<float>(R.t().val, angle_axis);

    camera_poses_6D.push_back(CameraVector(angle_axis[0], angle_axis[1],
                                           angle_axis[2], t(0), t(1), t(2)));
  }
  double focal = K.at<float>(0, 0);
  std::vector<cv::Vec3d> points_3d(pointcloud.size());

  for (int i = 0; i < pointcloud.size(); i++) {
    const PointCloudPoint &p = pointcloud[i];
    points_3d[i] = cv::Vec3d(p.point.x, p.point.y, p.point.z);

    for (const auto &view : p.orgin_view) {
      cv::Point2f point_2d = features[view.first].points[view.second];

      point_2d.x -= K.at<float>(0, 2);
      point_2d.y -= K.at<float>(1, 2);

      ceres::CostFunction *cost_function =
          SimpleReprojectionError::Create(point_2d.x, point_2d.y);

      problem.AddResidualBlock(cost_function, NULL,
                               camera_poses_6D[view.first].val,
                               points_3d[i].val, &focal);
    }
  }

  // Make Ceres automatically detect the bundle structure. Note that the
  // standard solver, SPARSE_NORMAL_CHOLESKY, also works fine but it is slower
  // for standard bundle adjustment problems.
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.minimizer_progress_to_stdout = true;
  options.max_num_iterations = 500;
  options.eta = 1e-2;
  options.max_solver_time_in_seconds = 10;
  options.logging_type = ceres::LoggingType::SILENT;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.BriefReport() << "\n";

  if (not(summary.termination_type == ceres::CONVERGENCE)) {
    std::cerr << "Bundle adjustment failed." << std::endl;
    return;
  }

  K.at<float>(0, 0) = focal;
  K.at<float>(1, 1) = focal;

  // Implement the optimized camera poses and 3D points back into the
  // reconstruction
  for (size_t i = 0; i < P_mats.size(); i++) {
    cv::Matx34f &pose = P_mats[i];
    cv::Matx34f poseBefore = pose;

    if (pose(0, 0) == 0 and pose(1, 1) == 0 and pose(2, 2) == 0) {
      // This camera pose is empty, it was not used in the optimization
      continue;
    }

    // Convert optimized Angle-Axis back to rotation matrix
    double rotationMat[9] = {0};
    ceres::AngleAxisToRotationMatrix(camera_poses_6D[i].val, rotationMat);

    for (int r = 0; r < 3; r++) {
      for (int c = 0; c < 3; c++) {
        pose(c, r) = rotationMat[r * 3 + c]; //`rotationMat` is col-major...
      }
    }

    // Translation
    pose(0, 3) = camera_poses_6D[i](3);
    pose(1, 3) = camera_poses_6D[i](4);
    pose(2, 3) = camera_poses_6D[i](5);
  }

  for (int i = 0; i < pointcloud.size(); i++) {
    pointcloud[i].point.x = points_3d[i](0);
    pointcloud[i].point.y = points_3d[i](1);
    pointcloud[i].point.z = points_3d[i](2);
  }
}
