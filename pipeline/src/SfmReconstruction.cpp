#include "../include/SfmReconstruction.h"
#include "../include/CommonUtil.h"
#include "../include/IOUtil.h"
#include "../include/PlottingUtil.h"
#include "../include/SfmBundleAdjustment.h"
#include "../include/StereoUtil.h"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <opencv2/core/cvdef.h>
#include <opencv2/core/matx.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <ostream>
#include <string>
#include <thread>
#include <vector>

#include "../include/PMVS2Reconstruction.h"
SfmReconstruction::SfmReconstruction(std::string directory,
                                     std::string reconstruction_name,
                                     FeatureExtractionType extract_type,
                                     FeatureMatchingType match_type,
                                     Intrinsics intrinsics,
                                     std::string input_type, bool verbose) {
    this->directory = directory;
    this->reconstruction_name = reconstruction_name;
    this->input_type = input_type;
    this->feature_util = FeatureUtil(extract_type, match_type);
    this->intrinsics = intrinsics;
    this->verbose = verbose;
};
SfmReconstruction::~SfmReconstruction(){};

bool SfmReconstruction::run_sfm_reconstruction(int resize_val) {

    std::cout << "=======================================" << std::endl;
    std::cout << "     Starting SfM Reconstruction...    " << std::endl;
    std::cout << "=======================================" << std::endl;

    // create file paths
    std::filesystem::create_directories("../reconstructions/" +
                                        this->reconstruction_name);
    std::filesystem::create_directories("../reconstructions/" +
                                        this->reconstruction_name + "/sparse");
    std::filesystem::create_directories("../reconstructions/" +
                                        this->reconstruction_name + "/dense");
    std::filesystem::create_directories("../reconstructions/" +
                                        this->reconstruction_name + "/mesh");

    // load images or video
    if (this->input_type == "images") {
        this->images = load_images(directory, resize_val, this->images_paths);
    } else {
        this->images =
            video_to_images(directory, 36, resize_val, this->images_paths);
        std::cout << "No. of extracted images: " << this->images.size()
                  << std::endl;
    }
    if (this->images.size() < 2) {
        std::cout << "ERROR: Please provide at least 2 images..." << std::endl;
        return false;
    }

    // allocating space for camera projection matrices
    this->n_P_mats.resize(this->images.size());

    // extract features
    this->images_features = std::vector<Features>(this->images.size());
    for (size_t i = 0; i < this->images.size(); i++) {
        this->images_features[i] =
            this->feature_util.extract_features(this->images[i]);

        if (verbose) {
            cv::Mat fm =
                draw_features(this->images[i], this->images_features[i]);
            cv::imshow(
                "IMG " + std::to_string(i) + ": " +
                    std::to_string(this->images_features[i].key_points.size()),
                fm);
            cv::waitKey(0);
            cv::destroyAllWindows();
        }
    }

    // match features
    create_match_matrix();

    find_baseline_triangulation();

    add_views_to_reconstruction();

    export_pointcloud_to_PLY("../reconstructions/" + this->reconstruction_name +
                             "/sparse/final");

    // sparse -> dense
    PMVS2Reconstruction pmvs2;
    pmvs2.dense_reconstruction(this->images, this->images_paths, this->n_P_mats,
                               this->intrinsics);

    int dont_care = std::system("../libs/pmvs2 denseCloud/ options.txt");
    if (dont_care > 0) {
        std::cout << "ERROR: pmvs2 failed" << std::endl;
    }

    return true;
}

void SfmReconstruction::create_match_matrix() {
    const size_t number_of_images = this->images.size();
    this->match_matrix.resize(
        number_of_images,
        std::vector<std::vector<cv::DMatch>>(number_of_images));

    std::vector<ImagePair> pairs;
    for (size_t i = 0; i < number_of_images; i++) {
        for (size_t j = i + 1; j < number_of_images; j++) {
            pairs.push_back({i, j});
        }
    }

    std::vector<std::thread> threads;

    const int number_of_threads = std::thread::hardware_concurrency() - 1;
    const int pairs_per_thread =
        (number_of_threads > pairs.size())
            ? 1
            : (int)ceilf((float)(pairs.size()) / number_of_threads);

    std::cout << "Threads: " << number_of_threads << " Pairs: " << pairs.size()
              << std::endl;
    std::cout << "Pairs per thread: " << pairs_per_thread << std::endl;

    std::mutex write_mutex;

    for (size_t thread_id = 0; thread_id < MIN(number_of_threads, pairs.size());
         thread_id++) {
        threads.push_back(std::thread([&, thread_id] {
            const int start_pair = pairs_per_thread * thread_id;

            for (int j = 0; j < pairs_per_thread; j++) {
                const int pair_id = start_pair + j;
                if (pair_id >= pairs.size()) {
                    std::cout << "WARNING: Thread overflow" << std::endl;
                    break;
                }
                const ImagePair &pair = pairs[pair_id];
                this->match_matrix[pair.left][pair.right] =
                    this->feature_util.match_features(
                        this->images_features[pair.left],
                        this->images_features[pair.right]);

                write_mutex.lock();
                std::cout << "Thread " << thread_id << ": Match (pair "
                          << pair_id << ") " << pair.left << ", " << pair.right
                          << ": "
                          << this->match_matrix[pair.left][pair.right].size()
                          << " matched features" << std::endl;

                write_mutex.unlock();
            }
        }));
    }

    for (auto &th : threads) {
        th.join();
    }
}

std::map<float, ImagePair> SfmReconstruction::sort_views_for_baseline() {
    std::cout << "===========================================" << std::endl;
    std::cout << "Sorting views for baseline triangulation..." << std::endl;
    std::cout << "===========================================" << std::endl;

    std::map<float, ImagePair> pairs_inliers;
    const size_t number_of_images = this->images.size();

    for (size_t i = 0; i < number_of_images - 1; i++) {
        for (size_t j = i + 1; j < number_of_images; j++) {
            if (this->match_matrix[i][j].size() < 100) {
                std::cout << "NOT ENOUGH MATCHES Pair (" << i << ", " << j
                          << ") -- [❌]" << std::endl
                          << std::endl;
                pairs_inliers[1.0] = {i, j};
                continue;
            }

            const int inliers = StereoUtil::homography_inliers(
                this->images_features[i], this->images_features[j],
                this->match_matrix[i][j]);

            const float inliers_ratio =
                (float)inliers / (float)(this->match_matrix[i][j].size());
            pairs_inliers[inliers_ratio] = {i, j};

            std::cout << "Homography inliers ratio pair(" << i << ", " << j
                      << "): " << inliers_ratio << std::endl
                      << std::endl;
        }
    }
    return pairs_inliers;
}

void SfmReconstruction::find_baseline_triangulation() {
    std::map<float, ImagePair> pairs_homography_inliers =
        sort_views_for_baseline();

    cv::Matx34f P_left = cv::Matx34f::eye();
    cv::Matx34f P_right = cv::Matx34f::eye();
    std::vector<PointCloudPoint> pointcloud;

    for (auto &pair : pairs_homography_inliers) {
        std::cout << "Pair (" << pair.second.left << ", " << pair.second.right
                  << ") ratio: " << pair.first << std::endl;

        size_t i = pair.second.left;
        size_t j = pair.second.right;
        std::vector<cv::DMatch> mask_matches;

        // try to remove outliers using homography (might not be good for
        // objects with different planes)
        /* bool dont_care = StereoUtil::remove_homography_outliers( */
        /*     this->images_features[i], this->images_features[j], */
        /*     this->match_matrix[i][j], mask_matches); */
        /* if (!dont_care) { */
        /*     std::cout << "Pair (" << pair.second.left << ", " */
        /*               << pair.second.right << ") UNSUCCESSFUL - (HOMOGRAPHY)"
         */
        /*               << std::endl; */
        /*     continue; */
        /* } */
        /**/
        /* this->match_matrix[i][j] = mask_matches; */
        /**/
        /* cv::Mat image_homo = draw_matches( */
        /*     this->images[i], this->images[j], this->images_features[i], */
        /*     this->images_features[j], this->match_matrix[i][j]); */
        /**/
        /* cv::imshow("Homography Inliers Matches", image_homo); */
        /* cv::waitKey(0); */
        /**/
        /* mask_matches.clear(); */
        ////////////////////////////////////////////////////////////////////

        bool success = StereoUtil::camera_matrices_from_matches(
            this->intrinsics, this->match_matrix[i][j],
            this->images_features[i], this->images_features[j], mask_matches,
            P_left, P_right);
        if (!success) {
            std::cout << "Pair (" << pair.second.left << ", "
                      << pair.second.right << ") UNSUCCESSFUL" << std::endl;
            continue;
        }

        float match_inlier_ratio =
            (float)mask_matches.size() / (float)this->match_matrix[i][j].size();

        if (match_inlier_ratio < 0.5) {
            std::cout << "Pair (" << pair.second.left << ", "
                      << pair.second.right
                      << ") UNSUCCESSFUL -- insufficient match inliers -- "
                      << match_inlier_ratio << std::endl;
            continue;
        }

        if (this->verbose) {
            cv::Mat m = draw_matches(this->images[i], this->images[j],
                                     this->images_features[i],
                                     this->images_features[j], mask_matches);
            cv::imshow("matches", m);
            cv::waitKey(0);
        }
        this->match_matrix[i][j] = mask_matches;

        success = StereoUtil::triangulate_views(
            this->intrinsics, pair.second, this->match_matrix[i][j],
            this->images_features[i], this->images_features[j], P_left, P_right,
            pointcloud);

        if (!success) {
            std::cout << "Pair (" << pair.second.left << ", "
                      << pair.second.right << ") UNSUCCESSFUL TRIANGULATION"
                      << std::endl;
            continue;
        }

        this->n_point_cloud = pointcloud;
        this->n_P_mats[i] = P_left;
        this->n_P_mats[j] = P_right;
        this->n_done_views.insert(i);
        this->n_done_views.insert(j);
        this->n_good_views.insert(i);
        this->n_good_views.insert(j);

        bool status = SfmBundleAdjustment::adjust_bundle(
            this->n_point_cloud, this->n_P_mats, this->intrinsics,
            this->images_features);

        if (status) {
            export_pointcloud_to_PLY(
                "../reconstructions/" + this->reconstruction_name +
                "/sparse/baseline_triangulation_" + std::to_string(i) + "-" +
                std::to_string(j) + "_ba");
        } else {
            export_pointcloud_to_PLY(
                "../reconstructions/" + this->reconstruction_name +
                "/sparse/baseline_triangulation_" + std::to_string(i) + "-" +
                std::to_string(j) + "_no-ba");
        }

        break;
    }
}

void SfmReconstruction::add_views_to_reconstruction() {
    std::cout << "=======================================" << std::endl;
    std::cout << "          Adding more views...         " << std::endl;
    std::cout << "=======================================" << std::endl;

    StereoUtil stereo_util;
    while (this->n_done_views.size() != this->images.size()) {
        Image2D3DMatches matches_2D3D = find_2D3D_matches();

        size_t best_view;
        size_t best_number_matches = 0;
        for (const auto &match : matches_2D3D) {
            const size_t number_matches = match.second.points_2D.size();
            if (number_matches > best_number_matches) {
                best_view = match.first;
                best_number_matches = number_matches;
            }
        }

        /* if (best_number_matches == 0) { */
        /*     std::cout */
        /*         << "ERROR: Not enough matches, please try with a different
         * set " */
        /*            "of images or another feature extraction method" */
        /*         << std::endl; */
        /*     return; */
        /* } */

        std::cout << "Best view to add next: " << best_view << " with "
                  << best_number_matches << " matches" << std::endl;

        this->n_done_views.insert(best_view);

        cv::Matx34f P_new;
        bool success = stereo_util.P_from_2D3D_matches(
            this->intrinsics, matches_2D3D[best_view], P_new);

        if (!success) {
            std::cout << "Error: Cannot recover camera pose for view: "
                      << best_view << std::endl;
            continue;
        }

        this->n_P_mats[best_view] = P_new;

        bool new_view_success_triangulation = false;
        for (const int good_view : n_good_views) {
            size_t left_view_idx =
                (good_view < best_view) ? good_view : best_view;
            size_t right_view_idx =
                (good_view < best_view) ? best_view : good_view;

            cv::Matx34f P_left = cv::Matx34f::eye();
            cv::Matx34f P_right = cv::Matx34f::eye();
            std::vector<cv::DMatch> mask_matches;
            // try to remove outliers using homography (might not be good for
            // objects with different planes)
            /* bool dont_care = StereoUtil::remove_homography_outliers( */
            /*     this->images_features[left_view_idx], */
            /*     this->images_features[right_view_idx], */
            /*     this->match_matrix[left_view_idx][right_view_idx], */
            /*     mask_matches); */
            /* if (!dont_care) { */
            /*     std::cout << "Pair (" << left_view_idx << ", " <<
             * right_view_idx */
            /*               << ") UNSUCCESSFUL - (HOMOGRAPHY)" << std::endl; */
            /*     continue; */
            /* } */
            /* this->match_matrix[left_view_idx][right_view_idx] = mask_matches;
             */
            /**/
            /* mask_matches.clear(); */

            bool success = StereoUtil::camera_matrices_from_matches(
                this->intrinsics,
                this->match_matrix[left_view_idx][right_view_idx],
                this->images_features[left_view_idx],
                this->images_features[right_view_idx], mask_matches, P_left,
                P_right);
            if (!success) {
                continue;
            }
            this->match_matrix[left_view_idx][right_view_idx] = mask_matches;

            // show matches now
            if (this->verbose) {
                cv::Mat m = draw_matches(
                    this->images[left_view_idx], this->images[right_view_idx],
                    this->images_features[left_view_idx],
                    this->images_features[right_view_idx], mask_matches);
                cv::imshow("Matches: " + std::to_string(left_view_idx) + " - " +
                               std::to_string(right_view_idx),
                           m);
                cv::waitKey(0);
            }
            std::vector<PointCloudPoint> pointcloud;
            success = StereoUtil::triangulate_views(
                this->intrinsics, {left_view_idx, right_view_idx},
                this->match_matrix[left_view_idx][right_view_idx],
                this->images_features[left_view_idx],
                this->images_features[right_view_idx],
                this->n_P_mats[left_view_idx], this->n_P_mats[right_view_idx],
                pointcloud);

            if (success) {
                std::cout
                    << "Merging Triangulation between " << left_view_idx
                    << " and " << right_view_idx << ", no. matching points = "
                    << this->match_matrix[left_view_idx][right_view_idx].size()
                    << std::endl;
                merge_point_cloud(pointcloud);
                new_view_success_triangulation = true;
            } else {
                std::cout << "Triangulation between " << left_view_idx
                          << " and " << right_view_idx << " FAILED"
                          << std::endl;
            }
        }

        if (new_view_success_triangulation) {
            bool status = SfmBundleAdjustment::adjust_bundle(
                this->n_point_cloud, this->n_P_mats, this->intrinsics,
                this->images_features);
            if (status) {
                export_pointcloud_to_PLY(
                    "../reconstructions/" + this->reconstruction_name +
                    "/sparse/" + std::to_string(this->n_done_views.size()) +
                    "__" + std::to_string(best_view) + "_ba");
            } else {
                export_pointcloud_to_PLY(
                    "../reconstructions/" + this->reconstruction_name +
                    "/sparse/" + std::to_string(this->n_done_views.size()) +
                    "__" + std::to_string(best_view) + "_no-ba");
            }
        }
        this->n_good_views.insert(best_view);
    }
}

SfmReconstruction::Image2D3DMatches SfmReconstruction::find_2D3D_matches() {
    Image2D3DMatches matches;
    // scan through the remaining images
    for (size_t view_idx = 0; view_idx < this->images.size(); view_idx++) {
        // skip the done views
        if (this->n_done_views.find(view_idx) != this->n_done_views.end()) {
            continue;
        }
        Image2D3DPair pair_2D3D;

        // scan through each point within the current point cloud
        for (const PointCloudPoint &cloud_point : this->n_point_cloud) {

            bool found_2D_point = false;
            // for each origin view of the poin cloud
            for (const auto &origin_view : cloud_point.orgin_view) {
                // 2D - 2D matching between views
                int origin_view_idx = origin_view.first;
                int origin_view_feature_idx = origin_view.second;

                // left index < right index (the match matrix is
                // upper triangular, i < j)
                //   | a00 a01 a02 a03|
                //   |  0  a11 a12 a13|
                //   |  0   0  a22 a23|
                //   |  0   0   0  a33|
                int left_view_idx =
                    (origin_view_idx < view_idx) ? origin_view_idx : view_idx;
                int right_view_idx =
                    (origin_view_idx < view_idx) ? view_idx : origin_view_idx;

                // scan all 2D - 2D matches between the new view and
                // the origin view
                for (const cv::DMatch &m :
                     this->match_matrix[left_view_idx][right_view_idx]) {

                    int matched_2D_point_in_new_view = -1;

                    if (origin_view_idx < view_idx) {
                        // origin is on the left
                        if (m.queryIdx == origin_view_feature_idx) {
                            matched_2D_point_in_new_view = m.trainIdx;
                        }
                    } else {
                        // origin is on the right
                        if (m.trainIdx == origin_view_feature_idx) {
                            matched_2D_point_in_new_view = m.queryIdx;
                        }
                    }

                    if (matched_2D_point_in_new_view >= 0) {
                        // point found
                        Features &new_view_features =
                            this->images_features[view_idx];
                        pair_2D3D.points_2D.push_back(
                            new_view_features
                                .points[matched_2D_point_in_new_view]);
                        pair_2D3D.points_3D.push_back(cloud_point.point);
                        found_2D_point = true;
                        break;
                    }
                }
                if (found_2D_point) {
                    break;
                }
            }
        }
        matches[view_idx] = pair_2D3D;
    }
    return matches;
}

void SfmReconstruction::export_pointcloud_to_PLY(const std::string &file_name) {
    std::cout << "Saving pointcloud to file: " << file_name << std::endl;

    std::ofstream stream(file_name + ".ply");
    stream << "ply                 " << std::endl
           << "format ascii 1.0    " << std::endl
           << "element vertex " << this->n_point_cloud.size() << std::endl
           << "property float x    " << std::endl
           << "property float y    " << std::endl
           << "property float z    " << std::endl
           << "property uchar red  " << std::endl
           << "property uchar green" << std::endl
           << "property uchar blue " << std::endl
           << "end_header          " << std::endl;

    for (const PointCloudPoint &point : this->n_point_cloud) {
        auto origin_view = point.orgin_view.begin();
        const int view_idx = origin_view->first;
        cv::Point2f point_2D =
            this->images_features[view_idx].points[origin_view->second];
        cv::Vec3b colour = this->images[view_idx].at<cv::Vec3b>(point_2D);

        stream << point.point.x << " " << point.point.y << " " << point.point.z
               << " " << (int)colour(2) << " " << (int)colour(1) << " "
               << (int)colour(0) << " " << std::endl;
    }

    stream.close();
}

void SfmReconstruction::merge_point_cloud(
    const std::vector<PointCloudPoint> new_pointcloud) {
    std::cout << "=======================================" << std::endl;
    std::cout << "        Merging Point Clouds...        " << std::endl;
    std::cout << "=======================================" << std::endl;

    std::vector<std::vector<std::vector<cv::DMatch>>> merged_match_matrix;
    merged_match_matrix.resize(
        this->images.size(),
        std::vector<std::vector<cv::DMatch>>(this->images.size()));

    size_t new_points = 0;
    size_t merged_points = 0;

    for (const PointCloudPoint &pt : new_pointcloud) {
        const cv::Point3f new_point = pt.point;

        bool found_matching_existing_views = false;
        bool found_3D_point_match = false;

        for (PointCloudPoint &existing_point : this->n_point_cloud) {
            if (cv::norm(existing_point.point - new_point) < 0.01) {
                found_3D_point_match = true;

                for (const auto &new_view : pt.orgin_view) {
                    for (const auto &existing_view :
                         existing_point.orgin_view) {
                        bool found_matching_feature = false;
                        const bool new_is_left =
                            new_view.first < existing_view.first;

                        const int left_view_idx = (new_is_left)
                                                      ? new_view.first
                                                      : existing_view.first;
                        const int left_view_feature_idx =
                            (new_is_left) ? new_view.second
                                          : existing_view.second;
                        const int right_view_idx = (new_is_left)
                                                       ? existing_view.first
                                                       : new_view.first;
                        const int right_view_feature_idx =
                            (new_is_left) ? existing_view.second
                                          : new_view.second;

                        const std::vector<cv::DMatch> &matching =
                            this->match_matrix[left_view_idx][right_view_idx];

                        for (const cv::DMatch &match : matching) {
                            if (match.queryIdx == left_view_feature_idx &&
                                match.trainIdx == right_view_feature_idx &&
                                match.distance < 20.0) {
                                merged_match_matrix[left_view_idx]
                                                   [right_view_idx]
                                                       .push_back(match);

                                found_matching_feature = true;
                                break;
                            }
                        }

                        if (found_matching_feature) {
                            existing_point.orgin_view[new_view.first] =
                                new_view.second;
                            found_matching_existing_views = true;
                        }
                    }
                }
            }
            if (found_matching_existing_views) {
                merged_points++;
                break;
            }
        }

        if (!found_matching_existing_views && !found_3D_point_match) {
            this->n_point_cloud.push_back(pt);
            new_points++;
        }
    }
}

std::vector<PointCloudPoint> SfmReconstruction::get_point_cloud() {
    return this->n_point_cloud;
}
