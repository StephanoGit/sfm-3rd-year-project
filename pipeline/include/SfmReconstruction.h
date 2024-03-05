#ifndef __SFM_RECONSTRUCTION
#define __SFM_RECONSTRUCTION

#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>

#include "FeatureUtil.h"
#include "SfmStructures.h"

class SfmReconstruction {
    // matrix of macthes from image i to image j
    typedef std::vector<std::vector<std::vector<cv::DMatch>>> MatchMatrix;
    typedef std::map<int, Image2D3DPair> Image2D3DMatches;

private:
    std::string directory;
    std::string reconstruction_name;
    FeatureUtil feature_util;

    Intrinsics intrinsics;
    MatchMatrix match_matrix;
    std::vector<Features> images_features;
    std::vector<cv::Mat> images;
    std::vector<std::string> images_paths;

    std::vector<PointCloudPoint> n_point_cloud;
    std::vector<cv::Matx34f> n_P_mats;
    std::set<int> n_done_views;
    std::set<int> n_good_views;

    bool verbose;

public:
    SfmReconstruction(std::string directory, std::string reconstruction_name,
                      FeatureExtractionType extract_type,
                      FeatureMatchingType match_type, Intrinsics intrinsics,
                      bool verbose);
    virtual ~SfmReconstruction();

    bool run_sfm_reconstruction(int resize_val);

    void create_match_matrix();

    void find_baseline_triangulation();

    void adjust_bundle();

    std::map<float, ImagePair> sort_views_for_baseline();

    void add_views_to_reconstruction();

    Image2D3DMatches find_2D3D_matches();

    void merge_point_cloud(const std::vector<PointCloudPoint> point_cloud);

    void export_pointcloud_to_PLY(const std::string &file_name);

    std::vector<PointCloudPoint> get_point_cloud();
};

#endif
