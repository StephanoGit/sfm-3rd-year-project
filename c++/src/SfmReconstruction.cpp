#include "../include/SfmReconstruction.h"

#include <stdio.h>

#include "../include/ImagePair.h"
#include "../include/ImageView.h"
#include "../include/drawUtil.h"
#include "../include/util.h"

#define PHOTO_WIDTH 3072
#define PHOTO_HEIGHT 2048
#define NEW_PHOTO_WIDTH 640
#define NEW_PHOTO_HEIGHT 480

#define VIDEO_WIDTH 3840
#define VIDEO_HEIGHT 2160
#define NEW_VIDEO_WIDTH 1280
#define NEW_VIDEO_HEIGHT 720

// double new_fx = (2759.48 * NEW_PHOTO_WIDTH) / PHOTO_WIDTH;
// double new_fy = (2764.16 * NEW_PHOTO_HEIGHT) / PHOTO_HEIGHT;
// double new_cx = (1520.69 * NEW_PHOTO_WIDTH) / PHOTO_WIDTH;
// double new_cy = (1006.81 * NEW_PHOTO_HEIGHT) / PHOTO_HEIGHT;

// double new_fx = (3278.68 * NEW_VIDEO_WIDTH) / VIDEO_WIDTH;
// double new_fy = (3278.68 * NEW_VIDEO_HEIGHT) / VIDEO_HEIGHT;
// double new_cx = ((VIDEO_WIDTH / 2) * NEW_VIDEO_WIDTH) / VIDEO_WIDTH;
// double new_cy = ((VIDEO_HEIGHT / 2) * NEW_VIDEO_HEIGHT) / VIDEO_HEIGHT;

double new_fx = 2759.48;
double new_fy = 2764.16;
double new_cx = 1520.69;
double new_cy = 1006.81;

double data[9] = {new_fx, 0, new_cx, 0, new_fy, new_cy, 0, 0, 1};

SfmReconstruction::SfmReconstruction(std::vector<ImageView> views,
                                     FeatureDetectionType detection_type,
                                     FeatureMatchingType matching_type) {
    if (views.size() < 2) {
        std::cout << "Please provide at least two images..." << std::endl;
        return;
    }

    this->K = cv::Mat(3, 3, CV_64F, data);
    this->distortion = {0.0, 0.0, 0.0, 0.0, 0.0};
    this->views = views;
    this->detection_type = detection_type;
    this->matching_type = matching_type;

    // extract kp and desc of each image
    for (int i = 0; i < views.size(); i++) {
        views[i].compute_kps_des(detection_type);
    }

    // create the initial pair (the first two views)
    ImagePair init_pair(views[4], views[5]);

    // start the initial reconstruction and return the point cloud
    this->point_cloud = init_pair.init_reconstruction(
        detection_type, matching_type, this->K, this->distortion);

    export_3d_points_to_txt("../points-3d/norm.json", this->point_cloud);

    // get info about the last view
    this->last_image = init_pair.get_image2();
    this->last_image_kps = init_pair.get_image2_good_kps();
    this->last_image_desc = init_pair.get_image2_good_desc();

    // add the next views (this will be a loop in the future)
    // for now we are testing with a third view
    // this->add_new_view(views[2], init_pair.get_image1());
}

void SfmReconstruction::add_new_view(ImageView new_image,
                                     ImageView last_image_raw) {
    // create a new ImageView object representing the last image
    // it should have the filtered descriptors and kps present in the current
    // point cloud
    ImageView last_image(last_image_raw.get_image(), last_image_raw.get_name(),
                         this->detection_type, this->last_image_kps,
                         this->last_image_desc);

    // create a pair between the last image and the new image and match them
    ImagePair pair(last_image, new_image);
    pair.match_descriptors(this->matching_type);

    // filter the outliers
    pair.compute_F();
    pair.compute_E(this->K);

    // find a way to keep only the 3d points represnted by the good_matches
    // between the two
}

std::vector<cv::Point3f> SfmReconstruction::get_point_cloud() {
    return this->point_cloud;
}

void SfmReconstruction::set_point_cloud(std::vector<cv::Point3f> point_cloud) {
    this->point_cloud = point_cloud;
}

cv::Mat SfmReconstruction::get_K() { return this->K; }

std::vector<double> SfmReconstruction::get_distortion() {
    return this->distortion;
}

SfmReconstruction::~SfmReconstruction(){};