#include "SfmReconstruction.h"
#include "util.h"
#include "ImagePair.h"

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

double new_fx = (3278.68 * NEW_VIDEO_WIDTH) / VIDEO_WIDTH;
double new_fy = (3278.68 * NEW_VIDEO_HEIGHT) / VIDEO_HEIGHT;
double new_cx = ((VIDEO_WIDTH / 2) * NEW_VIDEO_WIDTH) / VIDEO_WIDTH;
double new_cy = ((VIDEO_HEIGHT / 2) * NEW_VIDEO_HEIGHT) / VIDEO_HEIGHT;

// double new_fx = 2759.48;
// double new_fy = 2764.16;
// double new_cx = 1520.69;
// double new_cy = 1006.81;

double data[9] = {new_fx, 0, new_cx,
                  0, new_fy, new_cy,
                  0, 0, 1};

SfmReconstruction::SfmReconstruction(std::vector<ImagePair> frames)
{
    this->K = cv::Mat(3, 3, CV_64F, data);
    this->distortion = {0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0};
    this->R0 = cv::Mat::eye(3, 3, CV_64F);
    this->t0 = cv::Mat::zeros(3, 1, CV_64F);
    this->frames = frames;
}

cv::Mat SfmReconstruction::get_K()
{
    return this->K;
}

std::vector<double> SfmReconstruction::get_distortion()
{
    return this->distortion;
}

void SfmReconstruction::triangulation()
{
    cv::Mat R1 = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat t1 = cv::Mat::zeros(3, 1, CV_64F);

    cv::Mat P1(3, 4, CV_64F);
    cv::hconcat(R1, t1, P1);
    P1 = this->K * P1;
    append_P_mat(P1);

    for (size_t i = 0; i < this->frames.size(); i++)
    {
        cv::Mat points_3d;

        std::cout << "====== Matching descriptors ======" << std::endl;
        frames[i].match_descriptors(FeatureMatchingType::FLANN);
        std::cout << "==================================" << std::endl
                  << std::endl;

        std::cout << "========= Computing F Mat ========" << std::endl;
        frames[i].compute_F();
        // std::cout << frames[i].get_F() << std::endl;
        std::cout << "==================================" << std::endl
                  << std::endl;

        std::cout << "========= Computing E Mat ========" << std::endl;
        frames[i].compute_E(this->K);
        // std::cout << frames[i].get_E() << std::endl;
        std::cout << "==================================" << std::endl
                  << std::endl;

        std::cout << "======= Computing R and t ========" << std::endl;
        frames[i].compute_Rt(this->K);
        cv::Mat R2 = frames[i].get_R();
        cv::Mat t2 = frames[i].get_t();
        // std::cout << R2 << std::endl;
        // std::cout << t2 << std::endl;
        std::cout << "==================================" << std::endl
                  << std::endl;

        // cv::Mat P1(3, 4, CV_64F);
        // cv::hconcat(R1, t1, P1);
        // P1 = this->K * P1;

        cv::Mat P2(3, 4, CV_64F);
        cv::hconcat(R2, t2, P2);
        P2 = this->K * P2;
        append_P_mat(P2);

        cv::Mat points_4d;
        cv::triangulatePoints(P1, P2, frames[i].get_image1_good_matches(), frames[i].get_image2_good_matches(), points_4d);
        cv::convertPointsFromHomogeneous(points_4d.t(), points_3d);
        export_3d_points_to_txt("../points-3d/" + std::to_string(i) + ".json", points_3d);

        R1 = R2.clone();
        t1 = t2.clone();
        P1 = P2.clone();
    }
}

void SfmReconstruction::set_P_mats(std::vector<cv::Mat> P_mats)
{
    this->P_mats = P_mats;
}

void SfmReconstruction::append_P_mat(cv::Mat P)
{
    this->P_mats.push_back(P);
}

std::vector<cv::Mat> SfmReconstruction::get_P_mats()
{
    return this->P_mats;
}

SfmReconstruction::~SfmReconstruction(){};