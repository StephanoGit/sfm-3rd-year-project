#include "SfmReconstruction.h"
#include "util.h"

#define WIDTH 3072
#define HEIGHT 2048
#define NEW_WIDTH 640
#define NEW_HEIGHT 480

double new_fx = (2759.48 * NEW_WIDTH) / WIDTH;
double new_fy = (2764.16 * NEW_HEIGHT) / HEIGHT;
double new_cx = (1520.69 * NEW_WIDTH) / WIDTH;
double new_cy = (1006.81 * NEW_HEIGHT) / HEIGHT;

double data[9] = {new_fx, 0, new_cx,
                  0, new_fy, new_cy,
                  0, 0, 1};

SfmReconstruction::SfmReconstruction(std::vector<ImagePair> frames)
{
    this->K = cv::Mat(3, 3, CV_64F, data);
    this->R0 = cv::Mat::eye(3, 3, CV_64F);
    this->t0 = cv::Mat::zeros(3, 1, CV_64F);
    this->frames = frames;
}

cv::Mat SfmReconstruction::get_K()
{
    return this->K;
}

SfmReconstruction::~SfmReconstruction(){};