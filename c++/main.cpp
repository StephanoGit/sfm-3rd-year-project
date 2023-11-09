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

#define IMG_DIR "../images/fountain-P11"

static float distancePointLine(const cv::Point2f point, const cv::Vec3f& line)
{
  //Line is given as a*x + b*y + c = 0
  return std::fabsf(line(0)*point.x + line(1)*point.y + line(2))
      / std::sqrt(line(0)*line(0)+line(1)*line(1));
}


static void drawEpipolarLines(const std::string& title, const cv::Mat F,
                const cv::Mat& img1, const cv::Mat& img2,
                const std::vector<cv::Point2f> points1,
                const std::vector<cv::Point2f> points2,
                const float inlierDistance = -1)
{
  CV_Assert(img1.size() == img2.size() && img1.type() == img2.type());
  cv::Mat outImg(img1.rows, img1.cols*2, CV_8UC3);
  cv::Rect rect1(0,0, img1.cols, img1.rows);
  cv::Rect rect2(img1.cols, 0, img1.cols, img1.rows);
  /*
   * Allow color drawing
   */
  if (img1.type() == CV_8U)
  {
    cv::cvtColor(img1, outImg(rect1), cv::COLOR_GRAY2BGR);
    cv::cvtColor(img2, outImg(rect2), cv::COLOR_GRAY2BGR);
  }
  else
  {
    img1.copyTo(outImg(rect1));
    img2.copyTo(outImg(rect2));
  }
  std::vector<cv::Vec3f> epilines1, epilines2;
  cv::computeCorrespondEpilines(points1, 1, F, epilines1); //Index starts with 1
  cv::computeCorrespondEpilines(points2, 2, F, epilines2);
 
  CV_Assert(points1.size() == points2.size() &&
        points2.size() == epilines1.size() &&
        epilines1.size() == epilines2.size());
 
  cv::RNG rng(0);
  for(size_t i=0; i<points1.size(); i++)
  {
    if(inlierDistance > 0)
    {
      if(distancePointLine(points1[i], epilines2[i]) > inlierDistance ||
        distancePointLine(points2[i], epilines1[i]) > inlierDistance)
      {
        //The point match is no inlier
        continue;
      }
    }
    /*
     * Epipolar lines of the 1st point set are drawn in the 2nd image and vice-versa
     */
    cv::Scalar color(rng(256),rng(256),rng(256));
 
    cv::line(outImg(rect2),
      cv::Point(0,-epilines1[i][2]/epilines1[i][1]),
      cv::Point(img1.cols,-(epilines1[i][2]+epilines1[i][0]*img1.cols)/epilines1[i][1]),
      color);
    cv::circle(outImg(rect1), points1[i], 3, color, -1);
 
    cv::line(outImg(rect1),
      cv::Point(0,-epilines2[i][2]/epilines2[i][1]),
      cv::Point(img2.cols,-(epilines2[i][2]+epilines2[i][0]*img2.cols)/epilines2[i][1]),
      color);
    cv::circle(outImg(rect2), points2[i], 3, color, -1);
  }
  cv::imshow(title, outImg);
  cv::waitKey(1);
}



















int main(int argc, char **argv){
    std::vector <ImageView> images_raw;

    cv::Mat K = compute_K();
    std::cout << K << std::endl;

    images_raw = load_images(IMG_DIR);

    for(int i = 0; i < images_raw.size();i++){
        images_raw[i].compute_kps_des(FeatureDetectionType::SIFT);
        std::cout << "Feature detection done for: " << images_raw[i].get_name() << "\n";
    }

    ImagePair pair_12(images_raw[6], images_raw[9]);
    pair_12.match_descriptors(FeatureMatchingType::BF);
    std::cout << "Descriptors matched for: " << pair_12.get_image1().get_name() << " and " << pair_12.get_image2().get_name() <<"\n";

    pair_12.draw_matches();
    std::cout << "Images matched: " << pair_12.get_image1().get_name() << " and " << pair_12.get_image2().get_name() <<"\n";

    cv::Mat F = compute_F(pair_12);
    std::cout << "F mat:" << F << std::endl;

    cv::Mat E = compute_E(K, F);
    std::cout << "E mat:" << E << std::endl;

    drawEpipolarLines("epilines", F, pair_12.get_image1().get_image(),
                            pair_12.get_image2().get_image(),
                            pair_12.get_img1_good_matches(),
                            pair_12.get_img2_good_matches());


    cv::imshow("Good matches", pair_12.get_matches_image());
    cv::waitKey();
    return 0;
}