#include "util.h"

#include <stdio.h>
#include <string>
#include <vector>
#include <fstream>

#include <opencv2/opencv.hpp>

#define WIDTH 3072
#define HEIGHT 2048
#define NEW_WIDTH 640
#define NEW_HEIGHT 480

bool sortByName(const std::__fs::filesystem::directory_entry &entry1, const std::__fs::filesystem::directory_entry &entry2)
{
  return entry1.path().filename() < entry2.path().filename();
}

std::vector<ImageView> load_images(std::string directory)
{
  std::string current_file = "";
  std::vector<ImageView> images;
  ImageView current_image;

  std::vector<std::__fs::filesystem::directory_entry> entries;
  for (const auto &entry : std::__fs::filesystem::directory_iterator(directory))
  {
    entries.push_back(entry);
  }

  // Sort entries by name
  std::sort(entries.begin(), entries.end(), sortByName);

  // get all images within the directory
  for (const auto &entry : entries)
  {
    current_file = entry.path();
    current_image.set_name(current_file.substr(10));

    // might have to change the image colour channels
    current_image.set_image(cv::imread(current_file, cv::IMREAD_UNCHANGED));

    // check image for corrent format or existence
    if (current_image.get_image().data == NULL)
    {
      printf("Image not found or incorrect file type!\n");
      continue;
    }

    // resize
    // cv::Mat original_image = current_image.get_image();
    // cv::Mat resized_image;
    // cv::resize(original_image, resized_image, cv::Size(NEW_WIDTH, NEW_HEIGHT));
    // current_image.set_image(resized_image);

    // add image to vector
    images.push_back(current_image);
  }

  return images;
}

float distancePointLine(const cv::Point2f point, const cv::Vec3f &line)
{
  // Line is given as a*x + b*y + c = 0
  return std::fabsf(line(0) * point.x + line(1) * point.y + line(2)) / std::sqrt(line(0) * line(0) + line(1) * line(1));
}

void drawEpipolarLines(const std::string &title, const cv::Mat F,
                       const cv::Mat &image1, const cv::Mat &image2,
                       const std::vector<cv::Point2f> points1,
                       const std::vector<cv::Point2f> points2,
                       const float inlierDistance = -1)
{
  CV_Assert(image1.size() == image2.size() && image1.type() == image2.type());
  cv::Mat outImg(image1.rows, image1.cols * 2, CV_8UC3);
  cv::Rect rect1(0, 0, image1.cols, image1.rows);
  cv::Rect rect2(image1.cols, 0, image1.cols, image1.rows);
  /*
   * Allow color drawing
   */
  if (image1.type() == CV_8U)
  {
    cv::cvtColor(image1, outImg(rect1), cv::COLOR_GRAY2BGR);
    cv::cvtColor(image2, outImg(rect2), cv::COLOR_GRAY2BGR);
  }
  else
  {
    image1.copyTo(outImg(rect1));
    image2.copyTo(outImg(rect2));
  }
  std::vector<cv::Vec3f> epilines1, epilines2;
  cv::computeCorrespondEpilines(points1, 1, F, epilines1); // Index starts with 1
  cv::computeCorrespondEpilines(points2, 2, F, epilines2);

  CV_Assert(points1.size() == points2.size() &&
            points2.size() == epilines1.size() &&
            epilines1.size() == epilines2.size());

  cv::RNG rng(0);
  for (size_t i = 0; i < points1.size(); i++)
  {
    if (inlierDistance > 0)
    {
      if (distancePointLine(points1[i], epilines2[i]) > inlierDistance ||
          distancePointLine(points2[i], epilines1[i]) > inlierDistance)
      {
        // The point match is no inlier
        continue;
      }
    }
    /*
     * Epipolar lines of the 1st point set are drawn in the 2nd image and vice-versa
     */
    cv::Scalar color(rng(256), rng(256), rng(256));

    cv::line(outImg(rect2),
             cv::Point(0, -epilines1[i][2] / epilines1[i][1]),
             cv::Point(image1.cols, -(epilines1[i][2] + epilines1[i][0] * image1.cols) / epilines1[i][1]),
             color);
    cv::circle(outImg(rect1), points1[i], 3, color, -1);

    cv::line(outImg(rect1),
             cv::Point(0, -epilines2[i][2] / epilines2[i][1]),
             cv::Point(image2.cols, -(epilines2[i][2] + epilines2[i][0] * image2.cols) / epilines2[i][1]),
             color);
    cv::circle(outImg(rect2), points2[i], 3, color, -1);
  }

  std::cout << "dwdawd" << std::endl;
  cv::imshow(title, outImg);
  cv::waitKey();
}

void depl(cv::Mat image_left, cv::Mat image_right, cv::Mat fundemental, std::vector<cv::Point2f> selPoints1, std::vector<cv::Point2f> selPoints2)
{
  // Draw the left points corresponding epipolar lines in the right image
  std::vector<cv::Vec3f> lines1;
  cv::computeCorrespondEpilines(cv::Mat(selPoints1), 1, fundemental, lines1);
  for (size_t i = 0; i < selPoints1.size(); ++i)
  {
    cv::Vec3f line = lines1[i];
    cv::Point point(selPoints2[i].x, selPoints2[i].y);

    // Draw the epipolar line
    cv::line(image_right, cv::Point(0, -line[2] / line[1]),
             cv::Point(image_right.cols, -(line[2] + line[0] * image_right.cols) / line[1]),
             cv::Scalar(255, 255, 255));

    // Draw the corresponding point
    cv::circle(image_right, point, 5, cv::Scalar(0, 0, 255), -1);
  }

  // Draw the right points corresponding epipolar lines in the left image
  std::vector<cv::Vec3f> lines2;
  cv::computeCorrespondEpilines(cv::Mat(selPoints2), 2, fundemental, lines2);
  for (size_t i = 0; i < selPoints2.size(); ++i)
  {
    cv::Vec3f line = lines2[i];
    cv::Point point(selPoints1[i].x, selPoints1[i].y);

    // Draw the epipolar line
    cv::line(image_left, cv::Point(0, -line[2] / line[1]),
             cv::Point(image_left.cols, -(line[2] + line[0] * image_left.cols) / line[1]),
             cv::Scalar(255, 255, 255));

    // Draw the corresponding point
    cv::circle(image_left, point, 5, cv::Scalar(0, 0, 255), -1);
  }

  // Display the images with points and epipolar lines
  cv::imshow("Left Image", image_left);
  cv::waitKey(0);
  cv::imshow("Right Image", image_right);
  cv::waitKey(0);
}

void export_3d_points_to_txt(std::string file_name, cv::Mat points)
{
  cv::FileStorage fs(file_name, cv::FileStorage::WRITE);
  fs << "points_3d" << points;
  fs.release();
}