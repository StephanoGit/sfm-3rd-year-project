#include "../include/util.h"

#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/core/persistence.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>
#include <stdio.h>

#define IPHONE_PHOTO_WIDTH 3024
#define IPHONE_PHOTO_HEIGHT 4032
#define IPHONE_NEW_PHOTO_WIDTH 756
#define IPHONE_NEW_PHOTO_HEIGHT 1008

#define PHOTO_WIDTH 3072
#define PHOTO_HEIGHT 2048
#define NEW_PHOTO_WIDTH 640
#define NEW_PHOTO_HEIGHT 480

#define VIDEO_WIDTH 3840
#define VIDEO_HEIGHT 2160
#define NEW_VIDEO_WIDTH 1280
#define NEW_VIDEO_HEIGHT 720

cv::Mat downscale_image(cv::Mat image, int width, int height) {
  cv::Mat resized_image;
  cv::resize(image, resized_image, cv::Size(width, height));
  return resized_image;
}

bool is_image_blurred(cv::Mat image, double threshold) {
  cv::Mat gray;
  cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

  cv::Mat laplacianImage;
  cv::Laplacian(gray, laplacianImage, CV_64F);

  cv::Scalar mean, stddev;
  cv::meanStdDev(laplacianImage, mean, stddev);

  if (stddev.val[0] * stddev.val[0] < threshold) {
    // std::cout << "The image is blurry." << std::endl;
    return true;
  }
  // std::cout << "The image is sharp." << std::endl;
  return false;
}

std::vector<cv::Mat> video_to_images(std::string directory, int step) {
  std::vector<cv::Mat> images;
  cv::VideoCapture video(directory);

  if (!video.isOpened()) {
    std::cout << "Cannot open the video file.." << std::endl;
    return images;
  }

  int frameNumber = 0;
  while (video.isOpened()) {
    cv::Mat frame;
    bool success = video.read(frame);
    if (!success) {
      std::cout << "Video finished" << std::endl;
      break;
    }

    if (frameNumber % step == 0 && !is_image_blurred(frame, 10.0)) {
      images.push_back(
          downscale_image(frame, NEW_PHOTO_WIDTH, NEW_PHOTO_HEIGHT));
    }
    frameNumber++;
  }

  video.release();
  return images;
}

bool sortByName(const std::__fs::filesystem::directory_entry &entry1,
                const std::__fs::filesystem::directory_entry &entry2) {
  return entry1.path().filename() < entry2.path().filename();
}

std::vector<cv::Mat> load_images(std::string directory, bool downscale) {
  std::string current_file = "";
  std::vector<cv::Mat> images;
  cv::Mat current_image;

  std::vector<std::__fs::filesystem::directory_entry> entries;
  for (const auto &entry :
       std::__fs::filesystem::directory_iterator(directory)) {
    entries.push_back(entry);
  }

  // Sort entries by name
  std::sort(entries.begin(), entries.end(), sortByName);

  // get all images within the directory
  for (const auto &entry : entries) {
    current_file = entry.path();
    current_image = cv::imread(current_file, cv::IMREAD_UNCHANGED);

    // check image for corrent format or existence
    if (current_image.data == NULL) {
      printf("Image not found or incorrect file type!\n");
      continue;
    }

    // resize
    if (downscale) {
      current_image =
          downscale_image(current_image, NEW_PHOTO_WIDTH, NEW_PHOTO_HEIGHT);
    }

    // add image to vector
    images.push_back(current_image);
  }

  return images;
}

void export_point_cloud(std::vector<PointCloudPoint> point_cloud,
                        std::string file_name) {

  std::vector<cv::Point3f> points;
  for (size_t i = 0; i < point_cloud.size(); i++) {
    points.push_back(point_cloud[i].point);
  }

  cv::FileStorage fs("../points-3d/" + file_name, cv::FileStorage::WRITE);
  fs << "points" << points;
  fs.release();
}

void export_intrinsics(cv::Mat K, cv::Mat d) {
  cv::FileStorage fs_K("../calibration/K.xml", cv::FileStorage::WRITE);
  fs_K << "K" << K;
  fs_K.release();

  cv::FileStorage fs_d("../calibration/d.xml", cv::FileStorage::WRITE);
  fs_d << "d" << d;
  fs_d.release();
}
