#include "util.h"

#include <stdio.h>
#include <string>
#include <vector>
#include <fstream>

#include <opencv2/opencv.hpp>

#define PHOTO_WIDTH 3072
#define PHOTO_HEIGHT 2048
#define NEW_PHOTO_WIDTH 640
#define NEW_PHOTO_HEIGHT 480

#define VIDEO_WIDTH 3840
#define VIDEO_HEIGHT 2160
#define NEW_VIDEO_WIDTH 1280
#define NEW_VIDEO_HEIGHT 720

cv::Mat down_size_image(cv::Mat image, int width, int height)
{
    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(width, height));
    return resized_image;
}

bool is_image_blurred(cv::Mat image, double threshold)
{
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    cv::Mat laplacianImage;
    cv::Laplacian(gray, laplacianImage, CV_64F);

    cv::Scalar mean, stddev;
    cv::meanStdDev(laplacianImage, mean, stddev);

    if (stddev.val[0] * stddev.val[0] < threshold)
    {
        // std::cout << "The image is blurry." << std::endl;
        return true;
    }
    // std::cout << "The image is sharp." << std::endl;
    return false;
}

std::vector<ImageView> extract_frames_from_video(std::string directory, int step)
{
    std::vector<ImageView> images;
    cv::VideoCapture video(directory);

    if (!video.isOpened())
    {
        std::cout << "Cannot open the video file.." << std::endl;
        return images;
    }

    int frameNumber = 0;
    while (video.isOpened())
    {
        cv::Mat frame;
        bool success = video.read(frame);
        if (!success)
        {
            std::cout << "Video finished" << std::endl;
            break;
        }

        if (frameNumber % step == 0 && !is_image_blurred(frame, 10.0))
        {
            ImageView image;
            image.set_name("frame_" + std::to_string(frameNumber) + ".jpg");
            image.set_image(down_size_image(frame, NEW_VIDEO_WIDTH, NEW_VIDEO_HEIGHT));
            std::cout << "Frame " << frameNumber << " saved..." << std::endl;
            images.push_back(image);
        }
        frameNumber++;
    }

    video.release();
    return images;
}

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
        current_image.set_image(down_size_image(current_image.get_image(), NEW_PHOTO_WIDTH, NEW_PHOTO_HEIGHT));

        // add image to vector
        images.push_back(current_image);
    }

    return images;
}

void export_3d_points_to_txt(std::string file_name, cv::Mat points)
{
    cv::FileStorage fs(file_name, cv::FileStorage::WRITE);
    fs << "points_3d" << points;
    fs.release();
}