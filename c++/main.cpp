#include <stdio.h>
#include <string>
#include <iostream>
#include <filesystem>
#include <vector>


#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#define IMG_DIR "../images"

enum featureDetection {
    None,
    SIFT,
    SURF,
    FAST,
    ORB
};

struct Image {
    std::string name;
    cv::Mat image;

    featureDetection featureDetection = None;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
};

std::vector<Image> getImages(std::string directory, std::vector<Image> images){
    std::string currentFile = "";
    Image currentImage;

    // get all images within the directory
    for (const auto & entry : std::__fs::filesystem::directory_iterator(directory)){
        currentFile = entry.path();
        currentImage.name = currentFile.substr(10);
        currentImage.image = cv::imread(currentFile, cv::IMREAD_UNCHANGED);

        // check image for corrent format or existence
        if (currentImage.image.data == NULL){
            printf("Image not found or incorrect file type!\n");
            continue;
        }

        // add image to vector
        images.push_back(currentImage);
    }

    return images;
}


void getKeyPoints(std::vector<Image>& images, featureDetection type){
    switch(type){
        case SIFT:
        {
            std::cout << "Applying SIFT ..." << std::endl;

            cv::Ptr<cv::SIFT> detector = cv::SIFT::create();
            for(int i = 0; i < images.size(); i++){
                images[i].featureDetection = type;
                detector->detectAndCompute(images[i].image, cv::noArray(), images[i].keypoints, images[i].descriptors);
            }
            break;
        }
        case SURF:
        {
            std::cout << "Applying SURF ..." << std::endl;

            cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create();
            for(int i = 0; i < images.size(); i++){
                images[i].featureDetection = type;
                detector->detect(images[i].image, images[i].keypoints);
                detector->detectAndCompute(images[i].image, cv::noArray(), images[i].keypoints, images[i].descriptors);
            }
            break;
        }
        case FAST:
        {
            std::cout << "Applying FAST ..." << std::endl;

            cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create();
            for(int i = 0; i < images.size(); i++){
                images[i].featureDetection = type;
                detector->detectAndCompute(images[i].image, cv::noArray(), images[i].keypoints, images[i].descriptors);
            }
            break;
        }
        case ORB:
        {
            std::cout << "Applying ORB ..." << std::endl;

            cv::Ptr<cv::ORB> detector = cv::ORB::create();
            for(int i = 0; i < images.size(); i++){
                images[i].featureDetection = type;
                detector->detectAndCompute(images[i].image, cv::noArray(), images[i].keypoints, images[i].descriptors);
            }
            break;
        }
        default:
            break;
    }
}



int main(int argc, char **argv){
    std::vector <Image> imagesRaw;

    imagesRaw = getImages(IMG_DIR, imagesRaw);

    for(int i = 0; i < imagesRaw.size();i++){
        std::cout << imagesRaw[i].name << "\n";
    }

    getKeyPoints(imagesRaw, ORB);

    cv::Mat img_kp;
    cv::drawKeypoints(imagesRaw[0].image, imagesRaw[0].keypoints, img_kp);
    cv::imwrite("../images/sift_detector2.jpg", img_kp);
    std::cout << "Hello" << "\n";
}