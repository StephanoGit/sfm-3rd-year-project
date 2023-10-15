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

enum featureMatching {
    BF,
    FLANN
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


void computeKeyPointsDescriptors(std::vector<Image>& images, featureDetection type){
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

            cv::Ptr<cv::xfeatures2d::SurfFeatureDetector> detector = cv::xfeatures2d::SurfFeatureDetector::create();
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
            std::cout << "Please provide a valid feature detector (SIFT, SURF, FAST or ORB) ..." << std::endl;
            break;
    }
}


std::vector<cv::DMatch> matchDescriptors(cv::Mat des1, cv::Mat des2, cv::DescriptorMatcher::MatcherType type){
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(type);
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher->knnMatch(des1, des2, knn_matches, 2);

    const float ratio_th = 0.7f;
    std::vector<cv::DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++){
        if (knn_matches[i][0].distance < ratio_th * knn_matches[i][1].distance){
             good_matches.push_back(knn_matches[i][0]);
        }
    }
    return good_matches;
}



int main(int argc, char **argv){
    std::vector <Image> imagesRaw;
    std::vector <cv::DMatch> matches;

    imagesRaw = getImages(IMG_DIR, imagesRaw);

    for(int i = 0; i < imagesRaw.size();i++){
        std::cout << imagesRaw[i].name << "\n";
    }

    computeKeyPointsDescriptors(imagesRaw, SURF);
    matches = matchDescriptors(imagesRaw[0].descriptors, imagesRaw[1].descriptors, cv::DescriptorMatcher::FLANNBASED);

    cv::Mat img_matches;
    cv::drawMatches(imagesRaw[0].image, imagesRaw[0].keypoints, imagesRaw[1].image, imagesRaw[1].keypoints, matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    cv::imshow("Good matches", img_matches);
    cv::waitKey();
    return 0;
}