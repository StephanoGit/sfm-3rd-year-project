#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc, char **argv){
    cv::Mat grayImg, colorImg;

    cv::namedWindow("img", cv::WINDOW_AUTOSIZE);

    colorImg = cv::imread("../images/0.jpg", cv::IMREAD_COLOR);
    cv::cvtColor(colorImg, grayImg, cv::COLOR_BGR2GRAY);
    cv::imwrite("../images/1.jpg", grayImg);

    cv::imshow("img", grayImg);

    cv::waitKey(0);
    cv::destroyAllWindows();
}