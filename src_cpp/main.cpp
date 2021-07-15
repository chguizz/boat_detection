/*
Chiara Guizzaro
id number: 2019293
Computer vision project, boat detection, a.a. 2020-21
University of Padua

Boats detection
*/

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utils/filesystem.hpp>

#include "utils/boat_detector.h"

int main(int argc, char** argv)
{
    std::string path = argv[1];

    // Input image
    cv::Mat img = cv::imread(path);

    
    // Boats detection algorithm
    BoatDetector bd;
    bd.set_image(img);
    cv::Mat result = bd.process();

    cv::namedWindow("Final result", cv::WINDOW_FREERATIO);
    cv::imshow("Final result", result);
    

    cv::waitKey(0);
    return 0;
}