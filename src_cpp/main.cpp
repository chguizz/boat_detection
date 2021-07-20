/*
Chiara Guizzaro
id number: 2019293
Computer vision project, boat detection, a.a. 2020-21
University of Padua

Boat detection
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
    std::string ground_truth_path;
    if (argc == 3) ground_truth_path = argv[2];

    // Input image
    cv::Mat img = cv::imread(path);

    
    // Boat detection algorithm
    // In this case the image showed in output show boxes 
    // with respective probability given by the classifier that the box 
    // rapresents a boat
    BoatDetector bd;
    bd.set_image(img);
    cv::Mat result = bd.process();

    cv::namedWindow("Final result", cv::WINDOW_FREERATIO);
    cv::imshow("Final result", result);

    // Comparison with ground truth if there is a given path in input
    // In this case the image showed in output show boxes 
    // with respective pertanage of IoU
    if (argc == 3) {
        bd.load_ground_truth(ground_truth_path);

        cv::Mat comparison = bd.compare();
        cv::namedWindow("GT comparison", cv::WINDOW_FREERATIO);
        cv::imshow("GT comparison", comparison);
    }

    cv::waitKey(0);
    return 0;
}