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

// If for some reasons the correct input command line format is not respected,
// then a error message is showed in output
void error_message() {
    std::cerr << "The command line arguments must be one of the following types:" << std::endl;
    std::cerr << "  1. image_path" << std::endl;
    std::cerr << "  2. image_path ground_truth_path" << std::endl;
    std::cerr << "     Note: ground truth as --> boat:264;371;342;362;" << std::endl;
    std::cerr << "" << std::endl;
    std::cerr << "For example:" << std::endl;
    std::cerr << "  01.jpg" << std::endl;
    std::cerr << "or" << std::endl;
    std::cerr << "  01.jpg 01.txt" << std::endl;
}

int main(int argc, char** argv)
{

    try {

        if (argc == 1) {
            error_message();
            return -1;
        }

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
    }
    catch (...){
        error_message();
    }

    cv::waitKey(0);
    return 0;
}