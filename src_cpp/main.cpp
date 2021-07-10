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
    cv::Mat image = cv::imread(path);


    BoatDetector bd;
    bd.load_ground_truth("C:\\Users\\guizz\\Desktop\\boat_detection\\dataset\\FINAL_DATASET\\TEST_DATASET\\venice_labels_txt\\11.txt");


    
    

    cv::waitKey(0);
    return 0;
}