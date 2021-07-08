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

#include "utils/region_proposal.h"
#include "utils/classifier.h"

int main(int argc, char** argv)
{

    std::string path = argv[1];
    std::string mode = argv[2];

    // Input image
    cv::Mat image = cv::imread(path);
    cv::imshow("Input image", image);

    /*
    // Region proposal
    RegionProposal rp;
    std::vector<cv::Scalar> boxes = rp.process(image, mode);

    // Draw bounding boxes
    for (int i = 0; i < boxes.size(); i++) {
        int thickness = 2;
        cv::Scalar color = cv::Scalar(59, 235, 255, 0);
        cv::Scalar box = boxes.at(i);
        cv::rectangle(image, cv::Point(box[0], box[1]), cv::Point(box[2], box[3]), color, thickness);
    }

    // Show Bounding boxes
    cv::imshow("Boxes", image);
    */
    

    cv::waitKey(0);
    return 0;
}