/*
Chiara Guizzaro
id number: 2019293
Computer vision project, boat detection, a.a. 2020-21
University of Padua

Regional proposal class
*/

#include "region_proposal.h"

RegionProposal::RegionProposal() {
	// intentionaly blank
}

std::vector<cv::Scalar> RegionProposal::process(cv::Mat img, std::string mode) {

	
	int d = 15;
	int	sigmaColor = 5000;
	int	sigmaSpace = 1500;
	cv::Mat blur_img, blur_img2, rgb_image;
	cv::cvtColor(img, rgb_image, cv::COLOR_BGR2RGB);
	cv::medianBlur(rgb_image, blur_img, 15);
	cv::bilateralFilter(blur_img, blur_img2, d, sigmaColor, sigmaSpace);
	
	if (mode == "Selective_search_quality") {
		// Initial set up of the algorithm
		cv::Ptr<cv::ximgproc::segmentation::SelectiveSearchSegmentation> ss = 
			cv::ximgproc::segmentation::createSelectiveSearchSegmentation();
		ss->setBaseImage(blur_img2);
		ss->switchToSelectiveSearchQuality();

		// run selective search segmentation on given image
		std::vector<cv::Rect> rects;
		ss->process(rects);
		return changing_coordinates(rects);
	}
	else if (mode == "Selective_search_fast") {
		// Initial set up of the algorithm
		cv::Ptr<cv::ximgproc::segmentation::SelectiveSearchSegmentation> ss = 
			cv::ximgproc::segmentation::createSelectiveSearchSegmentation();

		ss->setBaseImage(blur_img2);
		ss->switchToSelectiveSearchFast();

		// run selective search segmentation on given image
		std::vector<cv::Rect> rects;
		ss->process(rects);
		return changing_coordinates(rects);
	}
	else {
		std::cout << "Error in the choice of the algorithm for regional proposal." << std::endl;
		exit(-1);
	}
}

std::vector<cv::Scalar> RegionProposal::changing_coordinates(const std::vector<cv::Rect> rects) {
	std::vector<cv::Scalar> boxes;
	for (int i = 0; i < rects.size(); i++) {
		cv::Rect box = rects.at(i);
		int br_x = box.x + box.width;
		int br_y = box.y + box.height;

		boxes.push_back(cv::Scalar(box.x, box.y, br_x, br_y));
	}
	return boxes;
}