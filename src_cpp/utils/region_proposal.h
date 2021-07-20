/*
Chiara Guizzaro 
id number: 2019293
Computer vision project, boat detection, a.a. 2020-21
University of Padua

Regional proposal class
*/

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <opencv2/ximgproc/segmentation.hpp>
#include <opencv2/ximgproc.hpp>

class RegionProposal {
public:

	// Public constructor
	RegionProposal();



	// Regional proposal for object detection given an image
	// the algorithm proposed is SelectiveSearch in fast mode (default) or quality mode
	// ---------------
	// Parameters:
	// 	   cv::Mat img                             image to process
	// 	   std::string mode                        algorithm to use
	// 	                                           posssible choices are 
	// 	                                           "Selective_search_quality", "Selective_search_fast" (default)
	// Returns:
	//     std::vector<cv::Scalar> boxes           regions proposal
	std::vector<cv::Scalar> process(cv::Mat img, std::string mode = "Selective_search_fast");

private:
	// Changing from notation (x, y, w, h) to (tl_x, tl_y, br_x, br_y)
	// top left corner coordinates and
	// bottom right coordinates
	// ---------------
	// Parameters:
	//     const std::vector<cv::Rect> rects        bounding boxes of type (x, y, w, h)
	// Returns:
	//     std::vector<cv::Scalar>     boxes        bounding boxes of type (tl_x, tl_y, br_x, br_y)
	std::vector<cv::Scalar> changing_coordinates(const std::vector<cv::Rect> rects);
};