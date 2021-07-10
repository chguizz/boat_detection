/*
Chiara Guizzaro
id number: 2019293
Computer vision project, boat detection, a.a. 2020-21
University of Padua

Boat detector class
*/

#include <iostream>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "region_proposal.h"

class BoatDetector {
public:

	// Public constructor
	BoatDetector();

	// Setting the image to process
	// ---------------------
	// Parameters:
	//     cv::Mat image                       Image to process
	void set_image(cv::Mat image);

	// Load the ground truth bounding boxes
	// ---------------------
	// Parameters:
	//     std::string path                    Path where there is a .txt file with bounding boxes
	//                                         boat:x_tl;x_br;y_tl;y_tr; (tl = top left, br = bottom right)
	//                                         Example: boat:264;371;342;362;
	void load_ground_truth(std::string path);

	// Get ground truth boxes
	// ---------------------
	// Returns:
	//     std::vector<cv::Scalar> true_boxes  Ground truth boxes of the type [x_tl, y_tl, x_br, y_br]
	std::vector<cv::Scalar> get_ground_truth();

	// Process the image as proposed by the algorithm
	// 1° step: object localization by regions proposal
	// 2° step: object classification by a chosen clissifier
	// 3° step: merge the previous results and apply Non Maxima Supression (NMS)
	// 4° step: draw all remained bounding boxes with relative score
	// ---------------------
	// Returns:
	//     cv::Mat final_image                 Final image with all bounding boxes and scores for relative objctes
	cv::Mat process();

private:
	// Image to process
	cv::Mat img;

	// Ground truth bounding boxes
	std::vector<cv::Scalar> true_boxes;

	// Proposed regions
	std::vector<cv::Scalar> boxes;

	// Vector of scores for each proposed region
	std::vector<float> scores;

	// Usefull function to split a string
	std::vector<std::string> split(const char* str, char c = ' ');

	// Interseption Over Union metric (IoU)
	// ---------------------
	// Parameters:
	//     cv::Scalar boxA                     First box to consider for IoU
	//     cv::Scalar boxB                     Second box to consider for IoU
	// Returns:
	//     float score                         Floating number representing the IoU for the two boxes
	//                                         given as parameters.
	// Cite:
	//     https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
	float iou(cv::Scalar boxA, cv::Scalar boxB);

	// Extract a patch given a certain bounding box
	// ---------------------
	// Parameters:
	//     cv::Mat image                       The image from which to extract the patch
	//     cv::Scalar box                      Bounding box used to extract the patch
	//     bool resize = true                  Boolean variable to decide wheter the patch must be
	//                                         resized or not
	//     cv::Size sz = cv::Size(224, 224)    If resize = true, then the patch is resized according to sz dimentions
	// Returns:
	//     cv::Mat patch                       Resulting patch
	cv::Mat patch(cv::Mat image, cv::Scalar box, bool resize = true, cv::Size sz = cv::Size(224, 224));

	// Draw a given bounding box on an image
	//  ---------------------
	// Parameters:
	//     cv::Mat image                        Image where to draw the bounding box
	//     cv::Scalar box                       Given bounding box
	//     cv::Scalar color                     Color of the bounding box
	//     std::string text                     Text to apply over the box (for example a score, probability etc..)
	void draw_box(cv::Mat image, cv::Scalar box, cv::Scalar color, std::string text);
};