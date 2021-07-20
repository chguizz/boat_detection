/*
Chiara Guizzaro
id number: 2019293
Computer vision project, boat detection, a.a. 2020-21
University of Padua

Boat detector class
*/

#include "boat_detector.h"

cv::Mat BoatDetector::process() {

	// *************** Region proposal algorithm *****************************
	std::cout << "Region proposal algorithm" << std::endl;
	RegionProposal region_detector;
	std::vector<cv::Scalar> boxes = region_detector.process(img, "Selective_search_fast");

	std::cout << "    --> number of regions: " << boxes.size() << std::endl;


	// Check ratio of box proposed
	std::vector<cv::Scalar> new_boxes;
	for (int i = 0; i < boxes.size(); i++) {
		cv::Scalar box = boxes.at(i);

		float height = box[3] - box[1];
		float width = box[2] - box[0];
		float ratio1 = height / width;
		float ratio2 = width / height;

		if (!(ratio1 < 0.2 || ratio2 < 0.2)) {
			new_boxes.push_back(box);
		}
	}
	boxes = new_boxes;
	

	// *************** Classification of bounding boxes ***********************
	std::cout << "Classification" << std::endl;
	// Loading a net
	cv::dnn::Net net = cv::dnn::readNetFromTensorflow("frozen_graph.pb");

	cv::Mat rgb_img;
	cv::cvtColor(img, rgb_img, cv::COLOR_BGR2RGB);
	for (int i = 0; i < boxes.size(); i++) {
		// Extract patch given a box
		cv::Mat region = patch(rgb_img, boxes.at(i), false);

		// Blob construction
		bool swapRB = false;
		cv::Size image_size(224, 224);
		cv::Mat blob = cv::dnn::blobFromImage(region, 1, image_size, cv::Scalar(), swapRB);
		net.setInput(blob);

		// Forward propagation
		cv::Mat out = net.forward();
		float s = out.at<float>(0);
		scores.push_back(s);
	}
	
	// *************** Non Maxima Supression (NMS) ******************************
	std::cout << "Non Maxima Supression (NMS)" << std::endl;
	
	// Changing bounding boxes to have format [x, y, height, width]
	std::vector<cv::Rect> boxes_rect;
	for (int i = 0; i < boxes.size(); i++) {
		cv::Scalar box = boxes.at(i);
		cv::Rect rect = cv::Rect(cv::Point(box[0], box[1]), cv::Point(box[2], box[3]));
		boxes_rect.push_back(rect);
	}


	// Apply NMS as implemented in OpenCV
	std::vector<int> indices_keept;
	float th_score = 0.9975;
	do {
		cv::dnn::NMSBoxes(boxes_rect, scores, th_score, 0.01, indices_keept);
		th_score -= 0.01;
	} while (indices_keept.size() == 0 && th_score >= 0.7);


	std::vector<cv::Rect> temp_boxes_rect;
	std::vector<float> temp_scores;
	for (int i = 0; i < indices_keept.size(); i++) {
		int index = indices_keept.at(i);
		temp_boxes_rect.push_back(boxes_rect.at(index));
		temp_scores.push_back(scores.at(index));
	}
	boxes_rect = temp_boxes_rect;
	scores = temp_scores;


	// *************** Draw all remained bounding boxes with relative score ******
	for (int i = 0; i < boxes_rect.size(); i++) {
		cv::Rect box = boxes_rect.at(i);
		char temp_s[10];
		std::sprintf(temp_s, "%.2f%%", scores.at(i) * 100);
		std::string s = temp_s;

		// Drawing box
		draw_box(img, box, cv::Scalar(0, 255, 0), s);

		// Store final result
		pred_boxes.push_back(cv::Scalar(box.x, box.y, box.x + box.width, box.y + box.height));
	}

	return img;
}