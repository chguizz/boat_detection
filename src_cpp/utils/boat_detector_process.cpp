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
	cv::dnn::NMSBoxes(boxes_rect, scores, 0.90, 0.01, indices_keept);



	// *************** SBS (small box supression) *******************************
	// It may happen that during the boats detection process when the algorithm
	// is evaluating a large boat, that some details such as windows etc..
	// are wrongly classified as boats, but not discarded with NMS, because too much
	// small w.r.t. the whole boat. For reason there is small box supression such that
	// if a small box is completly inside a bigger one, then it's removed.
	// This reduce the number of false positive boxes.
	// --------------
	// In general:
	// If a box B is completly inside a box A, then box B is discarded
	std::cout << "Small box supression (SBS)" << std::endl;
	std::vector<cv::Rect> temp_boxes_rect;
	std::vector<float> temp_scores;
	for (int i = 0; i < indices_keept.size(); i++) {
		int index = indices_keept.at(i);
		temp_boxes_rect.push_back(boxes_rect.at(index));
		temp_scores.push_back(scores.at(index));
	}
	boxes_rect = temp_boxes_rect;
	scores = temp_scores;

	std::vector<cv::Rect> sbs_boxes;
	std::vector<float> sbs_scores;
	SBS(boxes_rect, scores, sbs_boxes, sbs_scores);



	// *************** Draw all remained bounding boxes with relative score ******
	for (int i = 0; i < sbs_boxes.size(); i++) {
		cv::Rect box = sbs_boxes.at(i);
		char temp_s[10];
		std::sprintf(temp_s, "%.2f%%", sbs_scores.at(i) * 100);
		std::string s = temp_s;

		// Drawing box
		draw_box(img, box, cv::Scalar(0, 255, 0), s);
	}

	return img;
}