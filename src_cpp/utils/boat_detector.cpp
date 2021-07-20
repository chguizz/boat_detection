/*
Chiara Guizzaro
id number: 2019293
Computer vision project, boat detection, a.a. 2020-21
University of Padua

Boat detector class
*/

#include "boat_detector.h"

BoatDetector::BoatDetector() {
	// Intentionally blanck
}

void BoatDetector::set_image(cv::Mat image) {
	img = image.clone();
	comparison = image.clone();
}

void BoatDetector::load_ground_truth(std::string path) {
	// Order:
	// x coordinate top left corner
	// x coordinate bottom right corner
	// y coordinate top left corner
	// y coordinate bottom right corner

	std::fstream myfile;
	myfile.open(path);
	if (!myfile) {
		std::cout << "Error, file not found" << std::endl;
		std::exit(-1);
	}
	else {
		std::string line;
		while (std::getline(myfile, line)) {
			// Remove row header such as "boat:"
			std::string token;
			int i = line.find(":");
			token = line.substr(i + 1, line.size());

			// Taking coordinates values
			const char* c_token = token.c_str();
			std::vector<std::string> segments = split(c_token, ';');
			cv::Scalar box(stoi(segments.at(0)), stoi(segments.at(2)),
				stoi(segments.at(1)), stoi(segments.at(3)));

			true_boxes.push_back(box);
		}
	}
	myfile.close();
}

std::vector<std::string> BoatDetector::split(const char* str, char c){
	std::vector<std::string> result;
	do {
		const char* begin = str;
		while (*str != c && *str){
			str++;
		}
		result.push_back(std::string(begin, str));
	} while (0 != *str++);

	return result;
}

std::vector<cv::Scalar> BoatDetector::get_ground_truth() {
	return true_boxes;
}

float BoatDetector::iou(cv::Scalar boxA, cv::Scalar boxB) {
	// Compute the IOU metric
	// -----------------
	
	// computation of top left and bottom right coordinates
	// for the interseption rectangle
	int xi_1 = std::max(boxA[0], boxB[0]);
	int yi_1 = std::max(boxA[1], boxB[1]);
	int xi_2 = std::min(boxA[2], boxB[2]);
	int yi_2 = std::min(boxA[3], boxB[3]);

	// interseption area
	float area_i = std::max(0, xi_2 - xi_1 + 1) * std::max(0, yi_2 - yi_1 + 1);

	// areas box A and box B
	float area_A = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1);
	float area_B = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1);

	// Interseption over Union
	float iou = area_i / float(area_A + area_B - area_i);

	return iou;
}

cv::Mat BoatDetector::patch(cv::Mat image, cv::Scalar box, bool resize, cv::Size sz) {
	// It returns the corresponding patch given a bounding boxand an image
	// The box has structure[xa, ya, xb, yb]
	// (xa, ya) top left corner
	// (xb, yb) bottom right corner

	int xa = box[0];
	int ya = box[1];
	int xb = box[2];
	int yb = box[3];

	cv::Mat result = image(cv::Range(ya, yb), cv::Range(xa, xb));

	// Bilinear Interpolation
	if (resize) cv::resize(result, result, sz);

	return result;
}

void BoatDetector::draw_box(cv::Mat image, cv::Rect box, cv::Scalar color, std::string text) {
	int thickness = 2;
	cv::rectangle(image, box, color, thickness);
	thickness = 2;
	int x = box.x;
	int y = box.y;
	if (y - 10 < 0) y = box.y + box.height - 10;
	cv::putText(image, text, cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, thickness);
}


cv::Mat BoatDetector::compare() {
	
	// Draw all true boxes
	for (int i = 0; i < true_boxes.size(); i++) {
		cv::Scalar box = true_boxes.at(i);
		int thickness = 2;
		cv::rectangle(comparison, cv::Point(box[0], box[1]), cv::Point(box[2], box[3]),
			cv::Scalar(0, 0, 0), thickness);
	}
	
	
	// Compute IoU for each predicted box
	std::vector<float> list_iou;
	for (int i = 0; i < pred_boxes.size(); i++) {
		float max = 0;
		for (int j = 0; j < true_boxes.size(); j++) {
			float local_iou = iou(pred_boxes.at(i), true_boxes.at(j));
			if (local_iou > max) max = local_iou;
		}
		list_iou.push_back(max);
	}


	// Draw all predicted boxes
	for (int i = 0; i < pred_boxes.size(); i++) {
		cv::Scalar box = pred_boxes.at(i);
		int thickness = 2;
		cv::Scalar color;

		float local_iou = list_iou.at(i);
		if (local_iou >= 0.9) {
			color = cv::Scalar(0, 255, 0);       // green
		}
		else if (local_iou >= 0.7) {
			color = cv::Scalar(59, 235, 255);    // yellow
		}
		else if (local_iou >= 0.5) {
			color = cv::Scalar(3, 136, 252);     // orange
		}
		else {
			color = cv::Scalar(0, 0, 255);       // red
		}

		cv::rectangle(comparison, cv::Point(box[0], box[1]), cv::Point(box[2], box[3]),
			color, thickness);

		int x = box[0];
		int y = box[1];
		if (y - 10 < 0) y += box[3] - 10;
		char temp_s[10];
		std::sprintf(temp_s, "%.2f%%", local_iou * 100);
		std::string s = temp_s;

		cv::putText(comparison, s, cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, thickness);
	}

	return comparison;
}