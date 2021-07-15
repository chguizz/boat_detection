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

void BoatDetector::SBS(std::vector<cv::Rect> boxes, std::vector<float> scores, 
	std::vector<cv::Rect> &new_boxes, std::vector<float> &new_scores) {

	// Area of each box
	std::vector<float> areas;
	for (int i = 0; i < boxes.size(); i++) {
		cv::Rect b = boxes.at(i);
		areas.push_back(b.height * b.width);
	}
		
	while (boxes.size() != 0) {
		int m = argmax(areas);
		cv::Rect M = boxes.at(m);

		new_boxes.push_back(M);
		boxes.erase(boxes.begin() + m);

		new_scores.push_back(scores.at(m));
		scores.erase(scores.begin() + m);
		areas.erase(areas.begin() + m);

		std::vector<cv::Rect> temp_boxes;
		std::vector<float> temp_scores, temp_areas;
		for (int i = 0; i < boxes.size(); i++) {
			if (!inside(M, boxes.at(i))) {
				temp_boxes.push_back(boxes.at(i));
				temp_scores.push_back(scores.at(i));
				temp_areas.push_back(areas.at(i));
			}
		}
		boxes = temp_boxes;
		scores = temp_scores;
		areas = temp_areas;

	}
}

int BoatDetector::argmax(std::vector<float> A) {
	int m = 0;
	float max = A.at(m);

	for (int i = 0; i < A.size(); i++) {
		if (A.at(i) > max) {
			m = i;
			max = A.at(m);
		}
	}

	return m;
}

bool BoatDetector::inside(cv::Rect boxA, cv::Rect boxB) {
	// box A is the LARGE BOX
	// box B is the SMALLER BOX, eventually inside box A

	bool first_condition = (boxB.x > boxA.x) && (boxB.y > boxA.y) 
		&& (boxB.x + boxB.width < boxA.x + boxA.width) && (boxB.y + boxB.height < boxA.y + boxA.height);

	bool second_condition = (boxB.x >= boxA.x) && (boxB.y >= boxA.y)
		&& (boxB.x + boxB.width < boxA.x + boxA.width) && (boxB.y + boxB.height < boxA.y + boxA.height);

	bool third_condition = (boxB.x > boxA.x) && (boxB.y > boxA.y)
		&& (boxB.x + boxB.width <= boxA.x + boxA.width) && (boxB.y + boxB.height <= boxA.y + boxA.height);

	return first_condition || second_condition || third_condition;
}