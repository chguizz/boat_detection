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

std::vector<cv::Scalar> RegionProposal::process(cv::Mat img, std::string mode, cv::Mat custom_edges) {
	// EdgeBoxes requires a model: 
    // https://github.com/opencv/opencv_extra/blob/master/testdata/cv/ximgproc/model.yml.gz

	if (mode == "EdgeBoxes") {
		std::string model = "model.yml.gz";
		img.convertTo(img, cv::DataType<float>::type, 1 / 255.0);
		cv::Mat edges(img.size(), img.type());
		std::vector<cv::Rect> rects;

		// Edges detection
		cv::Ptr<cv::ximgproc::StructuredEdgeDetection> edge_detection = 
			cv::ximgproc::createStructuredEdgeDetection(model);
		edge_detection->detectEdges(img, edges);
		
		// Computation of orientation from edge map
		cv::Mat orimap;
		edge_detection->computeOrientation(edges, orimap);

		// Edge Boxes
		cv::Ptr< cv::ximgproc::EdgeBoxes > edge_boxes = cv::ximgproc::createEdgeBoxes();
		edge_boxes->setMaxBoxes(30);
		edge_boxes->getBoundingBoxes(edges, orimap, rects);

		return changing_coordinates(rects);
	}
	else if (mode == "EdgeBoxes_custom") {
		std::string model = "model.yml.gz";
		std::vector<cv::Rect> rects;

		// Edges detection
		cv::Ptr<cv::ximgproc::StructuredEdgeDetection> edge_detection =
			cv::ximgproc::createStructuredEdgeDetection(model);
		edge_detection->detectEdges(img, custom_edges);

		// Computation of orientation from edge map
		cv::Mat orimap;
		edge_detection->computeOrientation(custom_edges, orimap);

		// Edge Boxes
		cv::Ptr< cv::ximgproc::EdgeBoxes > edge_boxes = cv::ximgproc::createEdgeBoxes();
		edge_boxes->setMaxBoxes(30);
		edge_boxes->getBoundingBoxes(custom_edges, orimap, rects);

		return changing_coordinates(rects);
	}
	else if (mode == "Selective_search_quality") {
		// Initial set up of the algorithm
		cv::Ptr<cv::ximgproc::segmentation::SelectiveSearchSegmentation> ss = 
			cv::ximgproc::segmentation::createSelectiveSearchSegmentation();
		ss->setBaseImage(img);
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
		ss->setBaseImage(img);
		ss->switchToSelectiveSearchFast();

		// run selective search segmentation on given image
		std::vector<cv::Rect> rects;
		ss->process(rects);
		return changing_coordinates(rects);
	}
	else {
		std::cout << "Error in the choice of the algorithm for regional proposal." << std::endl;
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