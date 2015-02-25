#include "config.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include "facedetect.h"


int main(int argc, char *argv[]) {
	cv::VideoCapture *vc = new cv::VideoCapture[CAM_NUM];
	cv::Mat *frames = new cv::Mat[CAM_NUM];

	cv::FaceDetection faceCascade;

	// Initialize video capture
	for(int i=0; i<CAM_NUM; i++) {
		if(!vc[i].isOpened()) {
			std::cout << "Open " << i << std::endl;
			if(!vc[i].open("Wildlife.wmv")) {
				std::cout << "Cannot open the movie file" << std::endl;
				exit(1);
			}
		}
	}

	if(!faceCascade.load(FACE_CASCADE_NAME) || faceCascade.empty()) {
		std::cout << "Cannnot open cascade file" << std::endl;
		exit(1);
	}


	while(1) {
		for(int i=0; i<CAM_NUM; i++) {
			vc[i] >> frames[i];
			if(frames[i].empty()) continue;
			cv::Mat grayFrame;
			std::vector<cv::Rect> faces;

			cv::cvtColor(frames[i], grayFrame, CV_BGR2GRAY);

			faceCascade.detectMultiScaleMultiCamera(grayFrame, faces);

			for(int j=0; j<faces.size(); j++) {
				cv::rectangle(frames[i], faces[j], cv::Scalar(255, 0, 255));
			}

			cv::imshow(cv::format("Frame %d", i+1), frames[i]);
			cv::moveWindow(cv::format("Frame %d", i+1), i, i*100);
		}
		if(cv::waitKey(1) == ' ') break;
	}


	return 0;
}