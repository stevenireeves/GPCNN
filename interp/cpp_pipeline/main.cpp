#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "GP.h"
#include "weights.h"

int main(int argc, char* argv[]){
	cv::Mat img = cv::imread(argv[1], cv::IMREAD_COLOR);
	const float del[2] = {0.01, 0.01}; 
	const int ratio[2] = {4, 4}; 
	weights wgts(ratio, del); 
	
}

