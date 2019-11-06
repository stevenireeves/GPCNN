#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "GP.h"
#include "weights.h"

int main(int argc, char* argv[]){
	cv::Mat img = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
	int chan = img.channels()==3? CV_32FC3 : CV_32F;
	img.convertTo(img, chan);
	const float del[2] = {0.01, 0.01}; 
	const int ratio[2] = {4, 4}; 
	weights wgts(ratio, del);
	const int size[3] = {img.rows, img.cols, img.channels()}; 
	GP interp(wgts, size); 
	cv::Mat flat = img.reshape(1, img.total()*img.channels());
	std::vector<float> imgin = img.isContinuous()? flat : flat.clone();
	std::vector<float> imgout(size[0]*ratio[0]*size[1]*ratio[1]*size[2], 0);
	interp.MSinterp(imgin, imgout, ratio[0], ratio[1]); 
	std::cout <<"Superresolution completed" << std::endl; 
	cv::Mat img2 = cv::Mat(imgout).reshape(size[2], size[0]*ratio[0]);
//	cv::Mat img2 = cv::Mat(size[0]*ratio[0], size[1]*ratio[1], chan, imgout.data()); 
	
	int uchan = img.channels()==3? CV_8UC3: CV_8U; 
	img2.convertTo(img2, uchan); 
	cv::imshow("testMat",img2);
	cv::waitKey(0);
}

