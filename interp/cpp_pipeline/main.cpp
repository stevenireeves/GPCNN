#include <stdio.h>
#include <chrono>

#include <opencv2/opencv.hpp>

#include "GP.h"
#include "weights.h"

int main(int argc, char* argv[]){


	cv::Mat img = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
	int chan = img.channels()==3? CV_32FC3 : CV_32F;
	img.convertTo(img, chan);
	img /= 255;
	const float del[2] = {1.f/float(img.cols), 1.f/float(img.rows)}; 
	const int ratio[2] = {4, 4}; 
	weights wgts(ratio, del);
	const int size[3] = {img.cols, img.rows, img.channels()};
	GP interp(wgts, size); 
	cv::Mat flat = img.reshape(1, img.total()*img.channels());
	std::vector<float> imgin = img.isContinuous()? flat : flat.clone();
	std::vector<float> imgout(size[0]*ratio[0]*size[1]*ratio[1]*size[2], 0);
    auto start = std::chrono::high_resolution_clock::now(); 
	interp.gray_interp(imgin, imgout, ratio[0], ratio[1]); 
	auto stop = std::chrono::high_resolution_clock::now(); 
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start); 
	std::cout<< " Interp profile = " << duration.count() << " ms" << std::endl; 
//    interp.MSinterp(imgin, imgout, ratio[0], ratio[1]); 
	cv::Mat img2 = cv::Mat(imgout).reshape(size[2], size[1]*ratio[1]);
	img2 *= 255; 
	
	int uchan = img.channels()==3? CV_8UC3: CV_8U; 
	img2.convertTo(img2, uchan); 
	cv::imwrite("test_scale.png", img2);  
/*	cv::imshow("testMat",img2);
	cv::waitKey(0);*/ 
}

