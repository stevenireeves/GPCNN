#include <stdio.h>
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
        std::cout<< "Make Interp obj" << std::endl; 	
	GP interp(wgts, size); 
	cv::Mat flat = img.reshape(1, img.total()*img.channels());
	std::vector<float> imgin = img.isContinuous()? flat : flat.clone();
	std::cout<<img.total()<<std::endl;
	std::vector<float> imgout(size[0]*ratio[0]*size[1]*ratio[1]*size[2], 0);
	std::cout << " Interpolate " << std::endl; 
	interp.MSinterp(imgin, imgout, ratio[0], ratio[1]); 
	std::cout << "Superresolution completed" << std::endl; 
	cv::Mat img2 = cv::Mat(imgout).reshape(size[2], size[1]*ratio[1]);
	img2 *= 255; 
//	cv::Mat img2 = cv::Mat(size[0]*ratio[0], size[1]*ratio[1], chan, imgout.data()); 
	
	int uchan = img.channels()==3? CV_8UC3: CV_8U; 
	img2.convertTo(img2, uchan); 
	cv::imwrite("test.jpg", img2);  
/*	cv::imshow("testMat",img2);
	cv::waitKey(0);*/ 
}

