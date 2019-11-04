#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "GP.h"
#include "weights.h"

int main(int argc, char* argv[]){
	cv::Mat img = cv::imread(argv[1]);
	img.convertTo(img, CV_32F);
        std::cout<<img.rows << '\t' << img.cols << '\t' << img.channels() << std::endl; 	
	const float del[2] = {0.01, 0.01}; 
	const int ratio[2] = {4, 4}; 
	weights wgts(ratio, del);
	std::cout<<"Weights Created" << std::endl; 
	const int size[3] = {img.rows, img.cols, img.channels()}; 
	GP interp(wgts, size); 
	std::cout<< "Interpolation Object Constructed" << std::endl; 
	std::vector<float> imgin;
	if (img.isContinuous()) {
	  imgin.assign((float*)img.data, (float*)img.data + img.total());
	} 
	else {
  		for (int i = 0; i < img.rows; ++i) {
		    imgin.insert(imgin.end(), img.ptr<float>(i), img.ptr<float>(i)+img.cols);
 	 	}
	}
	std::cout <<"Vector loaded with image" << std::endl; 
	std::vector<float> imgout(size[0]*ratio[0]*size[1]*ratio[1]*size[2], 0); 
	interp.MSinterp(imgin, imgout, ratio[0], ratio[1]); 
	std::cout <<"Superresolution completed" << std::endl; 
	cv::Mat img2 = cv::Mat(imgout).reshape(0, size[0]*ratio[0]);
	img2.convertTo(img2, CV_8U); 
	cv::imshow("testMat",img2);
	cv::waitKey(0);
}

