#include <stdio.h>
#include <chrono>

#include <opencv2/opencv.hpp>

#include "GP.h"
#include "weights.h"

int main(int argc, char* argv[]){
	cv::Mat img = cv::imread(argv[1], cv::IMREAD_COLOR);
    auto st1 = std::chrono::high_resolution_clock::now(); 
    std::vector<cv::Mat> bgr_in;
    std::vector<cv::Mat> bgr_out;
//	int chan = img.channels()==3? CV_32FC3 : CV_32F;
//	img.convertTo(img, chan);
//	img /= 255;
    auto st6 = std::chrono::high_resolution_clock::now(); 
    cv::split(img, bgr_in); 
	const float del[2] = {1.f/float(img.cols), 1.f/float(img.rows)}; 
	const int ratio[2] = {2, 2}; 
	const int size[3] = {img.cols, img.rows, img.channels()};

    auto s6 = std::chrono::high_resolution_clock::now(); 
	weights wgts(ratio, del);
    auto s8 = std::chrono::high_resolution_clock::now(); 
	GP interp(wgts, size);
    auto s7 = std::chrono::high_resolution_clock::now(); 
    auto d5 = std::chrono::duration_cast<std::chrono::milliseconds>(s7 - s6); 
    auto d6 = std::chrono::duration_cast<std::chrono::milliseconds>(s8 - s6); 
    std::cout<< "Weight Calculation time " << d6.count() << " ms" << std::endl; 
    std::cout<< "GP Set Up time " << d5.count() << " ms" << std::endl; 
   
    
    auto st7 = std::chrono::high_resolution_clock::now(); 
    auto d4 = std::chrono::duration_cast<std::chrono::milliseconds>(st7 - st6); 
    std::cout<< "Set Up time " << d4.count() << " ms" << std::endl; 

#ifdef USE_GPU
    for(int i = 0; i < 3; ++i) bgr_out.push_back(cv::Mat::zeros
                                                (cv::Size(size[0]*ratio[0], size[1]*ratio[1]),
                                                CV_32FC1));
    size_t img1_size = img.cols*img.rows*sizeof(unsigned char); 
    size_t img2_size = img.cols*img.rows*ratio[0]*ratio[1]*sizeof(float);
    dim3 dimBlock(32, 32); 
    dim3 dimGrid(img.cols/dimBlock.x, img.rows/dimBlock.y); 
    unsigned char* b, *g, *r; 
    float* b2, *g2, *r2; 
    #ifdef __CUDACC__
        float ms = 0; 
        cudaEvent_t start, stop, memstart, memstop; 
        cudaEventCreate(&start); 
        cudaEventCreate(&stop); 
        cudaEventCreate(&memstart); 
        cudaEventCreate(&memstop); 
        cudaEventRecord(memstart); 
        cudaMalloc(&b, img1_size);
        cudaMalloc(&g, img1_size);
        cudaMalloc(&r, img1_size);
        cudaMalloc(&b2, img2_size);
        cudaMalloc(&g2, img2_size);
        cudaMalloc(&r2, img2_size);
        cudaMemcpy(b, bgr_in[0].data, img1_size, cudaMemcpyHostToDevice);
        cudaMemcpy(g, bgr_in[1].data, img1_size, cudaMemcpyHostToDevice);
        cudaMemcpy(r, bgr_in[2].data, img1_size, cudaMemcpyHostToDevice);
        cudaEventRecord(memstop); 
        cudaEventSynchronize(memstop); 
        cudaEventElapsedTime(&ms, memstart, memstop); 
        std::cout<<"Mem Malloc and CP H2D time " << ms << "ms"<<std::endl; 
        cudaEventRecord(start);
        single_channel_interp<<<dimGrid, dimBlock>>>(b, b2, interp, ratio[0], ratio[1]);
        single_channel_interp<<<dimGrid, dimBlock>>>(g, g2, interp, ratio[0], ratio[1]);
        single_channel_interp<<<dimGrid, dimBlock>>>(r, r2, interp, ratio[0], ratio[1]);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop); 
        std::cout<<"Kernels time " << ms << "ms" << std::endl;  
        cudaEventRecord(memstart);
        cudaMemcpy(bgr_out[0].data, b2, img2_size, cudaMemcpyDeviceToHost);
        cudaMemcpy(bgr_out[1].data, g2, img2_size, cudaMemcpyDeviceToHost);
        cudaMemcpy(bgr_out[2].data, r2, img2_size, cudaMemcpyDeviceToHost);
        cudaEventRecord(memstop); 
        cudaEventSynchronize(memstop); 
        cudaEventElapsedTime(&ms, memstart,memstop); 
        std::cout<<"Mem CP D2H "<< ms << "ms"<<std::endl; 
        cudaFree(b);
        cudaFree(g);
        cudaFree(r);
        cudaFree(b2);
        cudaFree(g2);
        cudaFree(r2);
    #else
        hipMalloc(&img1.b, img1_size);
        hipMalloc(&img1.g, img1_size);
        hipMalloc(&img1.r, img1_size);
        hipMalloc(&img2.b, img2_size);
        hipMalloc(&img2.g, img2_size);
        hipMalloc(&img2.r, img2_size);
        hipMemcpy(img1.b, bin, img1_size, hipMemcpyHostToDevice);
        hipMemcpy(img1.g, gin, img1_size, hipMemcpyHostToDevice);
        hipMemcpy(img1.r, rin, img1_size, hipMemcpyHostToDevice);
        single_channel_interp<<<dimGrid, dimBlock>>>(img1.b, img2.b, interp, upsample_ratio[0], upsample_ratio[1]);
        single_channel_interp<<<dimGrid, dimBlock>>>(img1.g, img2.g, interp, upsample_ratio[0], upsample_ratio[1]);
        single_channel_interp<<<dimGrid, dimBlock>>>(img1.r, img2.r, interp, upsample_ratio[0], upsample_ratio[1]);
        hipMemcpy(bout, img2.b, img2_size, hipMemcpyDeviceToHost);
        hipMemcpy(gout, img2.g, img2_size, hipMemcpyDeviceToHost);
        hipMemcpy(rout, img2.r, img2_size, hipMemcpyDeviceToHost);
        hipFree(img1.b);
        hipFree(img1.g);
        hipFree(img1.r);
        hipFree(img2.b);
        hipFree(img2.g);
        hipFree(img2.r);
    #endif
#else
    for(int i = 0; i < size[2]; i++){ 
        auto start = std::chrono::high_resolution_clock::now(); 
        cv::Mat flat = bgr_in[i].reshape(1, bgr_in[i].total());
        std::vector<unsigned char> imgin = bgr_in[i].isContinuous()? flat : flat.clone();
        std::vector<float> imgout(size[0]*ratio[0]*size[1]*ratio[1], 0);
        interp.single_channel_interp(imgin, imgout, ratio[0], ratio[1]); 
        auto stop = std::chrono::high_resolution_clock::now(); 
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start); 
	    bgr_out.push_back(cv::Mat(imgout).reshape(1, size[1]*ratio[1]));
        bgr_out[i].convertTo(bgr_out[i], CV_8U); 
        std::cout<< " Interp profile = " << duration.count() << " ms" << std::endl; 
    }
//    interp.MSinterp(imgin, imgout, ratio[0], ratio[1]); 
#endif 
    cv::Mat img2;
    std::cout<<"Merge"<<std::endl; 
    auto st3 = std::chrono::high_resolution_clock::now(); 
    cv::merge(bgr_out, img2); 
    auto st2 = std::chrono::high_resolution_clock::now(); 
    auto d3 = std::chrono::duration_cast<std::chrono::milliseconds>(st2 - st3);
    std::cout<<" Merge time = " << d3.count() << " ms" << std::endl;  
    auto d2 = std::chrono::duration_cast<std::chrono::milliseconds>(st2 - st1); 
    std::cout<<" Total time = " << d2.count() << " ms" << std::endl; 
    std::cout<<" Total time without setup = " << d2.count()-d4.count() << " ms" << std::endl; 

//	img2 *= 255; 
//	int uchan = img.channels()==3? CV_8UC3: CV_8U; 
//    img2.convertTo(img2, uchan); 
//    cv::imshow("test", img2);
//    cv::waitKey(); 
    std::cout<<"Write"<<std::endl; 
	cv::imwrite("test_scale.png", img2);  
/*	cv::imshow("testMat",img2);
	cv::waitKey(0);*/ 
    return 0;
}

