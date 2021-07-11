/* This is a c-style wrapper for for the GP interpolation to be used from python. */ 
#include "GP.h"
#include "weights.h"
#include <iostream>

#ifdef USE_GPU
template<typename T>
struct image {
	T* b; 
	T* g; 
	T* r; 
}; 
#endif

void driver(unsigned char *img_in, float *img_out, const int upsample_ratio[], const int in_size[]){
    const float del[2] = {1.f/float(in_size[0]), 1.f/float(in_size[1])};
    weights wgts(upsample_ratio, del); 
    const int size[3] = {in_size[0], in_size[1], 1}; 
    GP interp(wgts, size); 
#ifdef USE_GPU
    unsigned char *img1;
    float *img2; 
    size_t img1_size = size[0]*size[1]*sizeof(unsigned char); 
    size_t img2_size = size[0]*size[1]*upsample_ratio[0]*upsample_ratio[1]*sizeof(float);
    dim3 dimBlock(32, 32); 
    dim3 dimGrid(size[0]/dimBlock.x, size[1]/dimBlock.y); 
    #ifdef __CUDACC__
        cudaMalloc(&img1, img1_size);
        cudaMalloc(&img2, img2_size);
        cudaMemcpy(img1, img_in, img1_size, cudaMemcpyHostToDevice);
        single_channel_interp<<<dimGrid, dimBlock>>>(img1, img2, interp, upsample_ratio[0], upsample_ratio[1]);
        cudaMemcpy(img_out, img2, img2_size, cudaMemcpyDeviceToHost);
        cudaFree(img1);
        cudaFree(img2);
    #else
        hipMalloc(&img1, img1_size);
        hipMalloc(&img2, img2_size);
        hipMemcpy(img1, img_in, img1_size, hipMemcpyHostToDevice);
        single_channel_interp<<<dimGrid, dimBlock>>>(img1, img2, interp, upsample_ratio[0], upsample_ratio[1]);
        hipMemcpy(img_out, img2, img2_size, hipMemcpyDeviceToHost);
        hipFree(img1);
        hipFree(img2);
    #endif
#else
    std::vector<unsigned char> img1(img_in, img_in + in_size[0]*in_size[1]); 
    std::vector<float> img2(size[0]*upsample_ratio[0]*upsample_ratio[1]*size[1], 0.f);
    interp.single_channel_interp(img1, img2, upsample_ratio[0], upsample_ratio[1]);
    std::copy(img2.begin(), img2.end(), img_out); 
#endif
}

void driver_color(unsigned char *bin, unsigned char *gin, unsigned char *rin, 
                  float *bout, float *gout, float *rout,
                  const int upsample_ratio[], const int in_size[]){
    const float del[2] = {1.f/float(in_size[0]), 1.f/float(in_size[1])};
    weights wgts(upsample_ratio, del); 
    const int size[3] = {in_size[0], in_size[1], in_size[2]}; 
    GP interp(wgts, size); 
#ifndef USE_GPU
    std::vector<unsigned char> b1(bin, bin + in_size[0]*in_size[1]);
    std::vector<unsigned char> g1(gin, gin + in_size[0]*in_size[1]); 
    std::vector<unsigned char> r1(rin, rin + in_size[0]*in_size[1]);  
    std::vector<float> b2(size[0]*upsample_ratio[0]*upsample_ratio[1]*size[1], 0.f);
    std::vector<float> r2(size[0]*upsample_ratio[0]*upsample_ratio[1]*size[1], 0.f);
    std::vector<float> g2(size[0]*upsample_ratio[0]*upsample_ratio[1]*size[1], 0.f);
    interp.single_channel_interp(b1, b2, upsample_ratio[0], upsample_ratio[1]);
    interp.single_channel_interp(g1, g2, upsample_ratio[0], upsample_ratio[1]);
    interp.single_channel_interp(r1, r2, upsample_ratio[0], upsample_ratio[1]);
    std::copy(b2.begin(), b2.end(), bout); 
    std::copy(g2.begin(), g2.end(), gout); 
    std::copy(r2.begin(), r2.end(), rout); 
#else
    image<unsigned char> img1;
    image<float> img2; 
    size_t img1_size = size[0]*size[1]*sizeof(unsigned char); 
    size_t img2_size = size[0]*size[1]*upsample_ratio[0]*upsample_ratio[1]*sizeof(float);
    dim3 dimBlock(32, 32); 
    dim3 dimGrid(size[0]/dimBlock.x, size[1]/dimBlock.y); 
    #ifdef __CUDACC__
        cudaMalloc(&img1.b, img1_size);
        cudaMalloc(&img1.g, img1_size);
        cudaMalloc(&img1.r, img1_size);
        cudaMalloc(&img2.b, img2_size);
        cudaMalloc(&img2.g, img2_size);
        cudaMalloc(&img2.r, img2_size);
        cudaMemcpy(img1.b, bin, img1_size, cudaMemcpyHostToDevice);
        cudaMemcpy(img1.g, gin, img1_size, cudaMemcpyHostToDevice);
        cudaMemcpy(img1.r, rin, img1_size, cudaMemcpyHostToDevice);
        single_channel_interp<<<dimGrid, dimBlock>>>(img1.b, img2.b, interp, upsample_ratio[0], upsample_ratio[1]);
        single_channel_interp<<<dimGrid, dimBlock>>>(img1.g, img2.g, interp, upsample_ratio[0], upsample_ratio[1]);
        single_channel_interp<<<dimGrid, dimBlock>>>(img1.r, img2.r, interp, upsample_ratio[0], upsample_ratio[1]);
        cudaMemcpy(bout, img2.b, img2_size, cudaMemcpyDeviceToHost);
        cudaMemcpy(gout, img2.g, img2_size, cudaMemcpyDeviceToHost);
        cudaMemcpy(rout, img2.r, img2_size, cudaMemcpyDeviceToHost);
        cudaFree(img1.b);
        cudaFree(img1.g);
        cudaFree(img1.r);
        cudaFree(img2.b);
        cudaFree(img2.g);
        cudaFree(img2.r);
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
#endif 
}

extern "C"
{
	void interpolate(unsigned char *img_in, float *img_out, 
                     const int *upsample_ratio, const int *in_size){
			driver(img_in, img_out, upsample_ratio, in_size); 
	}

    void interpolate_color(unsigned char *b_in, unsigned char *g_in, unsigned char* r_in, 
                           float *b_out, float *g_out, float *r_out, 
                           const int *upsample_ratio, const int *in_size){
			driver_color(b_in, g_in, r_in,
                         b_out, g_out, r_out, 
                         upsample_ratio, in_size);
	}
}


