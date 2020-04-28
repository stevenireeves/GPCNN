#ifndef INTERP_H
#define INTERP_H
#include <array>
#include <vector>
#include <cmath> 
#include "weights.h"
 
class GP
{
public:


#ifdef __CUDAACC__
    float *weight; 
#else 
	/* Member data */ 
	std::vector<std::array<float, 9> > weight;
#endif

	int insize[3];
 
	/* Constructor if model weights are given by external function */ 
	GP(std::vector<std::array<float, 9> > wts, const int size[3]){
		weight = wts;
		insize[0] = size[0], insize[1] = size[1], insize[2] = size[2]; 
	}

    /* If weights object is used */ 
	GP(const weights wgts, const int size[3]){
		weight = wgts.ks;
		insize[0] = size[0], insize[1] = size[1], insize[2] = size[2]; 
	}

	/* Member functions */ 
#ifdef __CUDAACC__
    template <class T>
    static	inline
	__device__ float dot(const T* vec1, const T* vec2){
        float result = 0; 
        #pragma unroll
        for(int i = 0; i< 9; i++) result += vec1[i]*vec2[i]; 
		return result; 
	}
#else
    static
	inline
	float dot(const std::array<float, 9> &vec1, const std::array<float, 9> &vec2){
        float result = 0; 
        #pragma unroll
        for(int i = 0; i< 9; i++) result += vec1[i]*vec2[i]; 
		return result; 
	}
#endif 
	inline
	std::array<float, 9> load(const std::vector<float> &img_in, 
		       	        	  const int j, const int i)
	{
		std::array<float, 9> result;
        int left = (i-1);
        int right = (i+1);
		int id0 =  (j-1)*insize[0]; 
		int id1 =  j*insize[0]; 
        int id2 =  (j+1)*insize[0];
		result[0] = img_in[id0 + left];
		result[1] = img_in[id1 + left];
		result[2] = img_in[id2 + left];
		result[3] = img_in[id0 + i]; 
		result[4] = img_in[id1 + i];
		result[5] = img_in[id2 + i]; 
		result[6] = img_in[id0 + right];
		result[7] = img_in[id1 + right]; 
		result[8] = img_in[id2 + right]; 
		return result; 	
	}
 
	inline
	std::array<float, 9> load_borders(const std::vector<float> &img_in, 
                		       		  const int j, const int i)
	{
		std::array<float, 9> result;
		int bot = (j-1 < 0) ? 0 : j-1;
		int left = (i-1 < 0) ? 0 : i-1; 
		int top = (j+1 >= insize[1]) ? insize[1]-1 : j+1; 
		int right = (i+1 >= insize[0]) ? insize[0]-1 : i+1;
		int id0 =  bot*insize[0]; 
		int id1 =  j*insize[0]; 
        int id2 =  top*insize[0];
		result[0] = img_in[id0 + left];
		result[1] = img_in[id1 + left];
		result[2] = img_in[id2 + left];
		result[3] = img_in[id0 + i]; 
		result[4] = img_in[id1 + i];
		result[5] = img_in[id2 + i]; 
		result[6] = img_in[id0 + right];
		result[7] = img_in[id1 + right]; 
		result[8] = img_in[id2 + right]; 
		return result; 	
	} 

	void single_channel_interp(const std::vector<float> img_in, 
		      std::vector<float> &img_out, const int ry, const int rx); 
};

#endif
