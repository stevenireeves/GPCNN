#ifndef INTERP_H
#define INTERP_H
#include <array>
#include <vector>
#include <cmath> 
#include "weights.h"
 
class GP
{
public:

#ifdef USE_GPU
    float *weight; 
#else 
	/* Member data */ 
	std::vector<std::array<float, 9> > weight;
#endif

	int insize[3];
 
	/* Constructor if model weights are given by external function */ 
	GP(std::vector<std::array<float, 9> > wts, const int size[3]){
#if USE_GPU
#ifdef __CUDACC__
        cudaMalloc(&weight, 81*sizeof(float));
        cudaMemcpy(weight, wts.data(), 81*sizeof(float), cudaMemcpyHostToDevice)
#else
        hipMalloc(&weight, 81*sizeof(float));
        hipMemcpy(weight, wts.data(), 81*sizeof(float), hipMemcpyHostToDevice)
#endif 
#else
		weight = wts;
#endif
		insize[0] = size[0], insize[1] = size[1], insize[2] = size[2]; 
	}

    /* If weights object is used */ 
	GP(const weights wgts, const int size[3]){
#if USE_GPU
#ifdef __CUDACC__
        cudaMalloc(&weight, 81*sizeof(float));
        cudaMemcpy(weight, wts.ks, 81*sizeof(float), cudaMemcpyHostToDevice)
#else
        hipMalloc(&weight, 81*sizeof(float));
        hipMemcpy(weight, wts.ks, 81*sizeof(float), hipMemcpyHostToDevice)
#endif
#else
        weight = wgts.ks;
#endif
		insize[0] = size[0], insize[1] = size[1], insize[2] = size[2]; 
	}

#if USE_GPU
    ~GP(){
#ifdef __CUDACC__
        cudaFree(weights);
#else
        hipFree(weights);
#endif 
    }
#else
    ~GP(){}
#endif 

	/* Member functions */
#if USE_GPU
    template <typename T>
    inline  __device__
    std::array<T, 9> load(const T* img_in,
                          const int j, const int i)
#else
    template <typename T, typename A>
	inline
	std::array<T, 9> load(const std::vector<T, A> &img_in, 
		       	          const int j, const int i)
#endif
	{
		std::array<T, 9> result;
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

#if USE_GPU
    template <typename T>
    inline  __device__
    std::array<T, 9> load_borders(const T* img_in,
                          const int j, const int i)
#else 
    template <typename T, typename A>
	inline
	std::array<T, 9> load_borders(const std::vector<T, A> &img_in, 
                		       		  const int j, const int i)
#endif 
	{
		std::array<T, 9> result;
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


#ifdef USE_GPU
    template <class T>
    static	inline
	__device__ float dot(const float vec1[9], const T vec2[9]){
        float result = vec1[0]*vec2[0] + vec1[1]*vec2[1]
                     + vec1[2]*vec2[2] + vec1[3]*vec2[3]
                     + vec1[4]*vec2[4] + vec1[5]*vec2[5]
                     + vec1[6]*vec2[6] + vec1[7]*vec2[7]
                     + vec1[8]*vec2[8];
		return result; 
	}
#else
    template <typename T>
	static inline
	float dot(const std::array<float, 9> &vec1, const std::array<T, 9> &vec2){
        float result = vec1[0]*vec2[0] + vec1[1]*vec2[1]
                     + vec1[2]*vec2[2] + vec1[3]*vec2[3]
                     + vec1[4]*vec2[4] + vec1[5]*vec2[5]
                     + vec1[6]*vec2[6] + vec1[7]*vec2[7]
                     + vec1[8]*vec2[8];
		return result; 
	}
#endif 
#if USE_GPU

template<class T>
__global__ void single_channel_interp(const T* img_in, float* img_out, 
                                          const int ry, const int rx)
{
    const int i = threadIdx.x + blockIdx.x*blockDim.x;
    const int j = threadIdx.y + blockIdx.y*blockDim.y; 
    auto cen = GP::load(img_in, j, i);

    for(int idy = 0; idy < ry; idy++){
        int jj = j*ry + idy;
        int ind =jj*outsize[1];	
	    for(int idx = 0; idx < rx; idx++){
            int ii = i*rx + idx; 
        	int idk = idx*ry + idy;
            img_out[ind + ii] = GP::dot(weight[idk], cen); 
        }
    }
    /*---------------- Borders ------------------------- */
    //================ bottom =========================
    if(j == 0){ 
        auto cen = GP::load_borders(img_in, j, i); 
        for(int idy = 0; idy < ry; idy++){
            int ind = idy*outsize[1];
            for(int idx = 0; idx < rx; idx++){
                int ii = idx + i*rx; 
                int idk = idx*ry + idy;
                img_out[ind + ii] = GP::dot(weight[idk], cen); 
            }
        }
    }
	//================= top =======================
	else if(j == (insize[1]-1)){
		auto cen = GP::load_borders(img_in, j, i); 
		for(int idy = 0; idy < ry; idy++){
			int jj = idy + j*ry; 	
			int ind = jj*outsize[1];
			for(int idx = 0; idx < rx; idx++){
				int ii = idx + i*rx;
				int idk = idx*ry + idy;
				img_out[ind + ii] = GP::dot(weight[idk], cen); 
			}
		}
	}
	//============== left =========================
	if(i == 0){ 
		auto cen = GP::load_borders(img_in, j, i); 
		for(int idy = 0; idy < ry; idy++){
			int jj = idy + j*ry; 
			int ind =jj*outsize[1];
			for(int idx = 0; idx < rx; idx++){
				int idk = idx*ry + idy;
				img_out[ind + idx] = GP::dot(weight[idk], cen); 
			}
		}
	}
	//=============== right ======================
	else if(i == (insize[0]-1)){ 
		auto cen = GP::load_borders(img_in, j, i); 
		for(int idy = 0; idy < ry; idy++){
			int jj = idy + j*ry; 
			int ind = jj*outsize[1];
			for(int idx = 0; idx < rx; idx++){
				int ii = idx + i*rx; 
				int idk = idx*ry + idy;
				img_out[ind + ii] = GP::dot(weight[idk], cen); 
			}
		}
	}
}

#else
template<typename T, typename A>
void single_channel_interp(const std::vector<T, A> img_in,
			  std::vector<float> &img_out, const int ry, const int rx)
{
	const int outsize[2] = {insize[1]*ry, insize[0]*rx}; 
	/*------------------- Body -------------------------- */
#pragma omp parallel for	
	for( int j = 1; j < insize[1]-1; j++){
		for(int i = 1; i < insize[0]-1; i++){
            auto cen = GP::load(img_in, j, i);
		    for(int idy = 0; idy < ry; idy++){
		        int jj = j*ry + idy;
		        int ind =jj*outsize[1];	
		        for(int idx = 0; idx < rx; idx++){
			        int ii = i*rx + idx; 
			        int idk = idx*ry + idy;
			        img_out[ind + ii] = GP::dot(weight[idk], cen); 
		            }
	            }
	      }
	}
	/*---------------- Borders ------------------------- */
	//================ bottom =========================
	int j = 0; 
#pragma omp parallel for
	for(int i = 0; i < insize[0]; i++){
		auto cen = GP::load_borders(img_in, j, i); 
		for(int idy = 0; idy < ry; idy++){
			int ind = idy*outsize[1];
			for(int idx = 0; idx < rx; idx++){
				int ii = idx + i*rx; 
				int idk = idx*ry + idy;
				img_out[ind + ii] = GP::dot(weight[idk], cen); 
			}
		}
	}
	//================= top =======================
	j = (insize[1]-1); 	
#pragma omp parallel for
	for(int i = 0; i < insize[0]; i++){
		auto cen = GP::load_borders(img_in, j, i); 
		for(int idy = 0; idy < ry; idy++){
			int jj = idy + j*ry; 	
			int ind = jj*outsize[1];
			for(int idx = 0; idx < rx; idx++){
				int ii = idx + i*rx;
				int idk = idx*ry + idy;
				img_out[ind + ii] = GP::dot(weight[idk], cen); 
			}
		}
	}
	//============== left =========================
	int i = 0; 
#pragma omp parallel for
	for(j = 0; j < insize[1]; j++){
		auto cen = GP::load_borders(img_in, j, i); 
		for(int idy = 0; idy < ry; idy++){
			int jj = idy + j*ry; 
			int ind =jj*outsize[1];
			for(int idx = 0; idx < rx; idx++){
				int idk = idx*ry + idy;
				img_out[ind + idx] = GP::dot(weight[idk], cen); 
			}
		}
	}
	//=============== right ======================
	i = (insize[0]-1); 
#pragma omp parallel for
	for(j = 0; j < insize[1]; j++){
		auto cen = GP::load_borders(img_in, j, i); 
		for(int idy = 0; idy < ry; idy++){
			int jj = idy + j*ry; 
			int ind = jj*outsize[1];
			for(int idx = 0; idx < rx; idx++){
				int ii = idx + i*rx; 
				int idk = idx*ry + idy;
				img_out[ind + ii] = GP::dot(weight[idk], cen); 
			}
		}
	}
}


#endif
};
#endif
