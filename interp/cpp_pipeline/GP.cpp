#include <iostream>
#include <omp.h>
#include "GP.h" 

template<typename T, typename A>
void GP::single_channel_interp(const std::vector<T, A> img_in,
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



