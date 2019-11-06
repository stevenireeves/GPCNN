#include <iostream>
#include "GP.h" 

float GP::dot(const std::array<float, 9> vec1, const std::array<float, 9> vec2){
	float result = 0.f; 
	for( int i = 0; i < 9; i++) result += vec1[i]*vec2[i]; 
	return result; 
}

std::array<float, 9> GP::load(const std::vector <float> img_in,
		    		    const int k, const int j, const int i){
	std::array<float, 9> result; 
	int bot = (j-1 < 0) ? 0 : j-1;
	int left = (i-1 < 0) ? 0 : i-1; 
	int top = (j+1 > insize[1]) ? insize[1]-1 : j+1; 
	int right = (i+1 > insize[0]) ? insize[0]-1 : i+1;
	result[0] = img_in[(k*insize[0] + bot)*insize[1] + left];
	result[1] = img_in[(k*insize[0] + bot)*insize[1] + i]; 
	result[2] = img_in[(k*insize[0] + bot)*insize[1] + right];
        result[3] = img_in[(k*insize[0] + j)*insize[1] + left]; 
	result[4] = img_in[(k*insize[0] + j)*insize[1] + i]; 
	result[5] = img_in[(k*insize[0] + j)*insize[1] + right]; 
	result[6] = img_in[(k*insize[0] + top)*insize[1] + left];
	result[7] = img_in[(k*insize[0] + top)*insize[1] + i]; 
	result[8] = img_in[(k*insize[0] + top)*insize[1] + right]; 
	return result; 	
}


std::array<float, 9> GP::get_beta(std::array<float, 9> leftbot, std::array<float, 9> bot, std::array<float, 9> rightbot,
					std::array<float, 9> left   , std::array<float, 9> cen, std::array<float, 9> right    ,
					std::array<float, 9> lefttop, std::array<float, 9> top, std::array<float, 9> righttop )
{
	// beta = f^T K^(-1) f = sum 1/lam *(V^T*f)^2 
	std::array<float, 9> beta = {}; 
	std::array<float, 9> vs = {}; 
	for(int i =0; i < 9; i++){
		vs = vectors[i]; 
		float prod = GP::dot(vs, leftbot); 
	       	beta[0] += 1.f/(eigen[i])*(prod*prod); 
		prod = GP::dot(vs, bot); 
	       	beta[1] += 1.f/(eigen[i])*(prod*prod); 
		prod = GP::dot(vs, rightbot); 
	       	beta[2] += 1.f/(eigen[i])*(prod*prod); 
		prod = GP::dot(vs, left); 
	       	beta[3] += 1.f/(eigen[i])*(prod*prod); 
		prod = GP::dot(vs, cen); 
	       	beta[4] += 1.f/(eigen[i])*(prod*prod); 
		prod = GP::dot(vs, right); 
	       	beta[5] += 1.f/(eigen[i])*(prod*prod); 
		prod = GP::dot(vs, lefttop); 
	       	beta[6] += 1.f/(eigen[i])*(prod*prod); 
		prod = GP::dot(vs, top); 
	       	beta[7] += 1.f/(eigen[i])*(prod*prod); 
		prod = GP::dot(vs, righttop); 
	       	beta[8] += 1.f/(eigen[i])*(prod*prod); 
	}
	return beta; 
}

std::array<float, 9> GP::getMSweights(const std::array< float, 9> &beta, const int ksid){
	std::array<float, 9> w8ts; 
	float summ = 0; 
	for (int i = 0; i < 9; i++){
		w8ts[i] = gammas[ksid][i]/(beta[i] + 1e-36);
		summ += w8ts[i]; 	
	} 
	for(int i = 0; i < 9; i++) w8ts[i] /= summ;
	return w8ts; 
}

float GP::combine(std::array<float, 9> leftbot, std::array<float, 9> bot, std::array<float, 9> rightbot,
			std::array<float, 9> left   , std::array<float, 9> cen, std::array<float, 9> right   ,
			std::array<float, 9> lefttop, std::array<float, 9> top, std::array<float, 9> righttop, 
			std::array<std::array<float, 9>, 9> wts, std::array<float, 9> wsm)
{
	std::array<float, 9> gp; 
	gp[0] = GP::dot(wts[0], leftbot); 
	gp[1] = GP::dot(wts[1], bot); 
	gp[2] = GP::dot(wts[2], rightbot); 
	gp[3] = GP::dot(wts[3], left);
	gp[4] = GP::dot(wts[4], cen); 
	gp[5] = GP::dot(wts[5], right);
	gp[6] = GP::dot(wts[6], lefttop); 
	gp[7] = GP::dot(wts[7], top); 
	gp[8] = GP::dot(wts[8], righttop); 
	float summ = 0; 
	for(int i = 0; i < 9; i++){
		summ += wsm[i]*gp[i];
	}
	return summ; 
}

void GP::MSinterp(const std::vector<float> img_in, 
                        std::vector<float> &img_out, const int ry, const int rx){
	std::array<float, 9> beta = {};
	std::array<float, 9> leftbot = {}; 
	std::array<float, 9> bot = {};
	std::array<float, 9> rightbot = {}; 
	std::array<float, 9> left = {};
	std::array<float, 9> cen = {};
	std::array<float, 9> right = {};
	std::array<float, 9> lefttop = {};
	std::array<float, 9> top = {}; 
	std::array<float, 9> righttop = {};
	const int outsize[3] = {insize[0]*rx, insize[1]*ry, insize[0]*rx}; 
       	for( int k = 0; k < insize[2]; k++){
		/*------------------- Body -------------------------- */ 
		for( int j = 1; j < insize[1]-1; j++){
			for( int i = 1; i < insize[0]-1; i++){

				leftbot = GP::load(img_in, k, j-1, i-1);
				bot = GP::load(img_in, k, j-1, i);
				rightbot = GP::load(img_in, k, j-1,i+1);
				left = GP::load(img_in, k, j, i-1);
				cen = GP::load(img_in, k, j, i);
				right = GP::load(img_in, k, j, i+1);
				lefttop = GP::load(img_in, k, j+1, i-1);
				top = GP::load(img_in, k, j+1, i);
				righttop = GP::load(img_in, k, j+1, i);

				beta = GP::get_beta(leftbot, bot, rightbot, 
					            left   , cen, right   ,
					  	    lefttop, top, righttop);
				for(int idy = 0; idy < ry; idy++){
					int jj = j*ry + idy; 
					for(int idx = 0; idx < rx; idx++){
						int ii = i*rx + idx; 
						int idk = idx + idy*rx; 
						auto msweights = GP::getMSweights(beta, idk);
						img_out[(k*outsize[2] + jj)*outsize[1] + ii] = GP::combine(leftbot, bot, rightbot ,
									    	       			   left   , cen, right    ,
								     	     	       			   lefttop, top, righttop , 
									     	       			   weight[idk], msweights);
					}
				}
			}
		}
		std::cout << "Body Done" << std::endl;
		/*---------------- Borders ------------------------- */
	       	//================ bottom =========================
/*		int j = 0; 
		for(int i = 0; i < insize[0]; i++){
			cen = GP::load(img_in, k, j, i); 
			for(int idy = 0; idy < ry; idy++){
				for(int idx = 0; idx < rx; idx++){
					int ii = idx + i*rx; 
					int idk = idx + idy*rx;
					img_out[(k*outsize[2] + idy)*outsize[1] + ii] = GP::dot(weight[idk][4], cen); 
				}
			}
		}
		std::cout<<"bottom done" << std::endl; 
		//================= top =======================
		j = (insize[1]-2); 	
		for(int i = 0; i < insize[0]; i++){
			cen = GP::load(img_in, k, j, i); 
			for(int idy = 0; idy < ry; idy++){
				for(int idx = 0; idx < rx; idx++){
					int ii = idx + i*rx;
				        int jj = idy + j*ry; 	
					int idk = idx + idy*rx;
					img_out[(k*outsize[2] + jj)*outsize[1] + ii] = GP::dot(weight[idk][4], cen); 
				}
			}
		}
		std::cout<<"top done" << std::endl; 
		//============== left =========================
		int i = 0; 
		for(j = 0; j < insize[1]; j++){
			cen = GP::load(img_in, k, j, i); 
			for(int idy = 0; idy < ry; idy++){
				for(int idx = 0; idx < rx; idx++){
					int jj = idy + j*ry; 
					int idk = idx + idy*rx;
					img_out[(k*outsize[2] + jj)*outsize[1] + idx] = GP::dot(weight[idk][4], cen); 
				}
			}
		}
		std::cout<<"left done" << std::endl; 
		//=============== right ======================
		i = (insize[0]-2); 
		for(j = 0; j < insize[1]; j++){
			cen = GP::load(img_in, k, j, i); 
			for(int idy = 0; idy < ry; idy++){
				for(int idx = 0; idx < rx; idx++){
					int ii = idx + i*rx; 
					int jj = idy + j*ry; 
					int idk = idx + idy*rx;
					img_out[(k*outsize[2] + jj)*outsize[1] + ii] = GP::dot(weight[idk][4], cen); 
				}
			}
		}*/ 
	}
}

