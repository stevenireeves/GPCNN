#include "GPinterp.h" 

float GPinterp::dot(const std::array<float, 9> vec1, const std::array<float, 9> vec2){
	float result = 0.f; 
	for( int i = 0; i < 9; i++) result += vec1[i]*vec2[i]; 
	return result; 
}

std::array<float, 9> GPinterp::load(const std::array<std::vector<std::vector<const float> >, 3> img_in,
		    		    const int k, const int j, const int i){
	std::array<float, 9> result; 
	int bot = (j-1 < 0) ? 0 : j-1;
	int left = (i-1 < 0) ? 0 : i-1; 
	int top = (j+1 > img_in[0].size()) ? img_in[0].size()-1 : j+1; 
	int right = (i+1 > img_in[0][0].size()) ? img_in[0][0].size()-1 : i+1; 
	result[0] = img_in[k][bot][left]; 
	result[1] = img_in[k][bot][i]; 
	result[2] = img_in[k][bot][right];
        result[3] = img_in[k][j][left]; 
	result[4] = img_in[k][j][i]; 
	result[5] = img_in[k][j][right]; 
	result[6] = img_in[k][top][left];
	result[7] = img_in[k][top][i]; 
	result[8] = img_in[k][top][right]; 
	return result; 	
}


std::array<float, 9> GPinterp::get_beta(std::array<float, 9> leftbot, std::array<float, 9> bot, std::array<float, 9> rightbot,
					std::array<float, 9> left   , std::array<float, 9> cen, std::array<float, 9> right    ,
					std::array<float, 9> lefttop, std::array<float, 9> top, std::array<float, 9> righttop )
{
	// beta = f^T K^(-1) f = sum 1/lam *(V^T*f)^2 
	std::array<float, 9> beta; 
 	beta[0] = 0.f; 
	for(int i =0; i < 9; i++){
		float prod = GPinterp::dot(vectors[i], leftbot); 
	       	beta[0] += 1.f/(eigen[i])*(prod*prod); 
	}
 	beta[1] = 0.f; 
	for(int i =0; i < 9; i++){
		float prod = GPinterp::dot(vectors[i], bot); 
	       	beta[1] += 1.f/(eigen[i])*(prod*prod); 
	}
 	beta[2] = 0.f; 
	for(int i =0; i < 9; i++){
		float prod = GPinterp::dot(vectors[i], rightbot); 
	       	beta[2] += 1.f/(eigen[i])*(prod*prod); 
	}
 	beta[3] = 0.f; 
	for(int i =0; i < 9; i++){
		float prod = GPinterp::dot(vectors[i], left); 
	       	beta[3] += 1.f/(eigen[i])*(prod*prod); 
	}
 	beta[4] = 0.f; 
	for(int i =0; i < 9; i++){
		float prod = GPinterp::dot(vectors[i], cen); 
	       	beta[4] += 1.f/(eigen[i])*(prod*prod); 
	}
 	beta[5] = 0.f; 
	for(int i =0; i < 9; i++){
		float prod = GPinterp::dot(vectors[i], right); 
	       	beta[5] += 1.f/(eigen[i])*(prod*prod); 
	}
 	beta[6] = 0.f; 
	for(int i =0; i < 9; i++){
		float prod = GPinterp::dot(vectors[i], lefttop); 
	       	beta[6] += 1.f/(eigen[i])*(prod*prod); 
	}
 	beta[7] = 0.f; 
	for(int i =0; i < 9; i++){
		float prod = GPinterp::dot(vectors[i], top); 
	       	beta[7] += 1.f/(eigen[i])*(prod*prod); 
	}
 	beta[8] = 0.f; 
	for(int i =0; i < 9; i++){
		float prod = GPinterp::dot(vectors[i], righttop); 
	       	beta[8] += 1.f/(eigen[i])*(prod*prod); 
	}
	return beta; 
}

std::array<float, 9> GPinterp::getMSweights(const std::array< float, 9> &beta, const int ksid){
	std::array<float, 9> w8ts; 
	float summ; 
	for (int i = 0; i < 9; i++){
		w8ts[i] = gammas[ksid][i]/(std::pow(beta[i] + 1e-32, 2));	
	}
	return w8ts; 
}

float GPinterp::combine(std::array<float, 9> leftbot, std::array<float, 9> bot, std::array<float, 9> rightbot,
			std::array<float, 9> left   , std::array<float, 9> cen, std::array<float, 9> right   ,
			std::array<float, 9> lefttop, std::array<float, 9> top, std::array<float, 9> righttop, 
			std::array<std::array<float, 9>, 9> weights, std::array<float, 9> wsm)
{
	std::array<float, 9> gp; 
	gp[0] = GPinterp::dot(weights[0], leftbot); 
	gp[1] = GPinterp::dot(weights[1], bot); 
	gp[2] = GPinterp::dot(weights[2], rightbot); 
	gp[3] = GPinterp::dot(weights[3], left); 
	gp[4] = GPinterp::dot(weights[4], cen); 
	gp[5] = GPinterp::dot(weights[5], right);
	gp[6] = GPinterp::dot(weights[6], lefttop); 
	gp[7] = GPinterp::dot(weights[7], top); 
	gp[8] = GPinterp::dot(weights[8], righttop); 
	float summ = 0; 
	for(int i = 0; i < 9; i++) summ += wsm[i]*gp[i]; 
	return summ; 
}

void GPinterp::MSinterp(const std::array<std::vector<std::vector<const float> >, 3> img_in, 
                        std::array<std::vector<std::vector<float> >, 3> img_out, const int ry, const int rx){
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

       	for( int k = 0; k < img_in.size(); k++){
		/*------------------- Body -------------------------- */ 
		for( int j = 1; j < img_in[0].size()-1; j++){
			for( int i = 1; i < img_in[0][0].size()-1; i++){

				leftbot = GPinterp::load(img_in, k, j-1, i-1);
				bot = GPinterp::load(img_in, k, j-1, i);
				rightbot = GPinterp::load(img_in, k, j-1,i+1);
				left = GPinterp::load(img_in, k, j, i-1);
				cen = GPinterp::load(img_in, k, j, i);
				right = GPinterp::load(img_in, k, j, i+1);
				lefttop = GPinterp::load(img_in, k, j+1, i-1);
				top = GPinterp::load(img_in, k, j+1, i);
				righttop = GPinterp::load(img_in, k, j+1, i);

				beta = GPinterp::get_beta(leftbot, bot, rightbot, 
							  left   , cen, right   ,
							  lefttop, top, righttop); 

				for(int idy = 0; idy < ry; idy++){
					int jj = j + idy; 
					for(int idx = 0; idx < rx; idx++){
						int ii = i + idx; 
						int idk = idx + idy*rx; 
						auto msweights = GPinterp::getMSweights(beta, idk); 
						img_out[k][jj][ii] = GPinterp::combine(leftbot, bot, rightbot ,
									    	       left   , cen, right    ,
								     	     	       lefttop, top, righttop , 
									     	       weights[idk], msweights); 	     
					}
				}
			}
		}

		/*---------------- Borders ------------------------- */
	       	//================ bottom =========================
		int j = 0; 
		for(int i = 0; i < img_in[0][0].size(); i++){
			cen = GPinterp::load(img_in, k, j, i); 
			for(int idy = 0; idy < ry; idy++){
				for(int idx = 0; idx < rx; idx++){
					int ii = idx + i; 
					int idk = idx + idy*rx;
					img_out[k][idy][ii] = GPinterp::dot(weights[idk][4], cen); 
				}
			}
		}
		//================= top =======================
		j = img_in[0].size()-1; 	
		for(int i = 0; i < img_in[0][0].size(); i++){
			cen = GPinterp::load(img_in, k, j, i); 
			for(int idy = 0; idy < ry; idy++){
				for(int idx = 0; idx < rx; idx++){
					int ii = idx + i;
				        int jj = idy + j; 	
					int idk = idx + idy*rx;
					img_out[k][jj][ii] = GPinterp::dot(weights[idk][4], cen); 
				}
			}
		}
		//============== left =========================
		int i = 0; 
		for(j = 0; j < img_in[0].size(); j++){
			cen = GPinterp::load(img_in, k, j, i); 
			for(int idy = 0; idy < ry; idy++){
				for(int idx = 0; idx < rx; idx++){
					int jj = idy + j; 
					int idk = idx + idy*rx;
					img_out[k][jj][idx] = GPinterp::dot(weights[idk][4], cen); 
				}
			}
		}
		//=============== right ======================
		i = img_in[0][0].size()-1; 
		for(j = 0; j < img_in[0].size(); j++){
			cen = GPinterp::load(img_in, k, j, i); 
			for(int idy = 0; idy < ry; idy++){
				for(int idx = 0; idx < rx; idx++){
					int ii = idx + i; 
					int jj = idy + j; 
					int idk = idx + idy*rx;
					img_out[k][jj][ii] = GPinterp::dot(weights[idk][4], cen); 
				}
			}
		}
	}
}

