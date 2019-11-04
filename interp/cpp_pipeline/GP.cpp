#include "GP.h" 

float GP::dot(const std::array<float, 9> vec1, const std::array<float, 9> vec2){
	float result = 0.f; 
	for( int i = 0; i < 9; i++) result += vec1[i]*vec2[i]; 
	return result; 
}

std::array<float, 9> GP::load(const std::array<std::vector<std::vector<const float> >, 3> img_in,
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


std::array<float, 9> GP::get_beta(std::array<float, 9> leftbot, std::array<float, 9> bot, std::array<float, 9> rightbot,
					std::array<float, 9> left   , std::array<float, 9> cen, std::array<float, 9> right    ,
					std::array<float, 9> lefttop, std::array<float, 9> top, std::array<float, 9> righttop )
{
	// beta = f^T K^(-1) f = sum 1/lam *(V^T*f)^2 
	std::array<float, 9> beta; 
 	beta[0] = 0.f; 
	for(int i =0; i < 9; i++){
		float prod = GP::dot(vectors[i], leftbot); 
	       	beta[0] += 1.f/(eigen[i])*(prod*prod); 
	}
 	beta[1] = 0.f; 
	for(int i =0; i < 9; i++){
		float prod = GP::dot(vectors[i], bot); 
	       	beta[1] += 1.f/(eigen[i])*(prod*prod); 
	}
 	beta[2] = 0.f; 
	for(int i =0; i < 9; i++){
		float prod = GP::dot(vectors[i], rightbot); 
	       	beta[2] += 1.f/(eigen[i])*(prod*prod); 
	}
 	beta[3] = 0.f; 
	for(int i =0; i < 9; i++){
		float prod = GP::dot(vectors[i], left); 
	       	beta[3] += 1.f/(eigen[i])*(prod*prod); 
	}
 	beta[4] = 0.f; 
	for(int i =0; i < 9; i++){
		float prod = GP::dot(vectors[i], cen); 
	       	beta[4] += 1.f/(eigen[i])*(prod*prod); 
	}
 	beta[5] = 0.f; 
	for(int i =0; i < 9; i++){
		float prod = GP::dot(vectors[i], right); 
	       	beta[5] += 1.f/(eigen[i])*(prod*prod); 
	}
 	beta[6] = 0.f; 
	for(int i =0; i < 9; i++){
		float prod = GP::dot(vectors[i], lefttop); 
	       	beta[6] += 1.f/(eigen[i])*(prod*prod); 
	}
 	beta[7] = 0.f; 
	for(int i =0; i < 9; i++){
		float prod = GP::dot(vectors[i], top); 
	       	beta[7] += 1.f/(eigen[i])*(prod*prod); 
	}
 	beta[8] = 0.f; 
	for(int i =0; i < 9; i++){
		float prod = GP::dot(vectors[i], righttop); 
	       	beta[8] += 1.f/(eigen[i])*(prod*prod); 
	}
	return beta; 
}

std::array<float, 9> GP::getMSweights(const std::array< float, 9> &beta, const int ksid){
	std::array<float, 9> w8ts; 
	float summ; 
	for (int i = 0; i < 9; i++){
		w8ts[i] = gammas[ksid][i]/(std::pow(beta[i] + 1e-32, 2));	
	}
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
	for(int i = 0; i < 9; i++) summ += wsm[i]*gp[i]; 
	return summ; 
}

void GP::MSinterp(const std::array<std::vector<std::vector<const float> >, 3> img_in, 
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
					int jj = j + idy; 
					for(int idx = 0; idx < rx; idx++){
						int ii = i + idx; 
						int idk = idx + idy*rx; 
						auto msweights = GP::getMSweights(beta, idk); 
						img_out[k][jj][ii] = GP::combine(leftbot, bot, rightbot ,
									    	       left   , cen, right    ,
								     	     	       lefttop, top, righttop , 
									     	       weight[idk], msweights); 	     
					}
				}
			}
		}

		/*---------------- Borders ------------------------- */
	       	//================ bottom =========================
		int j = 0; 
		for(int i = 0; i < img_in[0][0].size(); i++){
			cen = GP::load(img_in, k, j, i); 
			for(int idy = 0; idy < ry; idy++){
				for(int idx = 0; idx < rx; idx++){
					int ii = idx + i; 
					int idk = idx + idy*rx;
					img_out[k][idy][ii] = GP::dot(weight[idk][4], cen); 
				}
			}
		}
		//================= top =======================
		j = img_in[0].size()-1; 	
		for(int i = 0; i < img_in[0][0].size(); i++){
			cen = GP::load(img_in, k, j, i); 
			for(int idy = 0; idy < ry; idy++){
				for(int idx = 0; idx < rx; idx++){
					int ii = idx + i;
				        int jj = idy + j; 	
					int idk = idx + idy*rx;
					img_out[k][jj][ii] = GP::dot(weight[idk][4], cen); 
				}
			}
		}
		//============== left =========================
		int i = 0; 
		for(j = 0; j < img_in[0].size(); j++){
			cen = GP::load(img_in, k, j, i); 
			for(int idy = 0; idy < ry; idy++){
				for(int idx = 0; idx < rx; idx++){
					int jj = idy + j; 
					int idk = idx + idy*rx;
					img_out[k][jj][idx] = GP::dot(weight[idk][4], cen); 
				}
			}
		}
		//=============== right ======================
		i = img_in[0][0].size()-1; 
		for(j = 0; j < img_in[0].size(); j++){
			cen = GP::load(img_in, k, j, i); 
			for(int idy = 0; idy < ry; idy++){
				for(int idx = 0; idx < rx; idx++){
					int ii = idx + i; 
					int jj = idy + j; 
					int idk = idx + idy*rx;
					img_out[k][jj][ii] = GP::dot(weight[idk][4], cen); 
				}
			}
		}
	}
}

