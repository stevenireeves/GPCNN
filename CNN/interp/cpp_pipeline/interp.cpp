#include "interp.h" 
#include "Array" 

GPinterp::GPinterp(std::vector<<std::array<float, 9>> wts, std::vector<std::array<float, 9>> gam, 
		   std::array<std::array<float, 9>, 9> vec, std::array<float, 9> eig){
	weights = wts; 
	gamma = gam;
	vectors = vec; 
	eigen = eig; 
}

float GPinterp::dot(const std::array<float, 9> vec1, const std::array<float, 9> vec2){
	float result = 0.f; 
	for( int i = 0; i < 9; i++) result += vec1[i]*vec2[i]; 
	return result; 
}

std::array<float, 9> GPinterp::load(const std::array<std::vector<std::vector<const float>>, 3> img_in,
		    const int k, const int j, const int i){
	std::array<float, 9> result; 
	int bot = (j-1 < 0) ? j+1 : j-1;
	int left = (i-1 < 0) ? i+1 : i-1; 
	int top = (j+1 > img_in[0].size()) ? j-1 : j+1; 
	int right = (i+1 > img_in[0][0].size()) ? i-1 : i+1; 
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

void GPinterp::MSinterp(const std::array<std::vector<std::vector<const float>>, 3> img_in, 
		      std::array<std::vector<std::vector<float>>, 3> img_out, const int rx, const int ry){
	
	std::array<float, 9> beta = {};
	std::array<float, 9> sten_leftbot = {}; 
	std::array<float, 9> sten_bot = {};
	std::array<float, 9> sten_rightbot = {}; 
	std::array<float, 9> sten_left = {};
	std::array<float, 9> sten_cen = {};
	std::array<float, 9> sten_right = {};
	std::array<float, 9> sten_lefttop = {};
	std::array<float, 9> sten_top = {}; 
	std::array<float, 9> sten_righttop = {};

       	for( int k = 0; k < img_in.size(); k++){
		/*------------------- Body -------------------------- */ 
		for( int j = 1; j < img_in[0].size()-1; j++){
			for( int i = 1; i < img_in[0][0].size()-1; i++){
				sten_leftbot = GPinterp::load(img_in, k, j-1, i-1);
				sten_bot = GPinterp::load(img_in, k, j-1, i);
				sten_rightbot = GPinterp::load(img_in, k, j-1,i+1);
				sten_left = GPinterp::load(img_in, k, j, i-1);
				sten_cen = GPinterp::load(img_in, k, j, i);
				sten_right = GPinterp::load(img_in, k, j, i+1);
				sten_lefttop = GPinterp::load(img_in, k, j+1, i-1);
				sten_top = GPinterp::load(img_in, k, j+1, i);
				sten_righttop = GPinterp::load(img_in, k, j+1, i);
			}
		}
		int j = 0; 
		int i = 0; 
		sten_cen = GPinterp::load(img_in, k, j, i); 
		
	}
}

