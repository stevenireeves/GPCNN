#ifndef GP_KERNELS_H
#define GP_KERNELS_H
#include <cmath>
#include <vector>

float matern3(std::vector<float> x, std::vector<float> y, float rho){
	float arg = 0;
	float norm = 0; 
	for(int i = 0; i < x.size(); ++i)
	       	norm += (x[i] - y[i])*(x[i] - y[i]);
	norm = std::sqrt(norm);
	arg = std::sqrt(3)*(norm/rho);
	return (1 + arg)*std::exp(-arg)
}
#endif 
