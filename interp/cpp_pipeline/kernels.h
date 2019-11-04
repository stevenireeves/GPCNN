#ifndef GP_KERNELS_H
#define GP_KERNELS_H
#include <cmath>
#include <array>

float matern3(std::array<float, 2> x, std::array<float, 2> y, float rho){
	float arg = 0;
	float norm = 0; 
	for(int i = 0; i < x.size(); ++i)
	       	norm += (x[i] - y[i])*(x[i] - y[i]);
	norm = std::sqrt(norm);
	arg = std::sqrt(3)*(norm/rho);
	return (1 + arg)*std::exp(-arg);
}
#endif 
