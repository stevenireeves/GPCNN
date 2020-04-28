#ifndef GP_KERNELS_H
#define GP_KERNELS_H
#include <cmath>
#include <array>

float matern3(std::array<float, 2> x, std::array<float, 2> y, float rho, float dx[]){
	float arg = 0;
	float norm = 0;
	float x1[2] = {x[0]*dx[0], x[1]*dx[1]}; 
	float y1[2] = {y[0]*dx[0], y[1]*dx[1]}; 
	for(int i = 0; i < x.size(); ++i)
	       	norm += (x1[i] - y1[i])*(x1[i] - y1[i]);
	norm = std::sqrt(norm);
	arg = std::sqrt(3)*(norm/rho);
	return (1 + arg)*std::exp(-arg);
}

float sqrexp(std::array<float, 2> x, std::array<float, 2> y, float l, float dx[]){ 
	float x1[2] = {x[0]*dx[0], x[1]*dx[1]}; 
    float y1[2] = {y[0]*dx[0], y[1]*dx[1]}; 
    float norm = 0.0; 
    for(int i = 0; i < 2; ++i)
        norm += (x1[i] - y1[i])*(x1[i] - y1[i]);
    float arg = -0.5*norm/(l*l); 
    return std::exp(arg); 
}
#endif 
