#ifndef INTERP_H
#define INTERP_H
#include "kernels.h"

class GPinterp
{
public:
	GPinterp(std::vector<float> wts, std::vector<float> gam);
	std::vector<float> weights;
	std::vector<float> gamma;

	void interp(const std::vector<const float> img_in, std::vector<float> img_out); 
}

#endif 
