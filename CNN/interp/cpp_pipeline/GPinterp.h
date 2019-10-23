#ifndef INTERP_H
#define INTERP_H
#include <array>

#include "kernels.h"

class GPinterp
{
public:
	/* Constructor */ 
	GPinterp(std::vector<float> wts, std::vector<float> gam);

	/* Member data */ 
	std::vector<float> weights;
	std::vector<float> gamma;


	/* Member functions */ 
	float dot(const std::array<float, 9> vec1, const std::array<float, 9> vec2); 

	void interp(const std::vector<const float> img_in, std::vector<float> img_out); 

	std::array<float, 9> load(const std::array< std::vector< std::vector< float > >, 3> img_in, 
		       		  const int k, const int j, const int i); 

	std::array<float, 9> get_beta(std::array<float, 9> lbot, std::array<float, 9> bot,
				      std::array<float, 9> rbot, std::array<float, 9> left, 
			      	      std::array<float, 9> cen,  std::array<float, 9> right,
				      std::array<float, 9> ltop, std::array<float, 9> top, 
				      std::array<float, 9> rtop); 

	std::array<float, 9> getMSweights(const std::array<const float, 9> &beta, const int ksit); 

	float combine(std::array<float, 9> lbot, std::array<float, 9> bot, std::array<float, 9> rbot,
		      std::array<float, 9> left, std::array<float, 9> cen, std::array<float, 9> right, 
		      std::array<float, 9> ltop, std::array<float, 9> top, std::array<float, 9> rtop, 
		      std::array< std::array<float, 9>, 9> weights, std::array<float, 9> wsm);

	void MSinterp(const std::array< std::vector<std::vector< float>>, 3>, img_in, 
		      std::array< std::vector< std::vector>>
}

#endif 
