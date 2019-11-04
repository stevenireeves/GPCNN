#ifndef INTERP_H
#define INTERP_H
#include <array>
#include <vector>
#include <cmath> 
#include "weights.h"


class GP
{
public:

	/* Member data */ 
	std::vector<std::array<std::array<float, 9>, 9> > weight;
	std::vector<std::array<float, 9> > gammas;
	std::array<std::array<float, 9>, 9> vectors; 
	std::array<float, 9> eigen;
	int insize[3];

	/* Constructor */ 
	GP(std::vector<std::array<std::array<float, 9>, 9> > wts, std::vector<std::array<float, 9> > gam, 
		   std::array<std::array<float, 9>, 9> vec, std::array<float, 9> eig, const int size[3]){
		weight = wts;
		gammas = gam;
		vectors = vec; 
		eigen = eig; 
		insize[0] = size[0], insize[1] = size[1], insize[2] = size[2]; 
	}

	GP(const weights wgts, const int size[3]){
		weight = wgts.ks; 
		gammas = wgts.gam; 
		vectors = wgts.V;
		eigen = wgts.lam;
		insize[0] = size[0], insize[1] = size[1], insize[2] = size[2]; 
	}

	/* Member functions */ 
	float dot(const std::array<float, 9> vec1, const std::array<float, 9> vec2); 

	void interp(const std::vector<const float> img_in, std::vector<float> img_out); 

	std::array<float, 9> load(const std::vector<float> img_in, 
		       		  const int k, const int j, const int i); 

	std::array<float, 9> get_beta(std::array<float, 9> lbot, std::array<float, 9> bot,
				      std::array<float, 9> rbot, std::array<float, 9> left, 
			      	      std::array<float, 9> cen,  std::array<float, 9> right,
				      std::array<float, 9> ltop, std::array<float, 9> top, 
				      std::array<float, 9> rtop); 

	std::array<float, 9> getMSweights(const std::array<float, 9> &beta, const int ksit); 

	float combine(std::array<float, 9> lbot, std::array<float, 9> bot, std::array<float, 9> rbot,
		      std::array<float, 9> left, std::array<float, 9> cen, std::array<float, 9> right, 
		      std::array<float, 9> ltop, std::array<float, 9> top, std::array<float, 9> rtop, 
		      std::array< std::array<float, 9>, 9> weight, std::array<float, 9> wsm);

	void MSinterp(const std::vector<float> img_in, 
		      std::vector<float> &img_out, const int ry, const int rx); 
};

#endif
