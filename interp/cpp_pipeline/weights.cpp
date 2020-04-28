#include "weights.h"
#include "kernels.h"
#include <iostream>
#include <iomanip> 
#include <lapacke.h> 


//Constructor 
weights::weights (const int Ratio[], const float del[])
{
    dx[0] = del[0], dx[1] = del[1]; 
    r[0] = Ratio[0], r[1] = Ratio[1];
    if(std::min(dx[0], dx[1]) > 0.01)  l = 0.01;
    else l = 9.*std::min(dx[0], dx[1]);
    std::array<std::array<float, 9>, 9> K = {}; //The same for every ratio;  
         // First dim is rx*ry; 
    GetK(K); // Builds Covariance Matrix of Pixel Window  
    Decomp(K); //Decomposes K and into their Cholesky Versions
    ks.resize(r[0]*r[1], std::array<float, 9>() ); 
    GetKs(K); 
}

//Performs Cholesky Backsubstitution
template<int n> 
void
weights::cholesky(float (&b)[n], std::array<std::array<float, n>, n> K)
{
    /* Forward sub Ly = b */
    for(int i = 0; i < n; ++i){
        for(int j = 0; j < i; ++j) b[i] -= b[j]*K[i][j];
        b[i] /= K[i][i];
    }

    /* Back sub Ux = y */
    for(int i = n-1; i >= 0; --i){
        for(int j = i+1; j < n; ++j) b[i] -= K[j][i]*b[j];
        b[i] /= K[i][i];
    }
}

//Performs Cholesky Backsubstitution
template<int n> 
void
weights::cholesky(std::array<float, n> &b, std::array<std::array<float, n>, n> K)
{
    /* Forward sub Ly = b */
    for(int i = 0; i < n; ++i){
        for(int j = 0; j < i; ++j) b[i] -= b[j]*K[i][j];
        b[i] /= K[i][i];
    }

    /* Back sub Ux = y */
    for(int i = n-1; i >= 0; --i){
        for(int j = i+1; j < n; ++j) b[i] -= K[j][i]*b[j];
        b[i] /= K[i][i];
    }
}

//Builds the Covariance matrix K if uninitialized --> if(!init) GetK, weights etc.
//Four K totals to make the gammas.  
void
weights::GetK(std::array<std::array<float, 9>, 9> &K)
{

	std::array<std::array<float, 2>, 9> pnt = {{{-1, -1}, { 0, -1}, { 1, -1},
   				                    {-1,  0}, { 0,  0}, { 1,  0},
		  			            {-1,  1}, { 0,  1}, { 1,  1}}}; 

//Small K
    for(int i = 0; i < 9; ++i){
        for(int j = i; j < 9; ++j){
            K[i][j] = matern3(pnt[i], pnt[j], l, dx);
            K[j][i] = K[i][j]; 
        }
    }
}

//We need to to the decomposition outside of the GetK routine so we can use K to get the 
//EigenVectors and Values. 

void
weights::Decomp(std::array<std::array<float, 9>, 9> &K)
{
    std::array<double, 81> kt = {}; 
    for(int i = 0; i < 9; i++) 
        for(int j = i; j < 9; j++) 
            kt[i*9+j] = double(K[j][i]); 

    int info = LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'U', 9, kt.data(), 9);
     for(int i = 0; i < 9; i++)
        for(int j = i; j < 9; j++){
            K[i][j] = float(kt[i*9+j]);
            K[j][i] = K[i][j]; 
        } 
}

//Use a Cholesky Decomposition to solve for k*K^-1 
//Inputs: K, outputs w = k*K^-1. 
void 
weights::GetKs(const std::array<std::array< float, 9>, 9> K)
{
    //Locations of new points relative to i,j 
    std::vector<std::array<float,2>> pnt(r[0]*r[1], std::array<float, 2>()); 
//    float pnt[16][2]; 
    if(r[0] == 2 && r[1] == 2){
        pnt[0][0] = -0.25,  pnt[0][1] = -0.25; 
        pnt[1][0] =  0.25,  pnt[1][1] = -0.25; 
        pnt[2][0] = -0.25,  pnt[2][1] =  0.25; 
        pnt[3][0] =  0.25,  pnt[3][1] =  0.25; 
    }
    else if(r[0] == 4 && r[1]==4){
        pnt[0][0] = -.375,  pnt[0][1] = -.375; 
        pnt[1][0] = -.125,  pnt[1][1] = -.375; 
        pnt[2][0] = 0.125,  pnt[2][1] = -.375; 
        pnt[3][0] = 0.375,  pnt[3][1] = -.375; 
        pnt[4][0] = -.375,  pnt[4][1] = -.125; 
        pnt[5][0] = -.125,  pnt[5][1] = -.125; 
        pnt[6][0] = 0.125,  pnt[6][1] = -.125; 
        pnt[7][0] = 0.375,  pnt[7][1] = -.125; 
        pnt[8][0] = -.375,  pnt[8][1] = 0.125; 
        pnt[9][0] = -.125,  pnt[9][1] = 0.125; 
        pnt[10][0] = 0.125, pnt[10][1] = 0.125; 
        pnt[11][0] = 0.375, pnt[11][1] = 0.125; 
        pnt[12][0] = -.375, pnt[12][1] = 0.375; 
        pnt[13][0] = -.125, pnt[13][1] = 0.375; 
        pnt[14][0] = 0.125, pnt[14][1] = 0.375; 
        pnt[15][0] = 0.375, pnt[15][1] = 0.375; 
    }
    std::array<std::array<float, 2>, 9> spnt = {{{-1, -1}, { 0, -1}, { 1, -1},
   					         {-1,  0}, { 0,  0}, { 1,  0},
		  			         {-1,  1}, { 0,  1}, { 1,  1}}}; 

    std::array<float, 2> temp = {}; 
    //Build covariance vector between interpolant points and stencil 
     for(int i = 0; i < r[0]*r[1]; ++i){
        for(int j = 0; j < 9; ++j){
            ks[i][j] = matern3(pnt[i], spnt[j], l, dx); //cen
        }
     //Backsubstitutes for k^TK^{-1} 
        cholesky<9>(ks[i], K); 
   }
}



