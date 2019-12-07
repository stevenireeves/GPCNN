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
    l = 0.01; // 12.*std::min(dx[0], dx[1]);
    sig = std::min(dx[0], dx[1]);
    std::array<std::array<float, 9>, 9> K = {}; //The same for every ratio;  
    std::array<std::array<float, 25>, 25> Ktot = {}; // The same for every ratio; 
    std::vector<std::array<float, 25> > kt(r[0]*r[1], std::array<float, 25>{{0}});
         // First dim is rx*ry; 
    GetK(K, Ktot); // Builds Covariance Matrices of Base Sample and Extended Samples/stencils  
    GetEigen(); //Gets Eigenvalues and Vectors from K for use in the interpolation
    Decomp(K, Ktot); //Decomposes K and Ktot into their Cholesky Versions
    ks.resize(r[0]*r[1], std::array<std::array<float, 9>, 9>() ); 
    GetKs(K); 
    // K and Ktot are not actually necessary for the rest of the weights interpolation 
    // They are only used to construct the weights w = ks^T Kinv 
    // and gam = Rinv Q^T kt; 
    // ks, gam, lam and V are part of the class and will be used in the main interpolation routine. 
    gam.resize(r[0]*r[1], std::array<float, 9>());
    GetKtotks(Ktot, kt);

    for(int i = 0; i < r[0]*r[1]; ++i){
        GetGamma(ks[i], kt[i], gam[i]); //Gets the gamma's
    }
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
weights::GetK(std::array<std::array<float, 9>, 9> &K, std::array<std::array<float, 25>, 25> &Ktot)
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

    std::array<std::array<float, 2>, 25>  spnt = {{{-2, -2}, {-1, -2}, {0, -2}, {1, -2}, {2, -2}, 
		  			  	   {-2, -1}, {-1, -1}, {0, -1}, {1, -1}, {2, -1},
		  			 	   {-2,  0}, {-1,  0}, {0,  0}, {1,  0}, {2,  0}, 
		  				   {-2,  1}, {-1,  1}, {0,  1}, {1,  1}, {2,  1}, 
		  				   {-2,  2}, {-1,  2}, {0,  2}, {1,  2}, {2,  2}}}; 

    for(int i = 0; i < 25; ++i)
        for(int j = i; j <25; ++j){
            Ktot[i][j] = matern3(spnt[i], spnt[j], l, dx); 
            Ktot[j][i] = Ktot[i][j]; 
        }
}

//We need to to the decomposition outside of the GetK routine so we can use K to get the 
//EigenVectors and Values. 

void
weights::Decomp(std::array<std::array<float, 9>, 9> &K, std::array<std::array<float, 25>, 25>  &Kt)
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

    std::vector<double> temp(625, 0); 
    for(int i = 0; i < 25; i++)
        for(int j = 0; j < 25; j++)
            temp[j+i*25] = double(Kt[j][i]); 
    info = LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'U', 25, temp.data(), 25); 
     for(int i = 0; i < 25; i++)
        for(int j = i; j < 25; j++){
            Kt[i][j] = float(temp[i*25+j]);
            Kt[j][i] = Kt[i][j]; 
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
	    temp[0] = spnt[j][0] - 1.f, temp[1] = spnt[j][1] - 1.f; //lbot
	    ks[i][0][j] = matern3(pnt[i], temp, l, dx); 

            temp[0] = spnt[j][0], temp[1] = spnt[j][1] - 1.f; //bot
            ks[i][1][j] = matern3(pnt[i], temp, l, dx);

	    temp[0] = spnt[j][0] + 1.f, temp[1] = spnt[j][1] - 1.f; //rbot
	    ks[i][2][j] = matern3(pnt[i], temp, l, dx);     

            temp[0] = spnt[j][0] - 1.f, temp[1] = spnt[j][1]; //left
            ks[i][3][j] = matern3(pnt[i], temp, l, dx);

            ks[i][4][j] = matern3(pnt[i], spnt[j], l, dx); //cen

            temp[0] = spnt[j][0] + 1.f, temp[1] = spnt[j][1]; //right
            ks[i][5][j] = matern3(pnt[i], temp, l, dx);
   
            temp[0] = spnt[j][0] - 1.f, temp[1] = spnt[j][1] + 1.f;
            ks[i][6][j] = matern3(pnt[i], temp, l, dx); //ltop

            temp[0] = spnt[j][0], temp[1] = spnt[j][1] + 1.f;
            ks[i][7][j] = matern3(pnt[i], temp, l, dx); //top

            temp[0] = spnt[j][0] +1.f, temp[1] = spnt[j][1] + 1.f; 
            ks[i][8][j] = matern3(pnt[i], temp, l, dx); //rtop
        }
     //Backsubstitutes for k^TK^{-1} 
        for(int k = 0; k < 9; ++k)
            cholesky<9>(ks[i][k], K); 
   }
}

// Here we are using Kt to get the weights for the overdetermined  
// In this case, we will have 16 new points
// Therefore, we will need 16 b =  k*^T Ktot^(-1)
// K1 is already Choleskied  
void 
weights::GetKtotks(const std::array<std::array<float, 25>, 25> K1, std::vector<std::array<float, 25> > &kt)
{
   //Locations of new points relative to i,j 
    std::vector<std::array<float,2>> pnt(r[0]*r[1], std::array<float,2>()); 
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
    //Super K positions
    std::array<std::array<float, 2>, 25>  spnt = 
                {{{-2, -2}, {-1, -2}, {0, -2}, {1, -2}, {2, -2}, 
		  {-2, -1}, {-1, -1}, {0, -1}, {1, -1}, {2, -1},
		  {-2,  0}, {-1,  0}, {0,  0}, {1,  0}, {2,  0}, 
		  {-2,  1}, {-1,  1}, {0,  1}, {1,  1}, {2,  1}, 
		  {-2,  2}, {-1,  2}, {0,  2}, {1,  2}, {2,  2}}}; 

    for(int i = 0; i < r[0]*r[1]; i++){
       for (int j = 0; j < 25; j++){
            kt[i][j] = matern3(pnt[i], spnt[j], l, dx); 
       }
       cholesky<25>(kt[i], K1); 
    } 
}

//Each point will have its
//own set of gammas. 
//Use x = R^-1Q'b 
void
weights::GetGamma(std::array<std::array<float, 9>, 9> const& k,
             std::array<float, 25> const& kt, 
             std::array<float,9> &ga)
{
//Extended matrix Each column contains the vector of coviarances corresponding 
//to each sample (weno-like stencil, dx)


    std::array<float, 225> A =
    {k[0][0], 0      , 0      , 0      , 0      , 0      , 0      , 0      , 0      ,  //i-2 j-2 
     k[0][1], k[1][0], 0      , 0      , 0      , 0      , 0      , 0      , 0      ,  //i-1 j-2
     k[0][2], k[1][1], k[2][0], 0      , 0      , 0      , 0      , 0      , 0      ,  //i   j-2
     0      , k[1][2], k[2][1], 0      , 0      , 0      , 0      , 0      , 0      ,  //i+1 j-2
     0      , 0      , k[2][2], 0      , 0      , 0      , 0      , 0      , 0      ,  //i+2 j-2
     k[0][3], 0      , 0      , k[3][0], 0      , 0      , 0      , 0      , 0      ,  //i-2 j-1
     k[0][4], k[1][3], 0      , k[3][1], k[4][0], 0      , 0      , 0      , 0      ,  //i-1 j-1
     k[0][5], k[1][4], k[2][3], k[3][2], k[4][1], k[5][0], 0      , 0      , 0      ,  //i   j-1
     0      , k[1][5], k[2][4], 0      , k[4][2], k[5][1], 0      , 0      , 0      ,  //i+1 j-1
     0      , 0      , k[2][5], 0      , 0      , k[5][2], 0      , 0      , 0      ,  //i+2 j-1
     k[0][6], 0      , 0      , k[3][3], 0      , 0      , k[6][0], 0      , 0      ,  //i-2 j
     k[0][7], k[1][6], 0      , k[3][4], k[4][3], 0      , k[6][1], k[7][0], 0      ,  //i-1 j
     k[0][8], k[1][7], k[2][6], k[3][5], k[4][4], k[5][3], k[6][2], k[7][1], k[8][0],  //i   j
     0      , k[1][8], k[2][7], 0      , k[4][5], k[5][4], 0      , k[7][2], k[8][1],  //i+1 j
     0      , 0      , k[2][8], 0      , 0      , k[5][5], 0      , 0      , k[8][2],  //i+2 j
     0      , 0      , 0      , k[3][6], 0      , 0      , k[6][3], 0      , 0      ,  //i-2 j+1
     0      , 0      , 0      , k[3][7], k[4][6], 0      , k[6][4], k[7][3], 0      ,  //i-1 j+1
     0      , 0      , 0      , k[3][8], k[4][7], k[5][6], k[6][5], k[7][4], k[8][3],  //i   j+1
     0      , 0      , 0      , 0      , k[4][8], k[5][7], 0      , k[7][5], k[8][4],  //i+1 j+1
     0      , 0      , 0      , 0      , 0      , k[5][8], 0      , 0      , k[8][5],  //i+2 j+1
     0      , 0      , 0      , 0      , 0      , 0      , k[6][6], 0      , 0      ,  //i-2 j+2
     0      , 0      , 0      , 0      , 0      , 0      , k[6][7], k[7][6], 0      ,  //i-1 j+2
     0      , 0      , 0      , 0      , 0      , 0      , k[6][8], k[7][7], k[8][6],  //i   j+2
     0      , 0      , 0      , 0      , 0      , 0      , 0      , k[7][8], k[8][7],  //i+1 j+2
     0      , 0      , 0      , 0      , 0      , 0      , 0      , 0      , k[8][8]}; //i+2 j+2
//    std::cout<<std::fixed; 
//    std::cout<<std::setprecision(6); 
//    for (int i = 0; i < 25; i++){
//	   for (int j = 0; j < 9; j++) std::cout << A[i*9 + j] << '\t'; 
//	   std::cout<<std::endl; 
//    }
    int m = 25, n = 9, nrhs = 1; 
    int lda = 9, ldb = 1, info; 
    float temp[25]; 
    for(int i = 0; i < 25; i++){
	    temp[i] = kt[i];
    }
//    std::cout<< "Kt = " << std::endl; 
//    for(int i = 0; i < 25; i++) std::cout<< temp[i] << std::endl; 

    info = LAPACKE_sgels(LAPACK_ROW_MAJOR, 'N', m, n, nrhs, A.data(), lda, temp, ldb);
//    std::cout << "Gam = " << std::endl; 
//    for(int i = 0; i < 9; i++) std::cout<< temp[i] << std::endl; 
//    std::cin.get(); 
    for(int i = 0; i < 9; ++i) ga[i] = temp[i];
}
 

void 
weights::GetEigen()
{
	//Need to use double precision for E-vecs to get correct order. 
    std::array<double, 81>  A;
    std::array<std::array< float, 2>, 9> pnt = 
                           {{{-1, -1}, {0, -1}, {1, -1},  
                             {-1,  0}, {0,  0}, {1,  0}, 
                             {-1,  1}, {0,  1}, {1,  1}}}; 

    for (int j = 0; j < 9; ++j){
        for(int i = 0; i < 9; ++i){
             A[j*9 + i] = double(matern3(pnt[j], pnt[i], sig, dx)); //this is K_sig
        }
    }
    int N = 9, lda = 9, info, lwork;
    double temp[9] = {}; 
    LAPACKE_dsyevd(LAPACK_ROW_MAJOR, 'V', 'L', N, A.data(), lda, temp); 
    for (int j = 0; j < 9; ++j){
	lam[j] = float(temp[j]); 
        for(int i = 0; i < 9; ++i){
	       	V[j][i] =  float(A[j*9 + i]);
	}
    }
}
