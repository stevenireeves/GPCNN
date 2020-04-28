#ifndef WEIGHTS_H
#define WEIGHTS_H
#include <vector>
#include <array> 
#include <cmath> 

// weights class for GP superresoltuion

class weights
{
    public: 
    weights(const int ratio[], const float del[]);
    ~weights(){};      
    
    // Member data
    int r[2]; 
    float dx[2];  
    //
    //  Weights to be applied for interpolation
    //
    std::vector<std::array<float, 9> > ks; 
    float l;

// Linear Algebra Functions
    template<int n>
    inline
    static float inner_prod(const float x[n], const float y[n])
    {
        float result = 0.f; 
        for(int i = 0; i < n; ++i) result += x[i]*y[i];
        return result;  
    }

    template<int n> 
    void
    cholesky(float (&b)[n], std::array<std::array<float, n>, n> K); 
    
    template<int n> 
    void
    cholesky(std::array<float, n> &b, std::array<std::array<float, n>, n> K);

    void
    Decomp(std::array<std::array<float, 9>, 9> &K); 
// GP functions 
    // Build K makes the Coviarance Kernel Matrices for each Samples 
    void GetK(std::array<std::array<float, 9>, 9> &K); 
    //
    // Get Weights builds k*Kinv 
    //
    void GetKs(const std::array<std::array<float, 9>, 9> K);
};
#endif 
