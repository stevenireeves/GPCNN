#include <cuda_runtime.h>
#include <GP.h> 
using namespace std;

__global__ void interp(float *img_in, float * img_out, int width, int height)
{
    int pos_x = blockIdx.x * blockDim.x + threadIdx.x;
    int pos_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (pos_x >= width || pos_y >= height)
        return;
    auto cen = GP::load(img_in, j, i);
    for(int idy = 0; idy < ry; idy++){
        int jj = j*ry + idy;
        int ind =jj*outsize[1];	
        for(int idx = 0; idx < rx; idx++){
	        int ii = i*rx + idx; 
	        int idk = idx*ry + idy;
	        img_out[ind + ii] = GP::dot(weight[idk], cen); 
            }
    }
}

__device__ float GP( float st[9], float weights[9])
{
	float result = 0.0f; 
	for(int i = 0; i<9; ++i) result+= weights[i]*st[i]; 
	return result; 
}

/*
int main()
{
    //load image
    CImg<unsigned char> src("SAGAN.bmp");
    int width = src.width();
    int height = src.height();
    unsigned long size = src.size();

    //create pointer to image
    unsigned char *h_src = src.data();

    CImg<unsigned char> gs(width, height, 1, 1);
    unsigned char *h_gs = gs.data();

    unsigned char *d_src;
    unsigned char *d_gs;

    cudaMalloc((void**)&d_src, size);
    cudaMalloc((void**)&d_gs, width*height*sizeof(unsigned char));

    cudaMemcpy(d_src, h_src, size, cudaMemcpyHostToDevice);

    //launch the kernel
    dim3 blkDim (16, 16, 1);
    dim3 grdDim ((width + 15)/16, (height + 15)/16, 1);
    rgb2gray<<<grdDim, blkDim>>>(d_src, d_gs, width, height);

    //wait until kernel finishes
    cudaDeviceSynchronize();

    //copy back the result to CPU
    cudaMemcpy(h_gs, d_gs, width*height, cudaMemcpyDeviceToHost);

    cudaFree(d_src);
    cudaFree(d_gs);

    CImg<unsigned char> out(h_gs,width,height);
	out.save("GSSAGAN.bmp");
    return 0;
} */ 
