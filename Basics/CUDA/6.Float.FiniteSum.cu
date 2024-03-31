#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thread>
#include <random>
#include "0.CUDA_functions.h"
#define MI_BLOCKSIZE 256
using namespace std;

__global__ void reduce0(int *g_idata, int *g_odata)
{
    extern __shared__ int sdata[];
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = g_idata[i];
    __syncthreads();
    // do reduction in shared mem
    for (unsigned int s = 1; s < blockDim.x; s *= 2)
    {
        if (tid % (2 * s) == 0)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    // write result for this block to global mem
    if (tid == 0)
        g_odata[blockIdx.x] = sdata[0];
}

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        std::cerr << "Usage of Finite SUM " << argv[0] << " 1/r (factor) n (Dimension)" << std::endl;
        return 1;
    }
    const float r = 1.0 / (float)std::atoi(argv[1]);
    const int size = std::atoi(argv[2]);

    auto start = std::chrono::high_resolution_clock::now();
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    long double elapsed_time, elapsed_time2;

    // Create a random number generator =>  Get a Seed from random device
    std::mt19937_64 rng(13);
    std::uniform_real_distribution<double> dist(0.0, 1.0); 

    float *a = new float[size];
    float *c_gpu = new float[1];

    init_sum (a,r,size);
    float sum_exact = finitesum_exact(r,size); 

    float *dev_a;
    float *dev_c;
    cudaMalloc((void **)&dev_a, size * sizeof(float));
    cudaMalloc((void **)&dev_c, sizeof(float));

    //start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(dev_a, a, size * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = MI_BLOCKSIZE;
    int gridSize = (size + blockSize - 1) / blockSize;
    start = std::chrono::high_resolution_clock::now();
    
    ///////////////////////////////////////////////////////////////////////////////
    finitesum_GPU<<<gridSize, blockSize>>>(dev_a, dev_c, size);
    ///////////////////////////////////////////////////////////////////////////////

    //stop = std::chrono::high_resolution_clock::now();
    cudaMemcpy(c_gpu, dev_c, sizeof(float), cudaMemcpyDeviceToHost);
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    elapsed_time = duration.count() * 1.e-9;
    std::cout << "GPU time:" << elapsed_time << std::endl;

    cudaFree(dev_a);
    cudaFree(dev_c);

    start = std::chrono::high_resolution_clock::now();
    float sum_CPU = finitesum_CPU(a,size);
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    elapsed_time2 = duration.count() * 1.e-9;
    std::cout << "CPU time:" << elapsed_time2 << std::endl;
    std::cout << "GPU speedup is x" << elapsed_time2 / elapsed_time << std::endl;
    std::cout << "GRIDSIZE:" << gridSize << std::endl;

    std::cout << "Exact results:" << sum_exact << " / GPU results: " << c_gpu[0] << " / CPU results: " << sum_CPU << std::endl;
    std::cout << "diff CPU:" << sum_exact-sum_CPU << " / diff GPU: " << sum_exact-c_gpu[0] << std::endl;

    delete[] a;
    delete[] c_gpu;

    return 0;
}
