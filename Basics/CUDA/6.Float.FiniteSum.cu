#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thread>
#include <random>
#include "0.CUDA_functions.h"
#define MI_BLOCKSIZE 256
using namespace std;


int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        std::cerr << "Usage of Finite SUM " << argv[0] << " 1/r (factor) n (Dimension)" << std::endl;
        return 1;
    }
    const float r = 1.0 / (float)std::atoi(argv[1]);
    const int size = std::atoi(argv[2]);
	
    // --- Creating events for timing
	float elapsed_timeGPU, elapsed_timeCPU;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

    // Create a random number generator =>  Get a Seed from random device
    std::mt19937_64 rng(13);
    std::uniform_real_distribution<float> dist(0.0, 1.0); 

    float *a = new float[size];

    init_sum(a,r,size);
    float sum_exact = finitesum_exact(r,size); 

    float *dev_a;
    float *dev_c;
    cudaMalloc((void **)&dev_a, size * sizeof(float));
    cudaMemcpy(dev_a, a, size * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = MI_BLOCKSIZE;
    int gridSize = (size + blockSize - 1) / blockSize;
    cudaMalloc((void **)&dev_c, gridSize * sizeof(float));
    float *c_gpu = new float[gridSize];
 
    
    cudaEventRecord(start, 0);
    ///////////////////////////////////////////////////////////////////////////////
    //finitesum_GPU<<<gridSize, blockSize>>>(dev_a, dev_c, size);
    reduce2<<<gridSize, blockSize, blockSize * sizeof(float)>>>(dev_a, dev_c);
    ///////////////////////////////////////////////////////////////////////////////
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_timeGPU, start, stop);
	printf("GPU Reduce0 - Elapsed time:  %3.3f ms.", elapsed_timeGPU);	
    cudaMemcpy(c_gpu, dev_c, gridSize * sizeof(float), cudaMemcpyDeviceToHost);
    
    float GPUfinalsum = kahanSum (c_gpu,gridSize);

    cudaFree(dev_a);
    cudaFree(dev_c);

    cudaEventRecord(start, 0);
    float sum_CPU = kahanSum(a,size);
    cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_timeCPU, start, stop);
	printf("\nCPU Reduce - Elapsed time:  %3.3f ms \n", elapsed_timeCPU);	

    std::cout << "GPU speedup is x" << elapsed_timeCPU / elapsed_timeGPU << std::endl;
    std::cout << "GRIDSIZE:" << gridSize << std::endl;

    std::cout << "Exact results:" << sum_exact << " / GPU results: " << GPUfinalsum << " / CPU results: " << sum_CPU << std::endl;
    //std::cout << "GPU results: " << GPUfinalsum << " / CPU results: " << sum_CPU << std::endl;
    std::cout << "diff CPU:" << sum_exact-sum_CPU << " / diff GPU: " << sum_exact-GPUfinalsum << std::endl;
    std::cout << "Diff /CPU-GPU/:" << GPUfinalsum-sum_CPU << "\n";

    delete[] a;
    delete[] c_gpu;

    return 0;
}
