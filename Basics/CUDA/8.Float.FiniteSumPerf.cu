#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thread>
#include <random>
#include "0.CUDA_functions.h"

#include <cblas.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <iostream>
#include <thread>
#include <lapacke.h>
#include <fstream>
#include <random>
#include "JBG_BLAS.single.h"

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
    const int max_size = std::atoi(argv[2]);

    // Create a random number generator =>  Get a Seed from random device
    std::mt19937_64 rng(13);
    std::uniform_real_distribution<float> dist(0.0, 1.0);

    float *results_GPU = (float *)malloc(max_size * 2 * sizeof(float));

    int n = 1; // Code For loop 2^n
    for (int i = 0; i < max_size; i++)
    {
        n *= 2;
        cout << "\n-------------------------------------------------------->  Vector Size:" << n << "\n";
        int size = n;
        results_GPU[2*i+0]=i;

        // --- Creating events for timing
        float elapsed_timeGPU, elapsed_timeCPU;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        float *a = new float[size];

        init_sum(a, r, size);
        float sum_exact = finitesum_exact(r, size);

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
        // finitesum_GPU<<<gridSize, blockSize>>>(dev_a, dev_c, size);
        reduce2<<<gridSize, blockSize, blockSize * sizeof(float)>>>(dev_a, dev_c);
        ///////////////////////////////////////////////////////////////////////////////
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed_timeGPU, start, stop);
        printf("GPU Reduce0 - Elapsed time:  %3.3f ms.", elapsed_timeGPU);
        cudaMemcpy(c_gpu, dev_c, gridSize * sizeof(float), cudaMemcpyDeviceToHost);

        float GPUfinalsum = kahanSum(c_gpu, gridSize);

        cudaFree(dev_a);
        cudaFree(dev_c);

        cudaEventRecord(start, 0);
        float sum_CPU = kahanSum(a, size);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed_timeCPU, start, stop);
        printf("\nCPU Reduce - Elapsed time:  %3.3f ms \n", elapsed_timeCPU);

        std::cout << "GPU speedup is x" << elapsed_timeCPU / elapsed_timeGPU << std::endl;
        results_GPU[2*i+1]=elapsed_timeCPU / elapsed_timeGPU;
        std::cout << "GRIDSIZE:" << gridSize << std::endl;

        std::cout << "Exact results:" << sum_exact << " / GPU results: " << GPUfinalsum << " / CPU results: " << sum_CPU << std::endl;
        // std::cout << "GPU results: " << GPUfinalsum << " / CPU results: " << sum_CPU << std::endl;
        std::cout << "diff CPU:" << sum_exact - sum_CPU << " / diff GPU: " << sum_exact - GPUfinalsum << std::endl;
        std::cout << "Diff /CPU-GPU/:" << GPUfinalsum - sum_CPU << "\n";

        delete[] a;
        delete[] c_gpu;
    } // End of loop for 2^n

    // Write the Results  to file in binary format
    std::ofstream outfilea("Results_gpu", std::ios::out | std::ios::binary);
    if (outfilea.is_open())
    {
        outfilea.write(reinterpret_cast<char *>(results_GPU), sizeof(float) * 2 * max_size);
        outfilea.close();
        cout << "\nFiles Written sucessfully\n\n";
    }
    else
    {
        std::cerr << "Failed to open file/s" << std::endl;
        return 1;
    }

    return 0;
}
