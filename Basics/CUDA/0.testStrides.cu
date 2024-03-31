#include <iostream>
#include <vector>
#include <cuda_runtime.h>

#define N 1024 // Size of vectors (should be multiple of block size)
#define THREADS_PER_BLOCK 256

__global__ void dot_product(float *a, float *b, float *result) {
    __shared__ float partialSum[THREADS_PER_BLOCK];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int localIndex = threadIdx.x;

    partialSum[localIndex] = a[tid] * b[tid];
    __syncthreads();

    // Perform reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (localIndex < stride) {
            partialSum[localIndex] += partialSum[localIndex + stride];
        }
        __syncthreads();
    }

    // Store the result
    if (localIndex == 0) {
        result[blockIdx.x] = partialSum[0];
    }
}

__global__ void sum_kernel(int nvalues, double nThreads, double *values)  // WRONG usage of reduction. Tries GRID WIDE REDUCT
{   int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (nThreads > 1)
    {   int middle = nThreads / 2;
        int j = i + middle;
        if ( i < middle )
        {   if (j < nvalues )
            {   values[i] += values[j]; }
            else
            {   values[i] += 0;  }
        }
        __syncthreads();
        nThreads = middle;
    }
    if (i == 0)
        printf("T0 ------> %0.f \n", values[0] );
}

int main() {
    float *a, *b, *result;
    float *d_a, *d_b, *d_result;
    float finalResult = 0;

    // Allocate memory on host
    a = new float[N];
    b = new float[N];
    result = new float[N / THREADS_PER_BLOCK];

    // Initialize input vectors
    for (int i = 0; i < N; i++) {
        a[i] = 1.0f;
        b[i] = 1.0f;
    }

    // Allocate memory on device
    cudaMalloc((void**)&d_a, N * sizeof(float));
    cudaMalloc((void**)&d_b, N * sizeof(float));
    cudaMalloc((void**)&d_result, N / THREADS_PER_BLOCK * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dot_product<<<N / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_a, d_b, d_result);

    // Copy result from device to host
    cudaMemcpy(result, d_result, N / THREADS_PER_BLOCK * sizeof(float), cudaMemcpyDeviceToHost);

    // Final reduction on CPU
    for (int i = 0; i < N / THREADS_PER_BLOCK; i++) {
        finalResult += result[i];
    }

    std::cout << "Dot Product: " << finalResult << std::endl;

    // Free memory
    delete[] a;
    delete[] b;
    delete[] result;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);

    return 0;
}
