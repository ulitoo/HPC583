#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thread>
#include <random>
#define MI_BLOCKSIZE 256
using namespace std;

__global__ void finitesum_GPU(float *a, float *c, int size)
{
    __shared__ float temp[MI_BLOCKSIZE];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int localIndex = threadIdx.x;
    temp[localIndex] = a[tid];
    __syncthreads();
    // Store the result in c Collecting from all Blocks with Atomic Add
    if (localIndex == 0)
    {
        float sum = 0;
        for (int i = 0; i < MI_BLOCKSIZE; i++)
        {
            sum += temp[i]; 
        }
        atomicAdd(c,sum);
    }
}

void init_sum(float *a, float r,int n)
{
        for (int i = 0; i < n; i++)
        {
            a[i]=std::pow(r,i); 
        }
}

float finitesum_CPU(float *a, int n)
{
        float sum=0;
        for (int i = 0; i < n; i++)
        {
            sum += a[i]; 
        }
        return sum;
}

float finitesum_exact(float r, int n)
{

    return ((1-std::pow(r,n))/(1-r));
}

