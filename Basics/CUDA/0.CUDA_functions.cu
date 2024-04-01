#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thread>
#include <random>
#define MI_BLOCKSIZE 256
using namespace std;


__global__ void reduce2(float *g_idata, float *g_odata)
{
    extern __shared__ float sdata[];
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = g_idata[i];
    __syncthreads();
    // do reduction in shared mem
    for (unsigned int s = blockDim.x/2;s>0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    // write result for this block to global mem
    
    if (tid == 0)
        {g_odata[blockIdx.x] = sdata[0];}
}

__global__ void reduce1(float *g_idata, float *g_odata)
{
    extern __shared__ float sdata[];
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = g_idata[i];
    __syncthreads();
    // do reduction in shared mem
    for (unsigned int s = 1; s < blockDim.x; s *= 2)
    {
        int index = 2*s*tid;
        if (index < blockDim.x)
        {
            sdata[index] += sdata[index + s];
        }
        __syncthreads();
    }
    // write result for this block to global mem
    
    if (tid == 0)
        {g_odata[blockIdx.x] = sdata[0];}
}

__global__ void reduce0(float *g_idata, float *g_odata)
{
    extern __shared__ float sdata[];
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
        {g_odata[blockIdx.x] = sdata[0];}
}

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

float kahanSum(const float* array, size_t size) {
    float sum = 0.0;
    float compensation = 0.0;

    for (size_t i = 0; i < size; ++i) {
        float value = array[i];
        float y = value - compensation;
        float temp_sum = sum + y;
        compensation = (temp_sum - sum) - y;
        sum = temp_sum;
    }

    return sum;
}

void init_sum(float *a, float r,int n)
{
        for (int i = 0; i < n; i++)
        {
            a[i]=std::pow(r,i); 
        }
}
void init_sum2(float *a, float r,int n)
{
        for (int i = 0; i < n; i++)
        {
            a[i]=(float)i; 
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

