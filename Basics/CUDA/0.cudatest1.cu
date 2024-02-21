#include <iostream>
#include <cuda_runtime.h>

__global__ void myKernel()
{
    // Kernel code
}

// CUDA kernel to print "Hello, World!" from each thread
__global__ void helloCUDA()
{
    printf("Hello, World! from thread=%d, and block=%d\n", threadIdx.x, blockIdx.x);
}

int main()
{
    int blockSize, minGridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, myKernel, 0, 0);

    // Calculate the maximum number of blocks based on the GPU's resource limits
    // gridSize = (1<<31) / blockSize;
    int32_t gridSize = static_cast<int32_t>((1U << 31) / blockSize);

    std::cout << "blocksize: " << blockSize << std::endl;
    std::cout << "Maximum number of blocks: " << gridSize << std::endl;

    // Launch the helloCUDA kernel with a single block and 10 threads
    helloCUDA<<<1, 30>>>();

    // Synchronize to make sure the kernel is done before exiting
    cudaDeviceSynchronize();

    // Check for errors during kernel launch
    cudaError_t cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess)
    {
        std::cerr << "CUDA error: " << cudaGetErrorString(cudaError) << std::endl;
        return -1;
    }

    std::cout << "CUDA Kernel executed successfully!" << std::endl;

    return 0;
}
