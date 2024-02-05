#include <iostream>

// CUDA kernel to print message from each thread
__global__ void helloCUDA()
{
    int threadId = threadIdx.x;
    int totalThreads = blockDim.x * gridDim.x;

    printf("Hello, World! from thread %d out of %d. GRID DIM %d and BLOCK DIM %d\n", threadId, totalThreads, blockDim.x, gridDim.x);
    //printf("GRID DIM %d and BLOCK DIM %d\n", blockDim.x, gridDim.x);
}

int main()
{
    // Specify the number of blocks and threads per block
    int numBlocks = 2; // You can change this to the desired number of blocks (Block Dimension)
    int threadsPerBlock = 5; // You can change this to the desired number of threads per block (Grid Dimension)

    // Launch the helloCUDA kernel with the specified number of blocks and threads per block
    helloCUDA<<<numBlocks, threadsPerBlock>>>();

    // Synchronize to make sure the kernel is done before exiting
    cudaDeviceSynchronize();

    // Check for errors during kernel launch
    cudaError_t cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(cudaError) << std::endl;
        return -1;
    }

    std::cout << "CUDA Kernel executed successfully!" << std::endl;

    return 0;
}
