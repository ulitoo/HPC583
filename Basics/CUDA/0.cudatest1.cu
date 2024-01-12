#include <iostream>

// CUDA kernel to print "Hello, World!" from each thread
__global__ void helloCUDA()
{
    printf("Hello, World! from thread %d\n", threadIdx.x);
}

int main()
{
    // Launch the helloCUDA kernel with a single block and 10 threads
    helloCUDA<<<1, 10>>>();

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
