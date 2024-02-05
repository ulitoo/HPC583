#include <iostream>

// Matrix multiplication kernel
__global__ void matrixMultiply(int *a, int *b, int *c, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N)
    {
        int sum = 0;
        for (int i = 0; i < N; ++i)
        {
            sum += a[row * N + i] * b[i * N + col];
        }
        c[row * N + col] = sum;
    }
}

int main( int argc, char* argv[] )
{
    
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " n (Dimension of Matrix)" << std::endl;
        return 1;
    }

    const int N = std::atoi(argv[1]);  // Matrix size (N x N)


    // Matrix dimensions
    const int matrixSize = N * N * sizeof(int);

    // Host matrices
    int *h_A, *h_B, *h_C;

    // Allocate memory for host matrices
    h_A = new int[N * N];
    h_B = new int[N * N];
    h_C = new int[N * N];

    // Initialize host matrices
    for (int i = 0; i < N * N; ++i)
    {
        h_A[i] = i;
        h_B[i] = i;
    }

    // Allocate memory on the device (GPU)
    int *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, matrixSize);
    cudaMalloc((void **)&d_B, matrixSize);
    cudaMalloc((void **)&d_C, matrixSize);

    // Copy host matrices to device
    cudaMemcpy(d_A, h_A, matrixSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, matrixSize, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockSize(2, 2);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);

    // Launch matrix multiplication kernel
    matrixMultiply<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);

    // Copy result matrix from device to host
    cudaMemcpy(h_C, d_C, matrixSize, cudaMemcpyDeviceToHost);

    // Print the a matrix
    std::cout << "A Matrix:" << std::endl;
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            std::cout << h_A[i * N + j] << "\t";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    // Print the b matrix
    std::cout << "B Matrix:" << std::endl;
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            std::cout << h_B[i * N + j] << "\t";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    // Print the result matrix
    std::cout << "Result Matrix:" << std::endl;
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            std::cout << h_C[i * N + j] << "\t";
        }
        std::cout << std::endl;
    }

    // Free allocated memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
