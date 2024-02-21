#include <iostream>
#include "cuda_runtime.h"
#include "cusolverDn.h"

// Matrix multiplication kernel
__global__ void matrixMultiply(double *a, double *b, double *c, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    //printf("N is:%d , thread x: %d and thread y: %d \n",N,threadIdx.x,threadIdx.y);
    printf("N is:%d , col %d and row: %d \n",N,col,row);
    if (row < N && col < N)
    {
        double sum = 0;
        for (int i = 0; i < N; ++i)
        {
            sum += a[row * N + i] * b[i * N + col];
        }
        c[row * N + col] = sum;
    }
}

void matrixPrint(double *a, int N)
{
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            std::cout << a[i * N + j] << "\t";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int getMaxBlockSize()
{
    int blockSize, minGridSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, matrixMultiply, 0, 0);
    return blockSize;
}

void solveLinearSystem(const double *A, const double *B, double *X, int N, int numSystems) {
    cusolverDnHandle_t cusolverH;
    cusolverDnCreate(&cusolverH);

    int *devInfo;
    cudaMalloc((void**)&devInfo, numSystems * sizeof(int));

    // Prepare pivoting array
    int *d_pivotArray;
    cudaMalloc((void**)&d_pivotArray, N * numSystems * sizeof(int));

    // Prepare workspace
    int workspaceSize;
    cusolverDnDgesvdjBatched_bufferSize(cusolverH, N, N, &workspaceSize);
    double *d_workspace;
    cudaMalloc((void**)&d_workspace, workspaceSize * numSystems * sizeof(double));

    // Solve the systems
    cusolverDnDgesvdjBatched(cusolverH, N, 1, A, N, d_pivotArray, B, N, X, N, d_workspace, workspaceSize, devInfo, numSystems);

    cudaFree(devInfo);
    cudaFree(d_pivotArray);
    cudaFree(d_workspace);
    cusolverDnDestroy(cusolverH);
}

int main(int argc, char *argv[])
{

    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " n (Dimension of Matrix)" << std::endl;
        return 1;
    }

    const int N = 3;  //std::atoi(argv[1]); // Matrix size (N x N)
    const int numSystems = 2; // Number of linear systems

    // Define the coefficient matrix A and the matrix B
    double A[N][N] = {{6, -2, 2},
                      {4,  2, 5},
                      {2,  8, 7}};
    double B[numSystems][N] = {{16, 35, 64},
                                {10, 20, 30}};

    // Copy matrix A and matrix B to device memory
    double *d_A, *d_B;
    cudaMalloc((void**)&d_A, N * N * sizeof(double));
    cudaMalloc((void**)&d_B, N * numSystems * sizeof(double));
    cudaMemcpy(d_A, A, N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * numSystems * sizeof(double), cudaMemcpyHostToDevice);

    // Allocate memory for the solution matrix X
    double *X;
    cudaMalloc((void**)&X, N * numSystems * sizeof(double));

    // Solve the linear systems
    solveLinearSystem(d_A, d_B, X, N, numSystems);


    // Print the a matrices
    //std::cout << "A Matrix:" << std::endl;
    //matrixPrint(h_A, N);
    //std::cout << "B Matrix:" << std::endl;
    //matrixPrint(h_B, N);
    //std::cout << "Result Matrix:" << std::endl;
    //matrixPrint(h_C, N);

    // Print the solution matrix X
    std::cout << "Solution matrix X:\n";
    double X_host[N * numSystems];
    cudaMemcpy(X_host, X, N * numSystems * sizeof(double), cudaMemcpyDeviceToHost);
    for (int i = 0; i < numSystems; ++i) {
        std::cout << "System " << i + 1 << ":\n";
        for (int j = 0; j < N; ++j) {
            std::cout << X_host[i * N + j] << " ";
        }
        std::cout << "\n";
    }


    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(X);

    return 0;
}
