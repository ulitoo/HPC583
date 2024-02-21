#include <iostream>
#include "cuda_runtime.h"
#include "cusolverDn.h"

void solveLinearSystem(const double *A, const double *B, double *X, int N, int numSystems) {
    cusolverDnHandle_t cusolverH;
    cusolverDnCreate(&cusolverH);

    int *devInfo;
    cudaMalloc((void**)&devInfo, numSystems * sizeof(int));

    // Prepare pivoting array
    int *d_pivotArray;
    cudaMalloc((void**)&d_pivotArray, N * numSystems * sizeof(int));

    // Prepare workspace
    size_t workspaceSize;
    cusolverDnDgetrsBatched_bufferSize(cusolverH, CUBLAS_OP_N, N, 1, &workspaceSize, numSystems);
    double *d_workspace;
    cudaMalloc((void**)&d_workspace, workspaceSize);

    // Solve the systems
    cusolverDnDgetrsBatched(cusolverH, CUBLAS_OP_N, N, 1, A, N, d_pivotArray, B, N, X, N, devInfo, numSystems);

    cudaFree(devInfo);
    cudaFree(d_pivotArray);
    cudaFree(d_workspace);
    cusolverDnDestroy(cusolverH);
}

int main() {
    const int N = 3; // Size of the matrix
    const int numSystems = 3; // Number of linear systems

    // Define the coefficient matrix A and the matrices B and X
    double A[numSystems][N][N] = {
        {{6, -2, 2}, {4, 2, 5}, {2, 8, 7}},
        {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
        {{3, 2, 1}, {6, 5, 4}, {9, 8, 7}}
    };
    double B[numSystems][N][N] = {
        {{16, 35, 64}, {16, 35, 64}, {16, 35, 64}},
        {{10, 20, 30}, {10, 20, 30}, {10, 20, 30}},
        {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}
    };
    double X[numSystems][N][N]; // Solution matrices

    // Copy matrices A and B to device memory
    double *d_A, *d_B, *d_X;
    cudaMalloc((void**)&d_A, N * N * numSystems * sizeof(double));
    cudaMalloc((void**)&d_B, N * N * numSystems * sizeof(double));
    cudaMemcpy(d_A, A, N * N * numSystems * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * N * numSystems * sizeof(double), cudaMemcpyHostToDevice);

    // Solve the linear systems
    solveLinearSystem(d_A, d_B, reinterpret_cast<double*>(X), N, numSystems);

    // Copy the solutions back to the host
    cudaMemcpy(X, d_X, N * N * numSystems * sizeof(double), cudaMemcpyDeviceToHost);

    // Print the solution matrices
    std::cout << "Solution matrices X:\n";
    for (int i = 0; i < numSystems; ++i) {
        std::cout << "System " << i + 1 << ":\n";
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                std::cout << X[i][j][k] << " ";
            }
            std::cout << "\n";
        }
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_X);

    return 0;
}
