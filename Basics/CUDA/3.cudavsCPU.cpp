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
#include "JBG_BLAS.h"

using namespace std;

int main(int argc, char *argv[])
{

    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " n (for 2^n Max Dimension of Matrices) s (seed)" << std::endl;
        return 1;
    }

    const int max_size = std::atoi(argv[1]);
    const int seed = std::atoi(argv[2]);

    cublasStatus_t stat;
    cudaError_t cudaStatus;
    cusolverStatus_t cusolverStatus;
    cusolverDnHandle_t handle;

    // declare arrays on the device
    double *d_A, *d_B, *d_Work;   // coeff. matrix, rhs, workspace
    int *d_pivot, *d_info, Lwork; // pivots, info, workspace size
    int info_gpu = 0;

    const double epsilon = double_machine_epsilon();
    int INFO;
    double *results_GPU = (double *)malloc(max_size * 7 * sizeof(double));
    double *results_lapack = (double *)malloc(max_size * 7 * sizeof(double));

    // Timers
    auto start = std::chrono::high_resolution_clock::now(); // variables for timing
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start); // elapsed time variable
    long double elapsed_time_GPU, elapsed_time_lapack;

    int n = 1; // Code For loop 2^n
    for (int i = 0; i < max_size; i++)
    {
        n *= 2;
        cout << "\n-------------------------------------------------------->  Matrix Size:" << n << "\n";

        // Create a random number generator =>  Get a Seed from random device
        std::mt19937_64 rng(seed);
        std::uniform_real_distribution<double> dist(0.0, 1.0);

        // Alloc Space for MATRICES Needed in Column Major Order
        double *matrixA = (double *)malloc(n * n * sizeof(double));
        double *matrixB = (double *)malloc(n * n * sizeof(double));
        double *matrixA_original = (double *)malloc(n * n * sizeof(double)); // in case they get overwritten
        double *matrixB_original = (double *)malloc(n * n * sizeof(double)); // in case they get overwritten

        // Other Variables
        int *IPIV = (int *)malloc(n * sizeof(int));

        // Create the HOST matrices A and B and fill it with random values
        for (int k = 0; k < n * n; k++)
        {
            matrixA[k] = dist(rng) - 0.5;
            matrixB[k] = dist(rng) - 0.5;
        }

        // Backup A and B Matrices
        Write_A_over_B(matrixA, matrixA_original, n, n);
        Write_A_over_B(matrixB, matrixB_original, n, n);

        cudaStatus = cudaGetDevice(0);
        cusolverStatus = cusolverDnCreate(&handle);
        // prepare memory on the device
        cudaStatus = cudaMalloc((void **)&d_A, n * n * sizeof(double));
        cudaStatus = cudaMalloc((void **)&d_B, n * n * sizeof(double));
        cudaStatus = cudaMalloc((void **)&d_pivot, n * sizeof(int));
        cudaStatus = cudaMalloc((void **)&d_info, sizeof(int));
        cudaStatus = cudaMemcpy(d_A, matrixA, n * n * sizeof(double), cudaMemcpyHostToDevice); // copy d_A <- A
        cudaStatus = cudaMemcpy(d_B, matrixB, n * n * sizeof(double), cudaMemcpyHostToDevice); // copy d_B <- B
        cusolverStatus = cusolverDnDgetrf_bufferSize(handle, n, n, d_A, n, &Lwork);            // compute buffer size and prepare memory
        cudaStatus = cudaMalloc((void **)&d_Work, Lwork * sizeof(double));
        // timer start
        start = std::chrono::high_resolution_clock::now();
        // LU factorization of d_A, with partial pivoting and row interchanges
        cusolverStatus = cusolverDnDgetrf(handle, n, n, d_A, n, d_Work, d_pivot, d_info);
        // use the LU factorization to solve the system d_A * X = d_B;
        // the solution overwrites d_B
        cusolverStatus = cusolverDnDgetrs(handle, CUBLAS_OP_N, n, n, d_A, n, d_pivot, d_B, n, d_info);
        
        stop = std::chrono::high_resolution_clock::now();                              // timer stop
                
        cudaStatus = cudaDeviceSynchronize();
        cudaStatus = cudaMemcpy(&info_gpu, d_info, sizeof(int), cudaMemcpyDeviceToHost); // d_info -> info_gpu
        printf("after getrf + getrs: info_gpu = %d\n", info_gpu);
        cudaStatus = cudaMemcpy(matrixB, d_B, n * n * sizeof(double), cudaMemcpyDeviceToHost); // copy d_B -> matrixB
        duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start); // elapsed time
        elapsed_time_GPU = duration.count() * 1.e-9;

        cout << "\nCheck Accuracy and time of **GPU** AX=B:";
        results_GPU[7 * i + 6] = elapsed_time_GPU;
        ErrorCalc_Display_v2(i, matrixA_original, matrixB_original, matrixB, results_GPU, n, n);

        // free GPU memory
        cudaStatus = cudaFree(d_A);
        cudaStatus = cudaFree(d_B);
        cudaStatus = cudaFree(d_pivot);
        cudaStatus = cudaFree(d_info);
        cudaStatus = cudaFree(d_Work);
        cusolverStatus = cusolverDnDestroy(handle);
        cudaStatus = cudaDeviceReset();

        // Restore A and B Matrices
        Write_A_over_B(matrixA_original, matrixA, n, n);
        Write_A_over_B(matrixB_original, matrixB, n, n);

        //  ----------------- Solve BLAS and compare with my implementation HERE!

        start = std::chrono::high_resolution_clock::now();
        LAPACK_dgesv(&n, &n, matrixA, &n, IPIV, matrixB, &n, &INFO);
        stop = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
        elapsed_time_lapack = duration.count() * 1.e-9;
        cout << "\nCheck Accuracy and time of LAPACK-CPU (dgesv) AX=B:";
        results_lapack[7 * i + 6] = elapsed_time_lapack;
        ErrorCalc_Display_v2(i, matrixA_original, matrixB_original, matrixB, results_lapack, n, n);

        // Free CPU Mem
        free(matrixA);
        free(matrixB);
        free(matrixA_original);
        free(matrixB_original);
        free(IPIV);

    } // End of loop for 2^n

    // Print the results
    cout << "\n================> FINAL RESULTS:\n";
    cout << "\nGPU:\n";
    for (int k = 0; k < max_size; k++)
    {
        for (int kl = 0; kl < 7; kl++)
        {
            cout << results_GPU[k * 7 + kl] << ",";
        }
        cout << "\n";
    }
    cout << "\nCPU-LAPACK:\n";
    for (int k = 0; k < max_size; k++)
    {
        for (int kl = 0; kl < 7; kl++)
        {
            cout << results_lapack[k * 7 + kl] << ",";
        }
        cout << "\n";
    }

    // Write the Results matrices to file in binary format
    std::ofstream outfilea("Results_gpu", std::ios::out | std::ios::binary);
    std::ofstream outfileb("Results_lapack", std::ios::out | std::ios::binary);
    if (outfilea.is_open() and outfileb.is_open())
    {
        outfilea.write(reinterpret_cast<char *>(results_GPU), sizeof(double) * 7 * max_size);
        outfilea.close();
        outfileb.write(reinterpret_cast<char *>(results_lapack), sizeof(double) * 7 * max_size);
        outfileb.close();
        cout << "\nFiles Written sucessfully\n\n";
    }
    else
    {
        std::cerr << "Failed to open file/s" << std::endl;
        return 1;
    }

    return 0;
}
