#include <cblas.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#define N 4
#define BILLION 1000000000L;

int main(int argc, char *argv[]) {
    struct timespec start, stop; // variables for timing
    double accum;                // elapsed time variable
    cublasStatus_t stat;
    cudaError_t cudaStatus;
    cusolverStatus_t cusolverStatus;
    cusolverDnHandle_t handle;

    // declare arrays on the host
    float *A, *B1, *B; // A - NxN matrix , B1 - auxiliary NxN matrix.
    // B=A*B1 - NxN matrix, all on the host
    // declare arrays on the device
    float *d_A, *d_B, *d_Work;    // coeff. matrix, rhs, workspace
    int *d_pivot, *d_info, Lwork; // pivots, info, workspace size
    int info_gpu = 0;

    // prepare memory on the host
    A = (float *)malloc(N * N * sizeof(float));
    B = (float *)malloc(N * N * sizeof(float));
    B1 = (float *)malloc(N * N * sizeof(float));
    for (int i = 0; i < N * N; i++)
        A[i] = rand() / (float)RAND_MAX; // A- random initialization
    for (int i = 0; i < N * N; i++)
        B[i] = 0.0; // initialize B
    for (int i = 0; i < N * N; i++)
        B1[i] = 1.0; // B1 - NxN matrix of ones

    float al = 1.0, bet = 0.0; // coefficients for sgemm
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, al, A, N, B1, N, bet, B, N); // multiply B=A*B1
    cudaStatus = cudaGetDevice(0);
    cusolverStatus = cusolverDnCreate(&handle);
    // prepare memory on the device
    cudaStatus = cudaMalloc((void **)&d_A, N * N * sizeof(float));
    cudaStatus = cudaMalloc((void **)&d_B, N * N * sizeof(float));
    cudaStatus = cudaMalloc((void **)&d_pivot, N * sizeof(int));
    cudaStatus = cudaMalloc((void **)&d_info, sizeof(int));
    cudaStatus = cudaMemcpy(d_A, A, N * N * sizeof(float), cudaMemcpyHostToDevice); // copy d_A <- A
    cudaStatus = cudaMemcpy(d_B, B, N * N * sizeof(float), cudaMemcpyHostToDevice); // copy d_B <- B
    cusolverStatus = cusolverDnSgetrf_bufferSize(handle, N, N, d_A, N, &Lwork);     // compute buffer size and prepare memory
    cudaStatus = cudaMalloc((void **)&d_Work, Lwork * sizeof(float));
    clock_gettime(CLOCK_REALTIME, &start); // timer start
    // LU factorization of d_A, with partial pivoting and row interchanges
    cusolverStatus = cusolverDnSgetrf(handle, N, N, d_A, N, d_Work, d_pivot, d_info);
    // use the LU factorization to solve the system d_A * X = d_B;
    // the solution overwrites d_B
    cusolverStatus = cusolverDnSgetrs(handle, CUBLAS_OP_N, N, N, d_A, N, d_pivot, d_B, N, d_info);
    cudaStatus = cudaDeviceSynchronize();
    clock_gettime(CLOCK_REALTIME, &stop);  // timer stop
    accum = (stop.tv_sec - start.tv_sec) + // elapsed time
            (stop.tv_nsec - start.tv_nsec) / (double)BILLION;
    printf("getrf + getrs time: %lf sec.\n", accum); // print elapsed time
    cudaStatus = cudaMemcpy(&info_gpu, d_info, sizeof(int), cudaMemcpyDeviceToHost); // d_info -> info_gpu
    printf("after getrf + getrs: info_gpu = %d\n", info_gpu);
    cudaStatus = cudaMemcpy(B1, d_B, N * N * sizeof(float), cudaMemcpyDeviceToHost); // copy d_B -> B1
    printf("solution: \n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%g, ", B1[i * N + j]);
        }
        printf("\n");
    }
    printf("\n");

    // free memory
    cudaStatus = cudaFree(d_A);
    cudaStatus = cudaFree(d_B);
    cudaStatus = cudaFree(d_pivot);
    cudaStatus = cudaFree(d_info);
    cudaStatus = cudaFree(d_Work);
    free(A);
    free(B);
    free(B1);
    cusolverStatus = cusolverDnDestroy(handle);
    cudaStatus = cudaDeviceReset();
    return 0;
}
