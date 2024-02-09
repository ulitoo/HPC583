#include <cblas.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#define N 8192
#define BILLION 1000000000L;
using namespace std;

int main(int argc, char *argv[])
{
    struct timespec start, stop; // variables for timing
    double accum;                // elapsed time variable
    cublasStatus_t stat;
    cudaError cudaStatus;
    cusolverStatus_t cusolverStatus;
    cusolverDnHandle_t handle;

    // declare arrays on the host
    float *A, *B1, *B; // A - NxN matrix , B1 - auxiliary N- vect .
    // B=A*B1 - N- vector of rhs , all on the host
    // declare arrays on the device
    float *d_A, *d_B, *d_Work;    // coeff .matrix , rhs , workspace
    int *d_pivot, *d_info, Lwork; // pivots , info , worksp . size
    int info_gpu = 0;

    // prepare memory on the host
    A = (float *)malloc(N * N * sizeof(float));
    B = (float *)malloc(N * sizeof(float));
    B1 = (float *)malloc(N * sizeof(float));
    for (int i = 0; i < N * N; i++)
        A[i] = rand() / (float)RAND_MAX; // A- rand
    for (int i = 0; i < N; i++)
        B[i] = 0.0; // initialize B
    for (int i = 0; i < N; i++)
        B1[i] = 1.0; // B1 - N- vector of ones

    float al = 1.0, bet = 0.0; // coefficients for sgemv
    int incx = 1, incy = 1;
    cblas_sgemv(CblasColMajor, CblasNoTrans, N, N, al, A, N, B1, incx, bet, B, incy); // multiply B=A*B1
    cudaStatus = cudaGetDevice(0);
    cusolverStatus = cusolverDnCreate(&handle);
    // prepare memory on the device
    cudaStatus = cudaMalloc((void **)&d_A, N * N * sizeof(float));
    cudaStatus = cudaMalloc((void **)&d_B, N * sizeof(float));
    cudaStatus = cudaMalloc((void **)&d_pivot, N * sizeof(int));
    cudaStatus = cudaMalloc((void **)&d_info, sizeof(int));
    cudaStatus = cudaMemcpy(d_A, A, N * N * sizeof(float), cudaMemcpyHostToDevice); // copy d_A < -A
    cudaStatus = cudaMemcpy(d_B, B, N * sizeof(float), cudaMemcpyHostToDevice);     // copy d_B < -B
    cusolverStatus = cusolverDnSgetrf_bufferSize(handle, N, N, d_A, N, &Lwork);     // compute buffer size and prep . memory
    cudaStatus = cudaMalloc((void **)&d_Work, Lwork * sizeof(float));
    clock_gettime(CLOCK_REALTIME, &start); // timer start
    // LU factorization of d_A , with partial pivoting and row
    // interchanges ; row i is interchanged with row d_pivot (i);
    cusolverStatus = cusolverDnSgetrf(handle, N, N, d_A, N, d_Work,
                                      d_pivot, d_info);
    // use the LU factorization to solve the system d_A *x=d_B ;
    // the solution overwrites d_B
    cusolverStatus = cusolverDnSgetrs(handle, CUBLAS_OP_N, N, 1,
                                      d_A, N, d_pivot, d_B, N, d_info);
    cudaStatus = cudaDeviceSynchronize();
    clock_gettime(CLOCK_REALTIME, &stop);  // timer stop
    accum = (stop.tv_sec - start.tv_sec) + // elapsed time
            (stop.tv_nsec - start.tv_nsec) / (double)BILLION;
    printf(" getrf + getrs time : %lf sec .\n", accum); // print el. time
    cudaStatus = cudaMemcpy(&info_gpu, d_info, sizeof(int),
                            cudaMemcpyDeviceToHost); // d_info -> info_gpu
    printf(" after getrf + getrs : info_gpu = %d\n", info_gpu);
    cudaStatus = cudaMemcpy(B1, d_B, N * sizeof(float),
                            cudaMemcpyDeviceToHost); // copy d_B - >B1
    printf(" solution : ");
    for (int i = 0; i < 5; i++)
        printf("%g, ", B1[i]);
    printf(" ... "); // print first components of the solution
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
// getrf + getrs time : 0.267574 sec .
// after getrf + getrs : info_gpu = 0
// solution : 1.04225 , 0.873826 , 1.05703 , 1.03822 , 0.883831