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
#include <cuda.h>
#include "magma_lapack.h"
#include "magma_v2.h"

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " n (Dimension of Matrix)" << std::endl;
        return 1;
    }
    //const int N = 3; // std::atoi(argv[1]); // Matrix size (N x N)
    magma_init(); // initialize Magma
    real_Double_t gpu_time;
    magma_int_t *piv, info;
    //magma_int_t m = 8192;                // a - mxm matrix
    //magma_int_t n = 100;                 // c - mxn matrix
    magma_int_t m; // a - mxm matrix
    magma_int_t n; // c - mxn matrix
    m = std::atoi(argv[1]);
    n = std::atoi(argv[1]);
    magma_int_t mm = m * m;              // size of a
    magma_int_t mn = m * n;              // size of c
    float *a;                            // a- mxm matrix on the host
    float *b;                            // b- mxn matrix on the host
    float *c;                            // c- mxn matrix on the host
    magma_int_t ione = 1;                // random uniform distr . in (0 ,1)
    magma_int_t ISEED[4] = {0, 0, 0, 1}; // seed
    magma_int_t err;
    const float alpha = 1.0; // alpha =1
    const float beta = 0.0;  // beta =0

    // allocate matrices on the host
    err = magma_smalloc_pinned(&a, mm); // host memory for a
    err = magma_smalloc_pinned(&b, mn); // host memory for b
    err = magma_smalloc_pinned(&c, mn); // host memory for c
    piv = (magma_int_t *)malloc(m * sizeof(magma_int_t));

    // generate random matrices a, b;
    //lapackf77_slarnv(&ione, ISEED, &mm, a); // randomize a
    LAPACK_slarnv(&ione, ISEED, &mm, a); // randomize a
    //lapackf77_slarnv(&ione, ISEED, &mn, b); // randomize b
    LAPACK_slarnv(&ione, ISEED, &mn, b); // randomize b
    printf(" upper left corner of the expected solution :\n");
    magma_sprint(4, 4, b, m);

    // right hand side c=a*b
    //blasf77_sgemm("N", "N", &m, &n, &n, &alpha, a, &m, b, &m, &beta, c, &m);
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, n, alpha, a, m, b, m, beta, c, m);
    // solve the linear system a*x=c
    // c -mxn matrix , a -mxm matrix ;
    // c is overwritten by the solution
    gpu_time = magma_sync_wtime(NULL);
    magma_sgesv(m, n, a, m, piv, c, m, &info);
    gpu_time = magma_sync_wtime(NULL) - gpu_time;

    printf(" magma_sgesv time : %7.5f sec .\n", gpu_time); // time
    printf(" upper left corner of the solution :\n");
    magma_sprint(4, 4, c, m); // part of the solution

    printf("\n Error : %i.\n", err); // time

    magma_free_pinned(a);     // free host memory
    magma_free_pinned(b);     // free host memory
    magma_free_pinned(c);     // free host memory
    free(piv);        // free host memory
    magma_finalize(); // finalize Magma
    return 0;
}
