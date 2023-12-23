#include <iostream>
#include <cstdlib>
#include <cmath>
//#include <scalapack.h>
#include <mpi.h>
#include <pblas.h>
#include <PBtools.h>
#include <PBblacs.h>
#include <PBpblas.h>
#include <PBblas.h>

extern "C" {
    // Function to initialize BLACS context
    void blacs_pinfo(int*, int*);
    void blacs_get(int, int, int*);
    void blacs_gridinit(int*, const char*, int, int);
    void blacs_gridinfo(int, int*, int*, int*, int*);
    void blacs_gridexit(int);
    void blacs_exit(int);
    // Function to initialize Scalapack descriptor
    void descinit_(int*, int*, int*, int*, int*, int*, int*, int*, int*, int*);
}

int main() {
    // MPI Initialization
    MPI_Init(NULL, NULL);
    int mpiRank, mpiSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);

    // BLACS Initialization
    int blacsContext, blacsRank, blacsSize;
    blacs_pinfo(&blacsRank, &blacsSize);
    blacs_get(0, 0, &blacsContext);
    blacs_gridinit(&blacsContext, "Row", 1, blacsSize);

    // Matrix Size
    int n = 1000;
    int nb = 128;
    int npRow = 2;
    int npCol = 2;

    // Local Matrix Size
    int localN = numroc_(&n, &nb, &blacsRank, &ZERO, &npCol);
    int localM = numroc_(&n, &nb, &blacsRank, &ZERO, &npRow);

    // Allocate Local Matrices
    double* A = new double[localM * localN];
    double* B = new double[localN * localM];
    double* C = new double[localM * localN];

    // Initialize Local Matrices
    for (int i = 0; i < localM * localN; ++i) {
        A[i] = 1.0;
        B[i] = 2.0;
        C[i] = 0.0;
    }

    // Initialize Scalapack Descriptors
    int descA[9], descB[9], descC[9];
    descinit_(descA, &n, &n, &nb, &nb, &ZERO, &ZERO, &blacsContext, &localM, &info);
    descinit_(descB, &n, &n, &nb, &nb, &ZERO, &ZERO, &blacsContext, &localN, &info);
    descinit_(descC, &n, &n, &nb, &nb, &ZERO, &ZERO, &blacsContext, &localM, &info);

    // Perform Matrix-Matrix Multiplication C = A * B
    pdgemm_("N", "N", &n, &n, &n, &ONE, A, &ONE, &ONE, descA, B, &ONE, &ONE, descB, &ZERO, C, &ONE, &ONE, descC);

    // Deallocate Matrices
    delete[] A;
    delete[] B;
    delete[] C;

    // Release BLACS Context
    blacs_gridexit(blacsContext);
    blacs_exit(ZERO);

    // Finalize MPI
    MPI_Finalize();

    return 0;
}
