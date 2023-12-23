#include <iostream>
#include <vector>
#include <mpi.h>
//#include <scalapack.h>

#include <pblas.h>
#include <PBtools.h>
#include <PBblacs.h>
#include <PBpblas.h>
#include <PBblas.h>

extern "C" {
  // ScaLAPACK pdgemm function declaration
  void pdgemm_(const char*, const char*, const int*, const int*, const int*,
               const double*, const double*, const int*, const int*, const int*,
               const double*, const double*, const int*, const int*, const int*,
               const double*, double*, const int*, const int*, const int*);
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int myrank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int nprow = 2;
    int npcol = nprocs / nprow;
    int myrow, mycol;
    int ictxt;

    // Initialize BLACS context
    Cblacs_pinfo(&myrank, &nprocs);
    Cblacs_get(-1, 0, &ictxt);
    Cblacs_gridinit(&ictxt, "Row", nprow, npcol);
    Cblacs_gridinfo(ictxt, &nprow, &npcol, &myrow, &mycol);

    // Matrix dimensions
    int m = 100;
    int n = 100;
    int k = 100;
    int nb = 64;

    // Array descriptors
    int descA[9], descB[9], descC[9];
    descinit_(descA, &m, &k, &nb, &nb, &myrow, &mycol, &ictxt, &m, &info);
    descinit_(descB, &k, &n, &nb, &nb, &myrow, &mycol, &ictxt, &k, &info);
    descinit_(descC, &m, &n, &nb, &nb, &myrow, &mycol, &ictxt, &m, &info);

    // Allocate and initialize matrices A, B, and C
    std::vector<double> A(m * k, 1.0);
    std::vector<double> B(k * n, 2.0);
    std::vector<double> C(m * n, 0.0);

    // Perform matrix multiplication: C = alpha * A * B + beta * C
    double alpha = 1.0;
    double beta = 0.0;
    pdgemm_("N", "N", &m, &n, &k, &alpha, A.data(), &descA[8], B.data(), &descB[8],
            &beta, C.data(), &descC[8]);

    // Deallocate matrices
    A.clear();
    B.clear();
    C.clear();

    // Finalize MPI and BLACS
    Cblacs_gridexit(ictxt);
    MPI_Finalize();

    return 0;
}
