
// Include the Scalapack headers
#include <mpi.h>
#include <cblas.h>
#include <stdio.h>
#include <scalapack.h>
#include <pblas.h>
#include <Bdef.h>
#include <PBtools.h>
#include <PBblacs.h>
#include <PBpblas.h>
#include <PBblas.h>


int main(int argc, char **argv)
{
    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Get the rank and size of the MPI communicator
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Set up the grid for Scalapack
    int context, nprow, npcol, myrow, mycol;
    char tmp[10] = "Row-major";
    Cblacs_pinfo(&rank, &rank);
    Cblacs_get(-1, 0, &context);
    Cblacs_gridinit(&context, tmp, 1, size);
    Cblacs_gridinfo(context, &nprow, &npcol, &myrow, &mycol);

    // Define the matrix sizes and block sizes
    int m = 4;
    int n = 4;
    int k = 4;
    int mb = 2;
    int nb = 2;
    int descA = m*mb;
    int descB = k*nb;
    int descC = m*nb;

    // Allocate memory for the local matrices
    double *A_local = new double[m * mb];
    double *B_local = new double[k * nb];
    double *C_local = new double[m * nb];

    // Initialize the local matrices (for simplicity, assuming a square matrix)
    for (int i = 0; i < m * mb; ++i)
        A_local[i] = 1.0;
    for (int i = 0; i < k * nb; ++i)
        B_local[i] = 2.0;
    for (int i = 0; i < m * nb; ++i)
        C_local[i] = 0.0;

    // Perform the matrix multiplication using pdgemm
    char transa = 'N';
    char transb = 'N';
    double alpha = 1.0;
    int uno = 1 ;
    double beta = 0.0;

    //pdgemm_(&transa, &transb, &m, &n, &k, &alpha, A_local, &uno, &uno, &(descA), B_local, &uno, &uno, &descB, &beta, C_local, &uno, &uno, &descC);

    // Clean up
    delete[] A_local;
    delete[] B_local;
    delete[] C_local;

    // Finalize MPI
    MPI_Finalize();

    return 0;
}
