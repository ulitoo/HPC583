#include <mpi.h>
#include <pblas.h>
#include <stdio.h>
#include <scalapack.h>

//#include <iostream>
//#include <cblas.h>
//#include <pblas.h>
//#include <Bdef.h>
//#include <PBtools.h>
//#include <PBblacs.h>
//#include <PBpblas.h>
//#include <PBblas.h>

using namespace std;

int main(int argc, char **argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Get the rank and size of the MPI communicator
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    cout << "Rank:" << rank << "\n";
    cout << "Size:" << size << "\n";

    // Set up the grid for Scalapack
    int context, nprow, npcol, myrow, mycol;
    char tmp[10] = "Col-major";
    Cblacs_pinfo(&rank, &size);
    Cblacs_get(-1, 0, &context);
    cout << "context-a:" << context << "\n";
    Cblacs_gridinit(&context, tmp, 1, size);
    cout << "context-b:" << context << "\n";
    Cblacs_gridinfo(context, &nprow, &npcol, &myrow, &mycol);

    //cout << "nprow:" << nprow << "\n";
    //cout << "npcol:" << npcol << "\n";

    cout << "context-c:" << context << "\n";
    
    // Define the matrix sizes and block sizes
    int m = 4;
    int n = 4;
    int k = 4;
    int mb = 2;
    int nb = 2;
    //int descA[9] = {1, 0, m, k, mb, nb, 0, 0, context};
    //int descA[9] = {1, 1, m, k, mb, nb, 1, 1, context};

    int sdescA2[9] = {1, 0, m, k, mb, nb, 0, 0, context};
    int descB[9] = {1, 0, k, n, mb, nb, 0, 0, context};
    int descC[9] = {1, 0, m, n, mb, nb, 0, 0, context};

    // Allocate memory for the local matrices
    double *A_local = new double[m * mb];
    double *B_local = new double[k * nb];
    double *C_local = new double[m * nb];

    // Initialize the local matrices
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
    double beta = 0.0;
    int uno = 1;

    pdgemm_(&transa, &transb, &m, &n, &k, &alpha, A_local, &uno, &uno, sdescA2, B_local, &uno, &uno, descB, &beta, C_local, &uno, &uno, descC);

    // Clean up
    delete[] A_local;
    delete[] B_local;
    delete[] C_local;

    // Finalize MPI
    MPI_Finalize();

    return 0;
}
