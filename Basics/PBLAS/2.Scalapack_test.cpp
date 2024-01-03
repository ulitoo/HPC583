#include <mpi.h>
#include <stdio.h>
#include <iostream>
#include "scalapack.h"

using namespace std;

void PrintColMatrix(double *matrix, int m, int n)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (matrix[i + (j * m)] < 0.00000000000000000001)
            {
                cout << 0.0 << " ";
            }
            else
            {
                cout << matrix[i + (j * m)] << " ";
            }
        }
        cout << "\n";
    }
    cout << "\n";
}

int main(int argc, char **argv) {
    // Initialize MPI

    //cout << "FUERA DE MPI. 1\n";

    MPI_Init(&argc, &argv);


    //cout << "DENTRO DE MPI. 2\n";

    // Get the rank and size of the MPI communicator
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    //cout << "Rank:" << rank << "\n";
    //cout << "Size:" << size << "\n";

    // Set up the grid for Scalapack
    int context, nprow, npcol, myrow, mycol;
    char tmp[10] = "Col-major";
    Cblacs_pinfo(&rank, &size);
    Cblacs_get(-1, 0, &context);
    Cblacs_gridinit(&context, tmp, 1, size);
    Cblacs_gridinfo(context, &nprow, &npcol, &myrow, &mycol);

    // Define the matrix sizes and block sizes
    int m = 4;
    int n = 4;
    int k = 4;
    int mb = 2;
    int nb = 2;

    int descA[9] = {1,context,m,k,mb,nb,0,0,m};
    int descB[9] = {1,context,k,n,mb,nb,0,0,k};
    int descC[9] = {1,context,m,n,mb,nb,0,0,m};

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

    pdgemm_(&transa, &transb, &m, &n, &k, &alpha, A_local, &uno, &uno, descA, B_local, &uno, &uno, descB, &beta, C_local, &uno, &uno, descC);

    cout << "Rank: " << rank << "\n";
    PrintColMatrix(C_local,m,nb);

    // Clean up
    delete[] A_local;
    delete[] B_local;
    delete[] C_local;

    // Finalize MPI
    MPI_Finalize();

    //cout << "FUERA DE MPI. 3\n";
    
    return 0;
}
