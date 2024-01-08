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

    // constants
    char transa = 'N';
    char transb = 'N';
    double alpha = 1.0;
    double beta = 0.0;
    int uno = 1;
    int zero = 0;
    int info, context, nprow, npcol, myrow, mycol,localrows, localcols;
    char tmp[10] = "Col-major";

    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Get the rank and size of the MPI communicator
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Set up the grid for Scalapack
    Cblacs_pinfo(&rank, &size);
    Cblacs_get(-1, 0, &context);
    //Cblacs_gridinit(&context, tmp, 1, size);

    // Determine the number of processes in each dimension of the grid
    nprow = static_cast<int>(sqrt(size));
    npcol = size / nprow;

    Cblacs_gridinit(&context, tmp, nprow, npcol);
    Cblacs_gridinfo(context, &nprow, &npcol, &myrow, &mycol);

    // Define the matrix sizes and block sizes
    int m = 4;
    int n = 4;
    int k = 4;
    int mb = 2;
    int nb = 2;
    int N = 10;
    int NB = 2;

    // Determine local matrix dimensions
    localrows = numroc_(&N, &NB, &myrow, &zero, &nprow);
    localcols = numroc_(&N, &NB, &mycol, &zero, &npcol);

    cout << "\nIn RANK: "<< rank << " , nprow:"<< nprow<<" npcol:" << npcol << ":: Also :: localrows:"<< localrows <<" and localcols:" << localcols << "\n";
    
    MDESC descA_1;
    //descinit_(descA_1, &N, &N, &NB, &NB, &nprow, &npcol, &context, &N, &info);

    int descA_local[9] = {1,context,m,k,mb,nb,0,0,m};
    int descB_local[9] = {1,context,k,n,mb,nb,0,0,k};
    int descC_local[9] = {1,context,m,n,mb,nb,0,0,m};

    // Allocate memory for the local matrices
    // double* A_local = new double[localrows * localcols];
    double *A_local = new double[m * mb];
    double *B_local = new double[k * nb];
    double *C_local = new double[m * nb];
    
    // Initialize the global matrix on the root process
    if (myrow == 0 && mycol == 0)
    {
        double* A_global = new double[N * N];

        // Initialize your global matrix A here
        // Example: A = identity matrix or ( 1 2 3 ....)
        for (int i = 0; i < N; ++i)
        {
            for (int j = 0; j < N; ++j)
            {
                // A_global[i + N * j] = (i == j) ? 1.0 : 0.0; // Identity
                A_global[i + N * j] = i * N + j + 1; // 1 2 3 4 ...
            }
        }

        // Print Global Matrix A
        // PrintColMatrix(A_global,N,N);
    }    

    // Local matrix descriptor
    MDESC descA_local_2;
    //int descA_local_2[9];
    //descinit_(descA_local_2, &N, &N, &NB, &NB, &nprow, &npcol, &ictxt, &localrows, &info);

    ///////////////////////////////////////////////////

    // Initialize the local matrices
    for (int i = 0; i < m * mb; ++i)
        A_local[i] = 1.0;
    for (int i = 0; i < k * nb; ++i)
        B_local[i] = 2.0;
    for (int i = 0; i < m * nb; ++i)
        C_local[i] = 0.0;


    // Perform the matrix multiplication using pdgemm
    pdgemm_(&transa, &transb, &m, &n, &k, &alpha, A_local, &uno, &uno, descA_local, B_local, &uno, &uno, descB_local, &beta, C_local, &uno, &uno, descC_local);

    cout << "Rank: " << rank << " of Size:" << size << "\n";
    PrintColMatrix(C_local,m,nb);

    // Deallocate memory and finalize BLACS
    // Clean up
    delete[] A_local;
    delete[] B_local;
    delete[] C_local;

    Cblacs_gridexit(context);
   
    // Finalize MPI
    MPI_Finalize();

    return 0;
}
