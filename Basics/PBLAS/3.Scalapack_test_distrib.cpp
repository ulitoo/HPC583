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

int main(int argc, char **argv)
{
    // constants
    char transa = 'N';
    char transb = 'N';
    double alpha = 1.0;
    double beta = 0.0;
    int uno = 1;
    int zero = 0;
    int info, context, nprow, npcol, myrow, mycol, localrows, localcols;
    char tmp[10] = "Col-major";

    // Define the matrix sizes and block sizes
    int N = 10;
    int NB = 2;

    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Get the rank and size of the MPI communicator
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Set up the grid for Scalapack
    Cblacs_pinfo(&rank, &size);
    Cblacs_get(-1, 0, &context);
    // Cblacs_gridinit(&context, tmp, 1, size);

    // Determine the number of processes in each dimension of the grid
    nprow = static_cast<int>(sqrt(size));
    npcol = size / nprow;

    Cblacs_gridinit(&context, tmp, nprow, npcol);
    Cblacs_gridinfo(context, &nprow, &npcol, &myrow, &mycol);

    // Determine local matrix dimensions
    localrows = numroc_(&N, &NB, &myrow, &zero, &nprow);
    localcols = numroc_(&N, &NB, &mycol, &zero, &npcol);

    cout << "\nIn RANK: " << rank << " , nprow:" << nprow << " npcol:" << npcol << ":: Also :: localrows:" << localrows << " and localcols:" << localcols << "\n";

    // Global matrix descriptor
    int descA_1[9];
    // int *descA_1;
    // descinit_(descA_1, &N, &N, &NB, &NB, &nprow, &npcol, &context, &N, &info);

    /*cout << "Info:" << info << "\n";
    for (int j = 0; j < 9; ++j)
    {
        cout << "DescA_1:[" << j <<"]:"<<descA_1[j]<<"\n";
    }
    */

    // Local matrix descriptor
    int descA_local[9], descB_local[9], descC_local[9];
    descinit_(descA_local, &N, &N, &NB, &NB, &zero, &zero, &context, &localrows, &info);
    descinit_(descB_local, &N, &N, &NB, &NB, &zero, &zero, &context, &localrows, &info);
    descinit_(descC_local, &N, &N, &NB, &NB, &zero, &zero, &context, &localrows, &info);

    // Allocate memory for the local matrices
    double *A_local = new double[localrows * localcols];
    double *B_local = new double[localrows * localcols];
    double *C_local = new double[localrows * localcols];

    // Initialize the global matrix on the root process
    if (myrow == 0 && mycol == 0)
    {
        double *A_global = new double[N * N];
        double *B_global = new double[N * N];

        // Initialize your global matrix A here
        // Example: A = identity matrix or ( 1 2 3 ....)
        for (int i = 0; i < N; ++i)
        {
            for (int j = 0; j < N; ++j)
            {
                // A_global[i + N * j] = (i == j) ? 1.0 : 0.0; // Identity
                // A_global[i + N * j] = i * N + j + 1; // 1 2 3 4 ...
                A_global[i + N * j] = 1.0; // 1 1 1 ...
                B_global[i + N * j] = 2.0; // 2 2 2 ...
            }
        }

        // Global matrix descriptor
        int descA_global[9], descB_global[9];
        descinit_(descA_global, &N, &N, &NB, &NB, &zero, &zero, &context, &N, &info);
        descinit_(descB_global, &N, &N, &NB, &NB, &zero, &zero, &context, &N, &info);

        // Distribute the global matrix
        //Cpdgemr2d(N, N, A_global, 1, 1, &descA_global, A_local, 1, 1, descA_local, context);

        // Print Global Matrix A
        PrintColMatrix(A_global, N, N);
        PrintColMatrix(B_global, N, N);
    }

    // Distribute the global Matrices into the different local processors with 2D block Cyclic 
    

    // Initialize the local matrices
    /*
    for (int i = 0; i < localrows * localcols; ++i)
        A_local[i] = 1.0;
    for (int i = 0; i < localrows * localcols; ++i)
        B_local[i] = 2.0;
    for (int i = 0; i < localrows * localcols; ++i)
        C_local[i] = 0.0;
    */
    // Perform the matrix multiplication using pdgemm
    pdgemm_(&transa, &transb, &N, &N, &N, &alpha, A_local, &uno, &uno, descA_local, B_local, &uno, &uno, descB_local, &beta, C_local, &uno, &uno, descC_local);

    cout << "Rank: " << rank << " of Size:" << size << "\n";
    PrintColMatrix(C_local, localrows, localcols);

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
