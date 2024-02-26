#include <mpi.h>
#include <stdio.h>
#include <iostream>
#include <cblas.h>
#include <cmath>
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

void printMatrix(double *matrix, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            std::cout << matrix[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Initialize BLACS context
    int context;
    Cblacs_pinfo(&rank, &size);
    Cblacs_get(-1, 0, &context);

    // Define matrix size and block size
    int N = 4;
    int NB = 2;
    char ColMajor[10] = "Col-major";

    // Number of rows and columns in the process grid
    int nprow = 2;
    int npcol = 2;

    // Process grid coordinates
    int myrow, mycol;
    Cblacs_gridinit(&context, ColMajor, nprow, npcol);
    Cblacs_gridinfo(context, &nprow, &npcol, &myrow, &mycol);

    // Local matrix dimensions
    int localrows = numroc_(&N, &NB, &myrow, &myrow, &nprow);
    int localcols = numroc_(&N, &NB, &mycol, &mycol, &npcol);

    // Allocate memory for the local portion of the matrix
    double *A_local = new double[localrows * localcols];

    // Allocate memory for the global matrix on root process

    double *A_global = nullptr;
    A_global = new double[N * N];
    // Initialize the global matrix (for demonstration)
    for (int i = 0; i < N * N; ++i)
    {
        A_global[i] = i + 1; // Each element is set to its index + 1
    }
    std::cout << "Global Matrix:" << std::endl;
    PrintColMatrix(A_global, N, N);
    // Scatter the global matrix to the grid
    Cdgesd2d(context, N, N, A_global, N, 0, 0);

    cout << "\nIn RANK: " << rank << " , nprow:" << nprow << " npcol:" << npcol << ":: Also :: localrows:" << localrows << " and localcols:" << localcols << " myrow:" << myrow << ", mycol:" << mycol << " \n";

    // Gather the local portion of the matrix back to root process (for verification)

    Cdgerv2d(context, localrows, localcols, A_local, localrows, 0, 0);
    std::cout << "Scattered Matrix:" << std::endl;
    PrintColMatrix(A_local, localrows, localcols);

    // Clean up
    delete[] A_global;
    delete[] A_local;
    Cblacs_gridexit(context);
    MPI_Finalize();

    return 0;
}
